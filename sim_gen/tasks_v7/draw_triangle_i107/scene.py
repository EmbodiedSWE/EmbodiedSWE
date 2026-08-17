"""BeamBalanceScene — weigh an unknown stone on a two-pan beam balance: load brass
unit weights into the hanging trays until the beam floats level, sustained-still and
hands-off (sim_gen task `draw_triangle_i107`).

Derived from maniskill/draw_triangle, but STRATEGICALLY different: the seed's Franka
holds a rigid stylus and TRACES a triangle outline on a canvas — one long, guarded,
continuous tool-tip sweep along a PRESCRIBED curve, judged on the coverage of the
trace the arm itself generated; the scene is passive and nothing about the goal is
hidden. Here there is no path, no held tool and no trace: the scene is an ARTICULATED
MECHANISM (a keeled beam balance on a revolute pivot with hard stops, carrying two
pendant tray pans) holding a HIDDEN QUANTITY (the slate stone's mass — a randomized
whole multiple of the brass unit, unreadable from its fixed size), and the judged
product is a PHYSICAL EQUILIBRIUM the solver must construct: the beam floats level
only when the moments genuinely cancel. A solver needs a different PLAN (infer the
hidden mass from the mechanism's response — the beam tips toward the heavier side —
via discrete place/observe/revise trials with the labeled 1/2/4-unit weights, then
leave the exact combination seated) and a different CODE STRUCTURE (closed-loop
subset search over discrete masses, settle-and-read iterations), not waypoint
tracking. Physics itself enforces the arithmetic: the keel is sized so a single-unit
error drives the equilibrium tilt PAST the hard stops (asserted), so every wrong
combination rests on a stop at 12 deg while the correct one floats level near 0 deg.
Against the tasks_v7 corpus (surveyed before design: aperture posting, frame
construction, containment/stacking/pouring transport, articulated open/close,
pendulum arrest...): no existing task hides a scalar the solver must MEASURE through
the scene's static response, and none judges a balanced equilibrium of a mechanism.

The scene (fully procedural, no external assets):
  - a light-gray kinematic BENCH slab (1.10 x 1.00 x 0.10 m, top at 0.10 m);
  - a BALANCE: a heavy dynamic STAND (foot slab + two fork posts + top crossbar;
    dynamic, not kinematic — a joint anchored to a teleported kinematic body0 stays
    world-fixed at spawn on this stack) carrying a BEAM on a revolute pivot (axis =
    stand-local Y, hard joint stops +/- `stop_deg`) with a keel bob hanging under
    the pivot (authored CoM below the pivot -> restoring torque; with MassAPI-only
    mass PhysX would leave the CoM AT the pivot and the balance would be neutral —
    forge-verified quirk), and TWO PENDANT TRAY PANS hung from the beam ends on
    their own revolute hinges: the trays stay level as the beam tilts, so the moment
    of a load is set by the HANG POINT, not by where in the tray it sits — placement
    position cannot fake or spoil a balance;
  - a slate STONE (50 mm cube) resting in one tray (side randomized): its mass is a
    randomized whole multiple (1..7) of the 60 g brass unit, realized by spawning
    seven identical-LOOKING stones of masses 1u..7u and placing the sampled one in
    the tray (per-episode mass randomization without runtime mass writes); the six
    spares wait in a floor depot far outside the exclusion radius;
  - three BRASS WEIGHTS staged on the bench front (slot permutation + jitter + free
    yaw): square blocks of 1, 2 and 4 units (60/120/240 g) whose footprints scale
    with mass (equal density), so the labels are visual.

success() (all judged live, simultaneously):
  A. LEVEL    — beam axis within `level_tol_deg` of horizontal;
  B. STILL    — SUSTAINED rest: beam angular speed and every pan/stone/weight linear
                speed below threshold for `still_steps` consecutive sim steps (a
                beam swinging through level is FAST at level — the sustained gate
                rejects fly-through readings);
  C. STONE HOME — the active stone still seated in the tray it spawned in (tray
                frame; removing the stone would let the empty balance level itself);
  D. CLEAN    — every brass weight either seated in one of the two trays or farther
                than `exclusion_r` from the stand (no weight parked on the beam, the
                stand or loose under the scale); the six spare stones all farther
                than `exclusion_r` from the stand (swapping a look-alike spare into
                the trays is declared out of bounds);
  E. STAND HOME — the stand within `home_xy_tol` / `home_tilt_deg` of its spawn pose
                (tipping or dragging the scale to fake "level" is rejected).
Given C/D/E and the geometry (tray floors hang far above any stack of every movable
in the scene — asserted), the only way to hold clause A+B hands-off is a genuine
moment balance: stone mass = net brass moment, unit-exact.

score() (latched in post_step, non-decreasing): 0.15 once at least one brass weight
has ever been SEATED (settled in a tray, stone home, stand home) + 0.10 once two
brass weights have ever been seated simultaneously, capped at 0.25; exactly 1.0 iff
success() now. The null policy scores ~0: the stone parks the beam on a stop and no
weight is ever seated.

Per-episode randomization (readback-verifiable): stone mass (1..7 units, via the
active-stone index), stone side, stand xy + yaw jitter, weight slot permutation +
xy jitter + free yaw. All discrete draws use torch.rand comparisons (the first
torch.randint after manual_seed is near-constant across seeds on this stack).

No execution order is required (weights may be seated in any order, in either tray:
counter-loading and differential loading both balance — the arithmetic, not the
order, is judged). Heavy imports (isaaclab, pxr) are deferred so importing this
module — and registering the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw], dim=-1)


# ----- custom compound spawners (stand / beam / pan) --------------------------------------------
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


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 32, com: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass, damping, no
    sleeping while velocities are judged, depenetration cap. `com` authors an
    explicit local centre of mass — REQUIRED for the beam: with only a mass on the
    root, PhysX keeps the CoM at the body ORIGIN (here the pivot), which erases the
    keel's restoring torque and the balance never rights (forge-verified quirk)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_mat(stage_path: str, mat_path: str, static: float, dynamic: float,
              restitution: float = 0.0) -> None:
    """Author (once) and bind a physics material (custom spawner colliders
    otherwise get the ~0.5-friction default with no restitution control)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=restitution))
    bind_physics_material(stage_path, mat_path)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the balance STAND: heavy DYNAMIC compound (dynamic so its beam joint
    follows reset teleports). Local origin: footprint centre at bench-top level.
    A foot slab, two fork posts at local y = +/- `post_y` (the beam swings in the
    x-z plane between them) and a top crossbar above the beam."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.stand_mass, lin_damp=0.5, ang_damp=0.5, iters=16,
                   com=(0.0, 0.0, 0.03))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/foot", center=(0.0, 0.0, c.foot_h / 2),
             size=(c.foot_x, c.foot_x, c.foot_h), color=c.stand_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/post_{tag}",
                 center=(0.0, sgn * c.post_y, c.foot_h + (c.post_top - c.foot_h) / 2),
                 size=(c.post_w, c.post_w, c.post_top - c.foot_h),
                 color=c.stand_color, collide=collide)
    _add_box(stage, f"{prim_path}/crossbar",
             center=(0.0, 0.0, c.post_top + c.post_w / 2),
             size=(c.post_w, 2 * c.post_y + c.post_w, c.post_w),
             color=c.stand_color, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the BEAM: dynamic compound whose origin IS the pivot. Children: the
    crossbar along local x and the keel (thin rod + bob) hanging below the pivot.
    Plus the REVOLUTE pivot joint to the sibling stand (axis Y, hard limits
    +/- `stop_deg` — the stops every wrong load rests on). The joint pair is
    collision-filtered by USD default (the bar passes between the fork posts)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    # Explicit CoM from the keel (bar is symmetric about the pivot): the restoring
    # torque coefficient beam_mass * |com_z| is the whole calibration — see the
    # SceneCfg assertions.
    _rigid_dynamic(root, c.beam_mass, lin_damp=0.2, ang_damp=c.beam_ang_damp,
                   com=(0.0, 0.0, -c.keel_com))
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
             size=(2 * c.arm_x + 0.04, c.bar_w, c.bar_w), color=c.beam_color,
             collide=collide)
    _add_box(stage, f"{prim_path}/keel_rod", center=(0.0, 0.0, -c.keel_len / 2),
             size=(0.016, 0.016, c.keel_len), color=c.beam_color, collide=collide)
    _add_box(stage, f"{prim_path}/keel_bob", center=(0.0, 0.0, -c.keel_len),
             size=(0.045, 0.030, 0.045), color=c.keel_color, collide=collide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.stop_deg))
    j.CreateUpperLimitAttr(float(c.stop_deg))
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one PENDANT TRAY PAN: dynamic compound whose origin IS its hang
    hinge. A thin stem drops to a square tray (floor plate + four rim walls).
    Plus the REVOLUTE hang joint to the sibling beam at beam-local (arm_x, 0, 0),
    axis Y: the tray stays level as the beam tilts, so a load's moment is set by
    the hang point — in-tray placement position cannot change the balance."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.pan_mass, lin_damp=0.3, ang_damp=c.pan_ang_damp,
                   com=(0.0, 0.0, -c.stem_len + 0.01))
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/stem", center=(0.0, 0.0, -c.stem_len / 2),
             size=(0.012, 0.012, c.stem_len), color=c.pan_color, collide=collide)
    half = c.tray_in + c.wall_t
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, -c.stem_len - c.floor_t / 2),
             size=(2 * half, 2 * half, c.floor_t), color=c.pan_color, collide=collide)
    for dx, dy, tag in ((1, 0, "e"), (-1, 0, "w"), (0, 1, "n"), (0, -1, "s")):
        sx = (c.wall_t, 2 * half, c.wall_h) if dx else (2 * half, c.wall_t, c.wall_h)
        _add_box(stage, f"{prim_path}/wall_{tag}",
                 center=(dx * (c.tray_in + c.wall_t / 2), dy * (c.tray_in + c.wall_t / 2),
                         -c.stem_len + c.wall_h / 2),
                 size=sx, color=c.pan_color, collide=collide)
    _bind_mat(f"{prim_path}/floor", "/World/simgenBalTrayMat", 0.85, 0.75, 0.0)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hang")
    j.CreateBody0Rel().SetTargets([f"{base}/Beam"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.arm_sign * c.arm_x), 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-60.0)
    j.CreateUpperLimitAttr(60.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            stand_mass: float = 26.0
            foot_x: float = 0.16
            foot_h: float = 0.030
            post_y: float = 0.06
            post_w: float = 0.040
            post_top: float = 0.46
            pivot_h: float = 0.40
            stand_color: tuple = (0.30, 0.30, 0.34)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            beam_mass: float = 0.60
            beam_ang_damp: float = 3.0
            arm_x: float = 0.26
            bar_w: float = 0.030
            keel_len: float = 0.165
            keel_com: float = 0.087
            pivot_h: float = 0.40
            stop_deg: float = 12.0
            beam_color: tuple = (0.55, 0.35, 0.14)
            keel_color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.002

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            pan_mass: float = 0.08
            pan_ang_damp: float = 6.0
            arm_x: float = 0.26
            arm_sign: float = 1.0
            stem_len: float = 0.10
            tray_in: float = 0.0475
            wall_t: float = 0.007
            wall_h: float = 0.016
            floor_t: float = 0.008
            pan_color: tuple = (0.78, 0.78, 0.82)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["pan"] = PanSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BeamBalanceSceneCfg(BaseCfg):
    """Config for `BeamBalanceScene`. The honesty calibration is asserted in
    `__post_init__`: a single-unit error must drive the equilibrium past the hard
    stops, and no stack of every movable object can reach a hanging tray floor."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    level_tol_deg: float = tunable(4.0)      # |beam elevation| below this = level
    beam_still_avel: float = tunable(0.10)   # max beam |ang vel| when judging (rad/s)
    # Max pan/stone/weight |lin vel| when judging. Sits ABOVE the measured GPU
    # contact-jitter floor of a loaded pendant pan (~0.06 m/s persistent limit
    # cycle, forge-measured) but far below any real motion; fly-through "level"
    # is rejected by the BEAM gate (crossing the +/-4 deg window within the
    # 0.5 s stillness latch needs >= 0.28 rad/s >> beam_still_avel).
    obj_still_speed: float = tunable(0.09)
    still_steps: int = tunable(60)           # consecutive sub-threshold steps (0.5 s)
    exclusion_r: float = tunable(0.42)       # loose weights/spare stones must exceed this
    home_xy_tol: float = tunable(0.04)       # stand drift tolerance (clause E)
    home_tilt_deg: float = tunable(10.0)     # stand uprightness tolerance (clause E)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    units_max: int = tunable(7)              # stone mass ~ U{1..units_max} units
    stand_jitter: float = tunable(0.03)      # stand xy jitter (+/- m)
    stand_yaw_deg: float = tunable(10.0)     # stand yaw jitter (+/- deg)
    slot_shuffle: bool = tunable(True)       # permute which slot holds which weight
    slot_jitter: float = tunable(0.02)       # weight xy jitter (+/- m)
    slot_yaw_deg: float = tunable(180.0)     # weight free yaw (+/- deg)

    # --- tunable: placement (bench frame) ----------------------------------------------------
    stand_xy: tuple = tunable((0.0, 0.08))   # stand centre
    slot_xy: tuple = tunable(((-0.20, -0.435), (0.0, -0.455), (0.20, -0.435)))  # weight slots

    # --- info: bench -------------------------------------------------------------------------
    bench_size: tuple = info((1.10, 1.00, 0.10))
    bench_top: float = info(0.10)
    bench_color: tuple = info((0.75, 0.75, 0.78))
    # --- info: balance geometry (see spawners) -----------------------------------------------
    pivot_h: float = info(0.40)              # pivot height above the bench top
    post_top: float = info(0.46)             # fork-post top (crossbar sits above)
    arm_x: float = info(0.26)                # hang-point half-span
    stop_deg: float = info(12.0)             # hard joint stops
    beam_mass: float = info(0.60)
    keel_com: float = info(0.087)            # authored |CoM depth| below the pivot
    beam_ang_damp: float = info(3.0)         # settles the level float in a few swings
    stem_len: float = info(0.10)             # tray floor hangs this far below the hinge
    tray_in: float = info(0.0475)            # tray inner half-width
    wall_h: float = info(0.016)
    floor_t: float = info(0.008)
    stand_mass: float = info(26.0)
    foot_x: float = info(0.16)
    post_y: float = info(0.06)
    post_w: float = info(0.040)
    # --- info: loads -------------------------------------------------------------------------
    unit_mass: float = info(0.060)           # one unit = 60 g
    # brass weights (name, units, square footprint half-width, height): footprint
    # scales with mass at equal density -> the labels are visual.
    weight_specs: tuple = info((("w1", 1, 0.011, 0.035),
                                ("w2", 2, 0.0155, 0.035),
                                ("w4", 4, 0.022, 0.035)))
    weight_color: tuple = info((0.72, 0.58, 0.25))
    stone_size: float = info(0.046)          # slate cube edge (same for all 7 stones)
    stone_color: tuple = info((0.42, 0.40, 0.50))
    depot_xy: tuple = info((0.95, 0.75))     # spare-stone floor depot (on the ground)
    depot_dx: float = info(0.14)
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.10 = the 0.25 non-success cap)
    w_seat1: float = info(0.15)
    w_seat2: float = info(0.10)

    # Derived (filled in __post_init__).
    tray_floor_z: float = field(default=None, init=False)  # tray floor above bench, level

    def __post_init__(self) -> None:
        g_arm = self.unit_mass * self.arm_x            # one-unit moment / g
        restore = self.beam_mass * self.keel_com       # restoring coefficient / g
        # a single-unit error must drive the equilibrium past the hard stop, so every
        # wrong combination rests visibly on a stop ...
        assert math.tan(math.radians(self.stop_deg)) < g_arm / restore / 1.25, \
            "one-unit imbalance must rest on the stop with margin"
        # ... and the stop must sit far outside the level tolerance
        assert self.stop_deg > self.level_tol_deg + 6.0
        # tray floors hang above any stack of every movable object (3 weights + the
        # active stone + a spare could in principle be piled): propping a tray from
        # the bench is geometrically impossible, even at full stop droop
        stack = sum(h for _n, _u, _hw, h in self.weight_specs) + 2 * self.stone_size
        droop = self.pivot_h - self.stem_len - self.floor_t \
            - self.arm_x * math.sin(math.radians(self.stop_deg))
        assert droop > stack + 0.03, "a pile of every movable must not reach a tray floor"
        # weight slots stay outside the exclusion radius under worst-case jitter
        sx0, sy0 = self.stand_xy
        for sx, sy in self.slot_xy:
            d = math.hypot(sx - sx0, sy - sy0) - self.slot_jitter - self.stand_jitter
            assert d > self.exclusion_r + 0.02, "weight slots must clear the exclusion"
            hw = max(hw for _n, _u, hw, _h in self.weight_specs)
            assert abs(sx) + hw + self.slot_jitter < self.bench_size[0] / 2
            assert abs(sy) + hw + self.slot_jitter < self.bench_size[1] / 2
        # the spare-stone depot is off the bench and far outside the exclusion
        dx, dy = self.depot_xy
        assert abs(dx) > self.bench_size[0] / 2 + self.stone_size, "depot must be off-bench"
        assert math.hypot(dx - sx0, dy - sy0) > self.exclusion_r + 0.25
        # everything the arm must pinch fits the Franka jaw
        assert all(2 * hw < 0.078 for _n, _u, hw, _h in self.weight_specs)
        # the stone (never grasped, but keep it plausible) and all three weights fit
        # in one tray footprint: 4u + 2u side by side, 1u in the spare row
        assert 2 * (0.022 + 0.0155) + 0.004 < 2 * self.tray_in
        assert self.stone_size * math.sqrt(2.0) < 2 * (self.tray_in + 0.007) + 0.04
        # the beam bar passes between the fork posts with clearance
        assert self.post_y - self.post_w / 2 > 0.030 / 2 + 0.015
        self.tray_floor_z = self.pivot_h - self.stem_len


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("beam_balance")
class BeamBalanceScene(BaseScene):
    cfg: BeamBalanceSceneCfg

    def __init__(self, cfg: BeamBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or BeamBalanceSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.bench_top
        sp = _spawner_classes()
        bench_mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.60, dynamic_friction=0.50,
                                                   restitution=0.0)
        load_mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.80, dynamic_friction=0.70,
                                                  restitution=0.0)
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
            "bench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=c.bench_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=bench_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.bench_size[2] / 2)),
            ),
            # spawn order matters: each joint's body0 must already exist
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=sp["stand"](stand_mass=c.stand_mass, foot_x=c.foot_x,
                                  post_y=c.post_y, post_w=c.post_w, post_top=c.post_top,
                                  pivot_h=c.pivot_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_xy[0], c.stand_xy[1], z0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=sp["beam"](beam_mass=c.beam_mass, beam_ang_damp=c.beam_ang_damp,
                                 arm_x=c.arm_x, keel_com=c.keel_com, pivot_h=c.pivot_h,
                                 stop_deg=c.stop_deg, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_xy[0], c.stand_xy[1], z0 + c.pivot_h)),
            ),
        }
        for sgn, tag in ((1.0, "P"), (-1.0, "N")):
            out[f"pan_{tag.lower()}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pan" + tag,
                spawn=sp["pan"](arm_x=c.arm_x, arm_sign=sgn, stem_len=c.stem_len,
                                tray_in=c.tray_in, wall_h=c.wall_h, floor_t=c.floor_t,
                                contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_xy[0] + sgn * c.arm_x, c.stand_xy[1], z0 + c.pivot_h)),
            )
        for i in range(c.units_max):
            out[f"stone_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stone_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.stone_size,) * 3,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        linear_damping=0.30, angular_damping=1.50),
                    mass_props=sim_utils.MassPropertiesCfg(mass=(i + 1) * c.unit_mass),
                    physics_material=load_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.stone_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_xy[0] + (i % 4) * c.depot_dx,
                         c.depot_xy[1] + (i // 4) * c.depot_dx,
                         c.stone_size / 2 + 0.003)),
            )
        for j, (name, _u, hw, h) in enumerate(c.weight_specs):
            out[f"weight_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Weight_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(2 * hw, 2 * hw, h),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        linear_damping=0.30, angular_damping=1.50),
                    mass_props=sim_utils.MassPropertiesCfg(mass=_u * c.unit_mass),
                    physics_material=load_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.weight_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xy[j][0], c.slot_xy[j][1], c.bench_top + h / 2 + 0.003)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.pans: list[RigidObject] = [env.iscene["pan_p"], env.iscene["pan_n"]]
        self.stones: list[RigidObject] = [
            env.iscene[f"stone_{i}"] for i in range(c.units_max)]
        self.weights: list[RigidObject] = [
            env.iscene[f"weight_{nm}"] for nm, _u, _hw, _h in c.weight_specs]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.stand_home_xy = torch.zeros(n, 2, device=dev)
        self.stand_yaw = torch.zeros(n, device=dev)
        self.active_idx = torch.zeros(n, dtype=torch.long, device=dev)  # stone index = units-1
        self.side_idx = torch.zeros(n, dtype=torch.long, device=dev)   # 0 = PanP (+x), 1 = PanN
        self.seat1_latch = torch.zeros(n, device=dev)
        self.seat2_latch = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the WHOLE linkage coherently (stand jittered +
        yawed; beam level at the pivot; both trays hanging level at the beam ends —
        teleporting only part of a linkage gets depenetrated back), seat the
        sampled stone just above its randomized tray, park the six spares in the
        floor depot, and stage the weights on shuffled, jittered slots. The stone
        then tips the beam onto a stop during the first second of physics."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.bench_top

        def write(body, xy: torch.Tensor, z: torch.Tensor | float, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- discrete draws via torch.rand (first randint after manual_seed is
        # near-constant across seeds on this stack) ---
        units = (torch.rand(m, device=dev) * c.units_max).long().clamp(0, c.units_max - 1)
        side = (torch.rand(m, device=dev) < 0.5).long()          # 0 = PanP(+x), 1 = PanN(-x)
        self.active_idx[env_ids] = units                          # stone i has mass (i+1) u
        self.side_idx[env_ids] = side

        # --- stand + beam + pans: one coherent linkage pose ---
        sxy = torch.tensor(c.stand_xy, device=dev).expand(m, 2).clone()
        sxy = sxy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        q = _qz(yaw)
        write(self.stand, sxy, z0, q)
        write(self.beam, sxy, z0 + c.pivot_h, q)
        ca, sa = torch.cos(yaw), torch.sin(yaw)
        for k, sgn in enumerate((1.0, -1.0)):
            pxy = sxy + torch.stack([sgn * c.arm_x * ca, sgn * c.arm_x * sa], dim=-1)
            write(self.pans[k], pxy, z0 + c.pivot_h, q)
        self.stand_home_xy[env_ids] = sxy
        self.stand_yaw[env_ids] = yaw

        # --- stones: the sampled one just above its tray floor, spares in the depot ---
        sgn_side = torch.where(side == 0, torch.ones(m, device=dev), -torch.ones(m, device=dev))
        tray_xy = sxy + torch.stack([sgn_side * c.arm_x * ca, sgn_side * c.arm_x * sa], dim=-1)
        stone_z = z0 + c.tray_floor_z + c.stone_size / 2 + 0.006
        for i, body in enumerate(self.stones):
            depot = torch.zeros(m, 2, device=dev)
            depot[:, 0] = c.depot_xy[0] + (i % 4) * c.depot_dx
            depot[:, 1] = c.depot_xy[1] + (i // 4) * c.depot_dx
            act = (units == i).unsqueeze(-1)
            xy = torch.where(act, tray_xy, depot)
            z = torch.where(act.squeeze(-1),
                            torch.full((m,), stone_z, device=dev),
                            torch.full((m,), c.stone_size / 2 + 0.003, device=dev))
            write(body, xy, z, torch.where(act, q, _qz(torch.zeros(m, device=dev))))

        # --- weights: shuffled slots + jitter + free yaw ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1) if c.slot_shuffle \
            else torch.arange(3, device=dev).unsqueeze(0).expand(m, 3)
        slots = torch.tensor(list(c.slot_xy), device=dev)
        for j, ((_nm, _u, _hw, h), body) in enumerate(zip(c.weight_specs, self.weights)):
            xy = slots[perm[:, j]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            wyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.slot_yaw_deg)
            write(body, xy, z0 + h / 2 + 0.003, _qz(wyaw))

        self.seat1_latch[env_ids] = 0.0
        self.seat2_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "pans": [b.data.root_state_w[env_ids].clone() for b in self.pans],
            "stones": [b.data.root_state_w[env_ids].clone() for b in self.stones],
            "weights": [b.data.root_state_w[env_ids].clone() for b in self.weights],
            "stand_home_xy": self.stand_home_xy[env_ids].clone(),
            "stand_yaw": self.stand_yaw[env_ids].clone(),
            "active_idx": self.active_idx[env_ids].clone(),
            "side_idx": self.side_idx[env_ids].clone(),
            "seat1_latch": self.seat1_latch[env_ids].clone(),
            "seat2_latch": self.seat2_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for b, st in zip(self.pans, state["pans"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.stones, state["stones"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.weights, state["weights"]):
            b.write_root_state_to_sim(st, env_ids)
        self.stand_home_xy[env_ids] = state["stand_home_xy"]
        self.stand_yaw[env_ids] = state["stand_yaw"]
        self.active_idx[env_ids] = state["active_idx"]
        self.side_idx[env_ids] = state["side_idx"]
        self.seat1_latch[env_ids] = state["seat1_latch"]
        self.seat2_latch[env_ids] = state["seat2_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        u = c.unit_mass * 1000
        return (
            f"A two-pan BEAM BALANCE stands on a light-gray bench: a dark fork stand "
            f"carries a brown beam on a nearly frictionless pivot (hard stops at "
            f"+/-{c.stop_deg:.0f} degrees), with a square TRAY hanging level from each "
            f"beam end. In one tray sits a SLATE-BLUE STONE (a {c.stone_size * 100:.0f} cm "
            f"cube). Its mass is a whole multiple — 1 to {c.units_max} — of "
            f"{u:.0f} g, and CANNOT be judged from its size: identical-looking stones "
            f"of different masses exist (the spares lie in a storage row on the floor, "
            f"off the bench — leave them there, at least {c.exclusion_r * 100:.0f} cm "
            f"from the stand). On the bench front stand three BRASS WEIGHTS of "
            f"{u:.0f} g, {2 * u:.0f} g and {4 * u:.0f} g — 1, 2 and 4 units — whose "
            f"footprints scale with mass (biggest = heaviest). The stone has parked the "
            f"beam on a stop; the beam tips toward whichever side carries more moment, "
            f"so each trial load you seat tells you whether you are under or over.\n"
            f"Goal: WEIGH THE STONE — seat brass weights in the hanging trays (either "
            f"tray; you may load both and use differences) until the beam floats LEVEL "
            f"(within {c.level_tol_deg:.0f} degrees) and everything hangs dead still, "
            f"hands-off. The stone must stay in its own tray; every brass weight must "
            f"end either seated in a tray or at least {c.exclusion_r * 100:.0f} cm from "
            f"the stand (nothing balanced on the beam or stand); do not shift or topple "
            f"the stand. The scene is judged only at SUSTAINED rest — a beam swinging "
            f"through level does not count; a single wrong unit leaves the beam pinned "
            f"on a stop."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Weigh the slate stone: seat brass weights in the balance's hanging trays "
            "until the beam floats level and still, hands-off. Keep the stone in its "
            "tray, leave no weight on the beam or stand or loose beside the scale, "
            "and do not move the stand."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def tilt(self) -> torch.Tensor:
        """(N,) signed beam elevation (radians): positive = the +x (PanP) end HIGH."""
        from isaaclab.utils.math import quat_apply

        q = self.beam.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        u = quat_apply(q, ex)
        return torch.atan2(u[:, 2], u[:, :2].norm(dim=-1).clamp(min=1e-9))

    def _local_in_pan(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 2, 3): a world point expressed in each pan's body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        out = []
        for pan in self.pans:
            d = pos_w - pan.data.root_pos_w
            out.append(quat_apply_inverse(pan.data.root_quat_w, d))
        return torch.stack(out, dim=1)

    def _in_tray(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 2) bool: point seated in pan k's tray (xy inside the walls, z in the
        resting band above the tray floor — loads may rest on each other)."""
        c = self.cfg
        loc = self._local_in_pan(pos_w)
        xy_ok = loc[:, :, :2].abs().amax(dim=-1) < c.tray_in
        z = loc[:, :, 2]
        z_ok = (z > -c.stem_len - 0.012) & (z < -c.stem_len + 0.13)
        return xy_ok & z_ok

    # ----- predicates -------------------------------------------------------------------------
    def level(self) -> torch.Tensor:
        """(N,) bool: beam within `level_tol_deg` of horizontal."""
        return self.tilt().abs() <= math.radians(self.cfg.level_tol_deg)

    def _active_stone_pos(self) -> torch.Tensor:
        pos = torch.stack([b.data.root_pos_w for b in self.stones], dim=1)   # (N, 7, 3)
        idx = self.active_idx.view(-1, 1, 1).expand(-1, 1, 3)
        return pos.gather(1, idx).squeeze(1)

    def stone_home(self) -> torch.Tensor:
        """(N,) bool: the active stone seated in the tray it spawned in (clause C —
        removing the stone would let the empty balance level itself)."""
        in_tray = self._in_tray(self._active_stone_pos())                    # (N, 2)
        return in_tray.gather(1, self.side_idx.view(-1, 1)).squeeze(1)

    def weight_in_tray(self) -> torch.Tensor:
        """(N, W) bool: weight seated in EITHER tray."""
        return torch.stack(
            [self._in_tray(b.data.root_pos_w).any(dim=1) for b in self.weights], dim=1)

    def weights_clean(self) -> torch.Tensor:
        """(N,) bool: every brass weight in a tray or beyond `exclusion_r` of the
        stand; every spare stone beyond `exclusion_r` (clause D)."""
        c = self.cfg
        sxy = self.stand.data.root_pos_w[:, :2]
        ok = torch.ones(sxy.shape[0], dtype=torch.bool, device=sxy.device)
        in_tray = self.weight_in_tray()
        for j, b in enumerate(self.weights):
            far = (b.data.root_pos_w[:, :2] - sxy).norm(dim=-1) > c.exclusion_r
            ok = ok & (in_tray[:, j] | far)
        act = self.active_idx
        for i, b in enumerate(self.stones):
            far = (b.data.root_pos_w[:, :2] - sxy).norm(dim=-1) > c.exclusion_r
            ok = ok & (far | (act == i))
        return ok

    def stand_home(self) -> torch.Tensor:
        """(N,) bool: stand within `home_xy_tol` of its spawn xy and upright within
        `home_tilt_deg` (clause E)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        xy = self.stand.data.root_pos_w[:, :2] - self.env_origins[:, :2]
        near = (xy - self.stand_home_xy).norm(dim=-1) < c.home_xy_tol
        q = self.stand.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        up = quat_apply(q, ez)[:, 2] > math.cos(math.radians(c.home_tilt_deg))
        return near & up

    def _load_vels(self) -> torch.Tensor:
        """(N, K) linear speeds of pans, all stones and all weights."""
        bodies = self.pans + self.stones + self.weights
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies], dim=1)

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        ok = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_still_avel
        ok = ok & (self._load_vels() < c.obj_still_speed).all(dim=1)
        ok = ok & (self.stand.data.root_lin_vel_w.norm(dim=-1) < c.obj_still_speed)
        return ok

    def still(self) -> torch.Tensor:
        """(N,) bool: SUSTAINED rest — sub-threshold for `still_steps` consecutive
        sim steps (counter kept by `post_step`)."""
        return self.still_count >= self.cfg.still_steps

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Per-step: advance the sustained-stillness counter, then latch the
        seating credit: a brass weight counts as SEATED when it rests settled in a
        tray while the stone is home and the stand is home. Latched credit does not
        evaporate when a weight is later lifted out for a revise step."""
        c = self.cfg
        now = self._still_now()
        self.still_count = torch.where(
            now, self.still_count + 1, torch.zeros_like(self.still_count))

        vels = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.weights], dim=1)
        seated = self.weight_in_tray() & (vels < c.obj_still_speed)
        gate = self.stone_home() & self.stand_home()
        n_seated = torch.where(gate, seated.sum(dim=1), torch.zeros_like(seated.sum(dim=1)))
        self.seat1_latch = torch.maximum(self.seat1_latch, (n_seated >= 1).float())
        self.seat2_latch = torch.maximum(self.seat2_latch, (n_seated >= 2).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beam level AND everything sustained-still AND the stone home
        AND every weight in a tray or far away (spares far) AND the stand at its
        spawn pose. Physics then guarantees the moments cancel unit-exactly: any
        wrong brass combination rests on a stop far outside the level tolerance."""
        return (self.level() & self.still() & self.stone_home()
                & self.weights_clean() & self.stand_home())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 once a weight was ever seated + 0.10 once two
        were ever seated at once, capped at 0.25; exactly 1.0 iff success() now.
        Null policy: the stone parks the beam on a stop, nothing is seated -> 0."""
        c = self.cfg
        base = (c.w_seat1 * self.seat1_latch + c.w_seat2 * self.seat2_latch).clamp(0.0, 0.25)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="beam_balance", robot="null"))
