"""CliffSweepScene — herd three loose balls off a raised stage into the catch basin
below the cliff edge, using the open-mouthed scoop dome as a plow; NEVER lift a ball
(sim_gen task `approach_grasp_bowl_i133`).

Derived from pick_place/approach_grasp_bowl, but STRATEGICALLY different: the seed is a
single-object prehensile plan — approach ONE bowl lying in tabletop clutter, close the
parallel jaw on its rim, lift it and carry it along marked waypoints; every load-bearing
step is one grasp affordance plus free-space transport of a rigidly held object. Here the
judged objects are THREE balls that must all end up inside a basin that sits BELOW the
stage they start on, and the task forbids the seed's entire verb: a ball that is ever
lifted above `lift_z` spoils the episode IRREVERSIBLY (latched; success is impossible and
the score is capped). The balls are trivially graspable (40 mm vs the 80 mm Franka jaw) —
that is the point: the graspable-looking strategy is the trap. The only sanctioned
transport is NON-PREHENSILE HERDING ALONG THE FLOOR: cage the rolling balls under the
scoop dome (a walled disk with a 90-degree open mouth), plow the pack across the stage
toward the red-striped cliff edge, and BRAKE before the edge so the balls coast out
through the mouth, fly off the cliff, and drop into the basin — a curling-style
plow-and-release delivery of a multi-body pack over a drop, with the tool itself required
to stay parked on the stage. Nothing is carried; gravity does the last leg.

success() (all live, judged on physical poses):
  - ALL THREE balls are inside the basin (mirrored-x window, |y| under the side walls,
    centre LOW — resting on the basin floor, not stacked or perched) and slow,
  - the scoop rests back on the stage top (xy inside the stage, origin z in the resting
    band — neither fallen into the basin nor riding on balls) and is settled,
  - the episode is NOT spoiled (no ball centre ever exceeded `lift_z`).
score() = latched progress anchored in the demonstrated solution: 0.10 * CAPTURED (some
ball was inside the scoop cage while the scoop sat low on the stage) + 0.25 per ball
DELIVERED (latched: in the basin and slow), capped at 0.85; a spoiled episode is capped
at 0.20 no matter what else happened; exactly 1.0 iff success(). Doing nothing ~0.

Assets are fully procedural (no external files):
  - stage: KINEMATIC raised gray block (0.76 x 0.52 m, top at z=0.12) with fixed guard
    rails along both long (y) edges; the cliff (basin) end is open, the other end is
    closed by a separate kinematic end rail teleported to the non-basin side each reset.
  - basin: KINEMATIC dark-green catch tray at ground level (floor + far wall + two side
    walls; the stage's own vertical face closes the near side), teleported each episode
    to a RANDOM side (+x or -x, yawed 180 deg for -x); a RED visual stripe rides with it
    on the stage top, marking which edge is the live cliff.
  - scoop: DYNAMIC blue dome (12-sector wall ring, 3 sectors skipped = 90-degree mouth
    on body +x, roof cap, yellow handle stub; 0.30 kg) — the herding tool.
  - balls: three DYNAMIC orange 40 mm spheres (30 g) — the judged objects.

Per-episode randomization (verified by readback in smoke): basin side (+x/-x flip of
cliff, end rail, and the whole delivery direction), ball-cluster centre xy with per-ball
disc offsets (pairwise-separated), scoop spawn xy near the closed end plus free yaw.
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

# Scoop wall ring: 12 sectors of 30 deg; these are skipped -> 90 deg mouth on body +x.
MOUTH_SECTORS = (11, 0, 1)

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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         yaw: float = 0.0) -> None:
    """A colored box prim (optional yaw about local z); collides iff `contact_offset`
    is not None."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        sxf.AddRotateZOp().Set(math.degrees(yaw))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _cylinder(stage, path: str, radius: float, height: float, center, color,
              contact_offset: float | None) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _phys_material(stage, path: str, static: float, dynamic: float) -> Any:
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


def _rigid_dynamic(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Dynamic rigid-body armor on a compound root: explicit MassAPI mass (custom
    spawners apply NO cfg schemas — author everything here), damping, no sleeping while
    we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)


def _spawn_stage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC stage at `prim_path`. Origin = ground-level centre of the
    block. Main block (top at stage_h) + two fixed guard rails hugging the long (y)
    edges, inner faces flush with the block sides so nothing rolls off sideways."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    hx, hy, h = cfg.half_x, cfg.half_y, cfg.stage_h
    _box(stage, f"{prim_path}/block", (2 * hx, 2 * hy, h), (0.0, 0.0, h / 2),
         cfg.stage_color, cfg.contact_offset)
    rt, rh = cfg.rail_t, cfg.rail_h
    for name, sy in (("rail_ny", -1.0), ("rail_py", 1.0)):
        _box(stage, f"{prim_path}/{name}", (2 * hx, rt, rh),
             (0.0, sy * (hy + rt / 2), h + rh / 2), cfg.rail_color, cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.top_friction, cfg.top_friction - 0.05)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/block"), mat)
    return root


def _spawn_basin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC catch basin at `prim_path`. Origin = ground-level centre of
    the floor. Floor + far wall (local +x) + two side walls; the local -x side is OPEN —
    the stage's own vertical face closes it. A RED visual stripe (no collision) sits at
    stage-top height over the local -x edge: it rides with the basin when it teleports,
    so it always marks the live cliff edge."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)

    fx, fy, ft = cfg.floor_x, cfg.floor_y, cfg.floor_t
    wh, wt = cfg.wall_h, cfg.wall_t
    _box(stage, f"{prim_path}/floor", (fx, fy, ft), (0.0, 0.0, ft / 2),
         cfg.basin_color, cfg.contact_offset)
    _box(stage, f"{prim_path}/far_wall", (wt, fy, wh), (cfg.far_wall_x, 0.0, wh / 2),
         cfg.basin_color, cfg.contact_offset)
    for name, sy in (("side_ny", -1.0), ("side_py", 1.0)):
        _box(stage, f"{prim_path}/{name}", (fx, wt, wh),
             (0.0, sy * cfg.side_wall_y, wh / 2), cfg.basin_color, cfg.contact_offset)
    _box(stage, f"{prim_path}/cliff_stripe", (0.015, cfg.stripe_len, 0.002),
         (cfg.stripe_x, 0.0, cfg.stripe_z), (0.90, 0.10, 0.08), None)  # visual only
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.5, 0.45)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/floor"), mat)
    return root


def _spawn_end_rail(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC end rail at `prim_path`. Origin = ground level under the
    rail centre; the bar sits on the stage top. Teleported to the NON-basin end each
    reset, closing the stage so balls can only leave over the cliff."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_kinematic(root)
    _box(stage, f"{prim_path}/bar", (cfg.rail_t, cfg.rail_len, cfg.rail_h),
         (0.0, 0.0, cfg.stage_h + cfg.rail_h / 2), cfg.rail_color, cfg.contact_offset)
    return root


def _spawn_scoop(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC scoop dome at `prim_path`. Origin = wall-ring centre at
    mid-wall height. 12-sector wall ring (sectors in MOUTH_SECTORS skipped -> 90 deg
    mouth on body +x), roof cap disk, yellow handle stub on top. MassAPI mass on the
    root (CoM stays at the body origin — symmetric enough for plowing)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.10, ang_damp=0.30)

    n_seg = 12
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n_seg) + 0.002
    for k in range(n_seg):
        if k in MOUTH_SECTORS:
            continue
        ang = 2 * math.pi * k / n_seg
        _box(stage, f"{prim_path}/wall_{k:02d}", (cfg.wall_t, seg_len, cfg.wall_h),
             (r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0),
             cfg.body_color, cfg.contact_offset, yaw=ang)
    _cylinder(stage, f"{prim_path}/cap", cfg.cap_r, cfg.cap_t,
              (0.0, 0.0, cfg.wall_h / 2 + cfg.cap_t / 2), cfg.body_color, cfg.contact_offset)
    hz = cfg.wall_h / 2 + cfg.cap_t + cfg.handle_h / 2
    _box(stage, f"{prim_path}/handle", (cfg.handle_a, cfg.handle_a, cfg.handle_h),
         (0.0, 0.0, hz), (0.95, 0.85, 0.10), cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.30, 0.25)
    for k in range(n_seg):
        if k in MOUTH_SECTORS:
            continue
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/wall_{k:02d}"), mat)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/cap"), mat)
    return root


def _stage_spawner_cfg(*, half_x: float, half_y: float, stage_h: float, rail_t: float,
                       rail_h: float, top_friction: float, stage_color: tuple,
                       rail_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stage" not in _SPAWNER_CACHE:

        @configclass
        class StageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stage)
            half_x: float = 0.38
            half_y: float = 0.26
            stage_h: float = 0.12
            rail_t: float = 0.012
            rail_h: float = 0.05
            top_friction: float = 0.45
            stage_color: tuple = (0.58, 0.58, 0.60)
            rail_color: tuple = (0.40, 0.40, 0.44)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["stage"] = StageSpawnerCfg

    return _SPAWNER_CACHE["stage"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        half_x=half_x, half_y=half_y, stage_h=stage_h, rail_t=rail_t, rail_h=rail_h,
        top_friction=top_friction, stage_color=stage_color, rail_color=rail_color,
        contact_offset=contact_offset,
    )


def _basin_spawner_cfg(*, floor_x: float, floor_y: float, floor_t: float, wall_h: float,
                       wall_t: float, far_wall_x: float, side_wall_y: float,
                       stripe_x: float, stripe_z: float, stripe_len: float,
                       basin_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "basin" not in _SPAWNER_CACHE:

        @configclass
        class BasinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basin)
            floor_x: float = 0.40
            floor_y: float = 0.60
            floor_t: float = 0.008
            wall_h: float = 0.10
            wall_t: float = 0.010
            far_wall_x: float = 0.195
            side_wall_y: float = 0.295
            stripe_x: float = -0.2075
            stripe_z: float = 0.121
            stripe_len: float = 0.52
            basin_color: tuple = (0.10, 0.35, 0.15)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["basin"] = BasinSpawnerCfg

    return _SPAWNER_CACHE["basin"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        floor_x=floor_x, floor_y=floor_y, floor_t=floor_t, wall_h=wall_h, wall_t=wall_t,
        far_wall_x=far_wall_x, side_wall_y=side_wall_y, stripe_x=stripe_x,
        stripe_z=stripe_z, stripe_len=stripe_len, basin_color=basin_color,
        contact_offset=contact_offset,
    )


def _end_rail_spawner_cfg(*, rail_t: float, rail_len: float, rail_h: float,
                          stage_h: float, rail_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "end_rail" not in _SPAWNER_CACHE:

        @configclass
        class EndRailSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_end_rail)
            rail_t: float = 0.012
            rail_len: float = 0.52
            rail_h: float = 0.05
            stage_h: float = 0.12
            rail_color: tuple = (0.40, 0.40, 0.44)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["end_rail"] = EndRailSpawnerCfg

    return _SPAWNER_CACHE["end_rail"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        rail_t=rail_t, rail_len=rail_len, rail_h=rail_h, stage_h=stage_h,
        rail_color=rail_color, contact_offset=contact_offset,
    )


def _scoop_spawner_cfg(*, inner_r: float, wall_t: float, wall_h: float, cap_r: float,
                       cap_t: float, handle_a: float, handle_h: float, mass: float,
                       body_color: tuple, contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "scoop" not in _SPAWNER_CACHE:

        @configclass
        class ScoopSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_scoop)
            inner_r: float = 0.078
            wall_t: float = 0.009
            wall_h: float = 0.055
            cap_r: float = 0.088
            cap_t: float = 0.008
            handle_a: float = 0.022
            handle_h: float = 0.062
            mass: float = 0.30
            body_color: tuple = (0.15, 0.35, 0.85)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["scoop"] = ScoopSpawnerCfg

    return _SPAWNER_CACHE["scoop"](
        inner_r=inner_r, wall_t=wall_t, wall_h=wall_h, cap_r=cap_r, cap_t=cap_t,
        handle_a=handle_a, handle_h=handle_h, mass=mass, body_color=body_color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CliffSweepSceneCfg(BaseCfg):
    """Config for `CliffSweepScene`. Honesty knobs asserted in `__post_init__`: the
    spoil gate sits above anything the sanctioned floor strategy can reach but below any
    genuine carry; the delivery window sits strictly inside the basin walls; the scoop
    spawn zone clears the end rail and its reach covers the whole ball cluster; the
    basin mouth is wider than the stage so every cliff drop is caught."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    basin_x_lo: float = tunable(0.395)  # ball mirrored-x window inside the basin (m)
    basin_x_hi: float = tunable(0.755)
    basin_half_w: float = tunable(0.272)  # ball |y| under the side walls (m)
    basin_ball_z: float = tunable(0.055)  # ball centre below this = ON the basin floor
    # (one ball on the floor is 0.028; a two-ball stack is 0.068 -> rejected)
    ball_settle: float = tunable(0.05)  # max ball |lin vel| at success (m/s)
    deliver_vel: float = tunable(0.30)  # ball |lin vel| gate for the delivered latch
    park_x: float = tunable(0.37)  # scoop park: |x| inside the stage (m)
    park_y: float = tunable(0.25)
    park_z_lo: float = tunable(0.14)  # scoop origin resting band (rest is 0.1475;
    park_z_hi: float = tunable(0.22)  # in-basin ~0.036, riding a ball >0.22 fails settle)
    settle_lin: float = tunable(0.05)  # scoop settled thresholds
    settle_ang: float = tunable(0.50)
    lift_z: float = tunable(0.19)  # any ball centre EVER above this -> spoiled (latch)
    capture_slack: float = tunable(0.003)  # capture radius = inner_r - ball_r - slack
    scoop_low_z: float = tunable(0.16)  # scoop origin below this = "down on the stage"

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cluster_x: float = tunable(0.05)  # ball-cluster centre mirrored-x in +/- this (m)
    cluster_y: float = tunable(0.09)  # ... and y in +/- this (m)
    cluster_r: float = tunable(0.045)  # per-ball disc radius around the centre (m)
    ball_gap: float = tunable(0.045)  # min pairwise ball-centre spawn distance (m)
    scoop_mx_lo: float = tunable(-0.27)  # scoop spawn mirrored-x range (closed end)
    scoop_mx_hi: float = tunable(-0.24)
    scoop_y: float = tunable(0.12)  # scoop spawn y in +/- this (m)

    # --- info: structure ----------------------------------------------------------------------
    half_x: float = info(0.38)  # stage half-length (x) — cliff faces +/-x
    half_y: float = info(0.26)  # stage half-width (y)
    stage_h: float = info(0.12)  # stage top height
    rail_t: float = info(0.012)
    rail_h: float = info(0.05)
    end_rail_x: float = info(0.374)  # end-rail centre |x| (inner face at 0.368)
    basin_off: float = info(0.58)  # basin origin |x| on the cliff side
    floor_x: float = info(0.40)  # basin floor spans world |x| in [0.38, 0.78]
    floor_y: float = info(0.60)
    floor_t: float = info(0.008)
    wall_h: float = info(0.10)
    wall_t: float = info(0.010)
    far_wall_x: float = info(0.195)  # basin-local far-wall centre (inner face 0.19)
    side_wall_y: float = info(0.295)  # basin-local side-wall centres (inner faces 0.29)
    inner_r: float = info(0.078)  # scoop cage inner radius
    scoop_wall_t: float = info(0.009)
    scoop_wall_h: float = info(0.055)
    cap_r: float = info(0.088)  # scoop outer (roof) radius
    cap_t: float = info(0.008)
    handle_a: float = info(0.022)
    handle_h: float = info(0.062)
    scoop_mass: float = info(0.30)
    ball_r: float = info(0.020)  # graspable on purpose — the spoil latch is the fence
    ball_mass: float = info(0.030)
    jaw_max: float = info(0.080)  # Franka parallel-jaw max opening (embodiment honesty)
    top_friction: float = info(0.45)
    stage_color: tuple = info((0.58, 0.58, 0.60))
    rail_color: tuple = info((0.40, 0.40, 0.44))
    basin_color: tuple = info((0.10, 0.35, 0.15))
    scoop_color: tuple = info((0.15, 0.35, 0.85))
    ball_color: tuple = info((0.95, 0.50, 0.10))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    scoop_rest_z: float = field(default=None, init=False)
    capture_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.scoop_rest_z = self.stage_h + self.scoop_wall_h / 2  # 0.1475
        self.capture_r = self.inner_r - self.ball_r - self.capture_slack  # ~0.055
        assert 2 * self.ball_r < self.jaw_max - 0.020, (
            "the balls must be trivially graspable — the spoil latch, not size, forbids the "
            "seed strategy")
        # Spoil gate: above anything floor herding can reach (a ball climbing over another
        # peaks at stage_h + 3r), below any genuine lift-and-carry over the rails.
        assert self.lift_z > self.stage_h + 3 * self.ball_r + 0.005, "herding must not spoil"
        assert self.lift_z < self.stage_h + self.rail_h + 2 * self.ball_r, (
            "carrying a ball over the rails must spoil")
        # Delivery window strictly inside the basin interior.
        assert self.basin_x_lo > self.basin_off - self.floor_x / 2 + 0.010, (
            "in-basin window must start past the cliff-side floor edge")
        assert self.basin_x_hi < self.basin_off + self.far_wall_x - self.wall_t / 2, (
            "in-basin window must end inside the far wall")
        assert self.basin_half_w < self.side_wall_y - self.wall_t / 2, (
            "in-basin |y| must sit inside the side walls")
        assert self.floor_t + self.ball_r < self.basin_ball_z < self.floor_t + 3 * self.ball_r, (
            "one floor-resting ball passes; a two-ball stack must fail the low gate")
        # Basin catches the whole stage width; drops from the cliff cannot miss.
        assert self.side_wall_y - self.wall_t / 2 > self.half_y + 0.02, (
            "basin interior must be wider than the stage")
        # Scoop spawn zone: clear of the end rail, reach covering the whole cluster.
        assert abs(self.scoop_mx_lo) + self.cap_r < self.end_rail_x - self.rail_t / 2 - 0.005, (
            "scoop spawn must clear the end rail (depenetration honesty)")
        assert abs(self.scoop_mx_hi) - self.cap_r > self.cluster_x + self.cluster_r + self.ball_r, (
            "scoop spawn must not overlap any possible ball spawn")
        assert self.scoop_y + self.cap_r < self.half_y - self.rail_t, (
            "scoop spawn must clear the side rails")
        # Cage feasibility: the whole cluster fits under the cage with clearance, and the
        # capture radius really means "inside the walls".
        assert self.cluster_r + self.ball_r < self.inner_r - 0.005, (
            "a covered cluster must fit inside the cage walls")
        assert self.ball_gap > 2 * self.ball_r + 0.004, "balls must never spawn intersecting"
        # The equilateral-triangle spawn fallback (circumradius 0.75*cluster_r) must
        # itself satisfy the gap: side = sqrt(3) * r_c > ball_gap.
        assert 0.75 * self.cluster_r * math.sqrt(3.0) > self.ball_gap + 0.002, (
            "fallback triangle side must exceed the pairwise spawn gap")
        assert self.capture_r > self.ball_r, "capture radius must be meaningful"
        # Scoop park band brackets the resting height only.
        assert self.park_z_lo < self.scoop_rest_z < self.park_z_hi


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cliff_sweep")
class CliffSweepScene(BaseScene):
    cfg: CliffSweepSceneCfg

    def __init__(self, cfg: CliffSweepSceneCfg | None = None) -> None:
        super().__init__(cfg or CliffSweepSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ball_spawn = sim_utils.SphereCfg(
            radius=c.ball_r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.06, angular_damping=0.22,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,  # kills the GPU sphere-creep artifact
                sleep_threshold=0.0, stabilization_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.5, restitution=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
        )
        ball_z = c.stage_h + c.ball_r + 0.002
        assets: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "stage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stage",
                spawn=_stage_spawner_cfg(
                    half_x=c.half_x, half_y=c.half_y, stage_h=c.stage_h, rail_t=c.rail_t,
                    rail_h=c.rail_h, top_friction=c.top_friction,
                    stage_color=c.stage_color, rail_color=c.rail_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "basin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basin",
                spawn=_basin_spawner_cfg(
                    floor_x=c.floor_x, floor_y=c.floor_y, floor_t=c.floor_t,
                    wall_h=c.wall_h, wall_t=c.wall_t, far_wall_x=c.far_wall_x,
                    side_wall_y=c.side_wall_y, stripe_x=-(c.basin_off - c.half_x + 0.0075),
                    stripe_z=c.stage_h + 0.001, stripe_len=2 * c.half_y,
                    basin_color=c.basin_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.basin_off, 0.0, 0.0)),
            ),
            "end_rail": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/EndRail",
                spawn=_end_rail_spawner_cfg(
                    rail_t=c.rail_t, rail_len=2 * c.half_y, rail_h=c.rail_h,
                    stage_h=c.stage_h, rail_color=c.rail_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-c.end_rail_x, 0.0, 0.0)),
            ),
            "scoop": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Scoop",
                spawn=_scoop_spawner_cfg(
                    inner_r=c.inner_r, wall_t=c.scoop_wall_t, wall_h=c.scoop_wall_h,
                    cap_r=c.cap_r, cap_t=c.cap_t, handle_a=c.handle_a,
                    handle_h=c.handle_h, mass=c.scoop_mass, body_color=c.scoop_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.255, 0.0, c.scoop_rest_z + 0.003)),
            ),
        }
        for i, (bx, by) in enumerate(((0.0, 0.0), (0.05, 0.03), (-0.05, -0.03))):
            assets[f"ball_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball" + str(i),
                spawn=ball_spawn.replace() if hasattr(ball_spawn, "replace") else ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, ball_z)),
            )
        return assets

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
        self.stage: RigidObject = env.iscene["stage"]
        self.basin: RigidObject = env.iscene["basin"]
        self.end_rail: RigidObject = env.iscene["end_rail"]
        self.scoop: RigidObject = env.iscene["scoop"]
        self.balls: list[RigidObject] = [env.iscene[f"ball_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.side = torch.ones(n, device=dev)  # +1: cliff on +x; -1: cliff on -x
        self.captured = torch.zeros(n, device=dev)
        self.delivered = torch.zeros(n, 3, device=dev)
        self.spoiled = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: flip a coin for the cliff side; teleport the basin (yawed 180
        deg on the -x side so its far wall stays outboard) and the end rail (opposite
        end); scatter the three balls around a random cluster centre with pairwise
        separation; drop the scoop near the closed end with free yaw; zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self.side[env_ids] = side

        # --- basin: cliff side, yawed to keep the open mouth toward the stage ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = side * c.basin_off
        st[:, 3] = (side > 0).float()  # identity quat on +x ...
        st[:, 6] = (side < 0).float()  # ... yaw pi on -x
        st[:, 0:3] += origin
        self.basin.write_root_state_to_sim(st, env_ids)

        # --- end rail: the NON-cliff end ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = -side * c.end_rail_x
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.end_rail.write_root_state_to_sim(st, env_ids)

        # --- balls: cluster centre + per-ball disc offsets, pairwise separated ---
        cmx = (torch.rand(m, device=dev) * 2 - 1) * c.cluster_x
        cy = (torch.rand(m, device=dev) * 2 - 1) * c.cluster_y
        ctr = torch.stack([side * cmx, cy], dim=-1)  # (m,2) world-frame cluster centre

        def _sample_offsets() -> torch.Tensor:
            ang = torch.rand(m, 3, device=dev) * 2 * math.pi
            rad = c.cluster_r * torch.sqrt(torch.rand(m, 3, device=dev))
            return torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=-1)

        def _min_gap(o: torch.Tensor) -> torch.Tensor:
            d = (o.unsqueeze(2) - o.unsqueeze(1)).norm(dim=-1)  # (m,3,3)
            return (d + torch.eye(3, device=dev) * 1.0).amin(dim=(1, 2))

        off = _sample_offsets()  # (m,3,2)
        for _ in range(64):
            bad = _min_gap(off) < c.ball_gap
            if not bad.any():
                break
            off = torch.where(bad.view(m, 1, 1), _sample_offsets(), off)
        bad = _min_gap(off) < c.ball_gap
        if bad.any():
            # Deterministic fallback (still randomized in rotation): an equilateral
            # triangle whose side exceeds ball_gap BY CONSTRUCTION (asserted in cfg),
            # so the pairwise-gap guarantee is unconditional — rejection sampling alone
            # occasionally fails and would leave intersecting spheres for PhysX to
            # depenetrate into contact.
            r_c = 0.75 * c.cluster_r
            th0 = torch.rand(m, device=dev) * 2 * math.pi
            ang = th0.unsqueeze(1) + torch.tensor(
                [0.0, 2 * math.pi / 3, 4 * math.pi / 3], device=dev)
            fb = torch.stack([r_c * torch.cos(ang), r_c * torch.sin(ang)], dim=-1)
            off = torch.where(bad.view(m, 1, 1), fb, off)
        ball_z = c.stage_h + c.ball_r + 0.002
        for i, ball in enumerate(self.balls):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = ctr + off[:, i]
            st[:, 2] = ball_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            ball.write_root_state_to_sim(st, env_ids)

        # --- scoop: near the closed end, free yaw ---
        smx = c.scoop_mx_lo + torch.rand(m, device=dev) * (c.scoop_mx_hi - c.scoop_mx_lo)
        sy = (torch.rand(m, device=dev) * 2 - 1) * c.scoop_y
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = side * smx
        st[:, 1] = sy
        st[:, 2] = c.scoop_rest_z + 0.003
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.scoop.write_root_state_to_sim(st, env_ids)

        # --- latches ---
        self.captured[env_ids] = 0.0
        self.delivered[env_ids] = 0.0
        self.spoiled[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "basin": self.basin.data.root_state_w[env_ids].clone(),
            "end_rail": self.end_rail.data.root_state_w[env_ids].clone(),
            "scoop": self.scoop.data.root_state_w[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "captured": self.captured[env_ids].clone(),
            "delivered": self.delivered[env_ids].clone(),
            "spoiled": self.spoiled[env_ids].clone(),
        }
        for i, ball in enumerate(self.balls):
            out[f"ball_{i}"] = ball.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.basin.write_root_state_to_sim(state["basin"], env_ids)
        self.end_rail.write_root_state_to_sim(state["end_rail"], env_ids)
        self.scoop.write_root_state_to_sim(state["scoop"], env_ids)
        for i, ball in enumerate(self.balls):
            ball.write_root_state_to_sim(state[f"ball_{i}"], env_ids)
        self.side[env_ids] = state["side"]
        self.captured[env_ids] = state["captured"]
        self.delivered[env_ids] = state["delivered"]
        self.spoiled[env_ids] = state["spoiled"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised gray STAGE ({2 * c.half_x * 100:.0f} x {2 * c.half_y * 100:.0f} cm, "
            f"top {c.stage_h * 100:.0f} cm above the ground) carries three loose ORANGE BALLS "
            f"({2 * c.ball_r * 1000:.0f} mm) and a BLUE SCOOP DOME — a walled disk with a "
            f"90-degree open mouth in its wall and a yellow handle on top. Guard rails close "
            f"three edges of the stage; the fourth edge is a CLIFF marked by a RED STRIPE, "
            f"and a dark-green CATCH BASIN sits on the ground below it. Which end is the "
            f"cliff varies by episode.\n"
            f"Goal: get ALL THREE balls resting inside the basin, and leave the scoop "
            f"sitting on the stage. The balls must NEVER be lifted: raising any ball above "
            f"{c.lift_z * 100:.0f} cm (about one ball above another) SPOILS the task "
            f"irreversibly — picking a ball up and carrying it is exactly the forbidden "
            f"move. Herd them along the floor instead: set the scoop down over the balls so "
            f"they are caged inside its walls, slide it — mouth forward, toward the red "
            f"stripe — to plow the pack across the stage, and stop short of the edge so the "
            f"balls run on out of the mouth, over the cliff, and drop into the basin. "
            f"Repeat for stragglers, then park the scoop back on the stage and let "
            f"everything come to rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Cage the orange balls under the blue scoop dome and slide it along the stage "
            "to plow them over the red-striped cliff edge into the green basin below — "
            "never lift a ball. Deliver all three, then leave the scoop parked on the stage."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _ball_pos(self) -> torch.Tensor:
        """(N,3,3) ball centres in the env frame (balls indexed on dim 1)."""
        return torch.stack(
            [b.data.root_pos_w - self.env_origins for b in self.balls], dim=1)

    def _ball_vel(self) -> torch.Tensor:
        """(N,3) ball linear-speed norms."""
        return torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.balls], dim=1)

    def _scoop_pos(self) -> torch.Tensor:
        """(N,3) scoop origin in the env frame."""
        return self.scoop.data.root_pos_w - self.env_origins

    def _in_basin(self, bp: torch.Tensor) -> torch.Tensor:
        """(N,3) bool: ball centres inside the basin interior, resting LOW."""
        c = self.cfg
        mx = bp[:, :, 0] * self.side.unsqueeze(1)  # mirrored x: cliff side is always +
        return ((mx > c.basin_x_lo) & (mx < c.basin_x_hi)
                & (bp[:, :, 1].abs() < c.basin_half_w) & (bp[:, :, 2] < c.basin_ball_z))

    # ----- predicates -------------------------------------------------------------------------
    def scoop_parked(self) -> torch.Tensor:
        """(N,) bool: scoop resting on the stage top (xy inside, origin in the resting
        z band) and settled — not in the basin, not riding on balls, not moving."""
        c = self.cfg
        sp = self._scoop_pos()
        in_xy = (sp[:, 0].abs() < c.park_x) & (sp[:, 1].abs() < c.park_y)
        in_z = (sp[:, 2] > c.park_z_lo) & (sp[:, 2] < c.park_z_hi)
        still = ((self.scoop.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                 & (self.scoop.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))
        return in_xy & in_z & still

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the spoil gate (any ball ever high), capture (a ball inside the cage
        while the scoop is down), and per-ball delivery (in the basin and slow) each
        physics substep, so transient progress keeps its credit and transient cheating
        keeps its penalty."""
        c = self.cfg
        bp = self._ball_pos()
        self.spoiled = self.spoiled | (bp[:, :, 2] > c.lift_z).any(dim=1)
        sp = self._scoop_pos()
        d_xy = (bp[:, :, :2] - sp[:, None, :2]).norm(dim=-1)  # (N,3)
        caged = (d_xy < self.cfg.capture_r).any(dim=1) & (sp[:, 2] < c.scoop_low_z)
        self.captured = torch.maximum(self.captured, caged.float())
        slow = self._ball_vel() < c.deliver_vel
        self.delivered = torch.maximum(self.delivered, (self._in_basin(bp) & slow).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three balls resting in the basin and slow NOW, scoop parked
        and settled on the stage, and the episode never spoiled."""
        bp = self._ball_pos()
        all_in = (self._in_basin(bp) & (self._ball_vel() < self.cfg.ball_settle)).all(dim=1)
        return all_in & self.scoop_parked() & ~self.spoiled

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * captured + 0.25 per delivered ball, capped at
        0.85; spoiled episodes are capped at 0.20 regardless; exactly 1.0 iff
        success(). Doing nothing scores ~0; the seed's strategy (grasp a ball and carry
        it) trips the spoil latch on the way up."""
        base = (0.10 * self.captured + 0.25 * self.delivered.sum(dim=1)).clamp(0.0, 0.85)
        base = torch.where(self.spoiled, base.clamp(max=0.20), base)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="cliff_sweep", robot="null"))
