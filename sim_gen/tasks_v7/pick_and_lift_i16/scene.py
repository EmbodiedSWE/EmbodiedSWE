"""BallastLeverScene — raise the caged red cube by COUNTERWEIGHT: load steel blocks
into the basket on the far end of a seesaw beam until torque tips it (sim_gen task
`pick_and_lift_i16`).

Derived from rlbench/pick_and_lift, but STRATEGICALLY different: the seed is the purest
prehensile plan there is — grasp the red 5 cm cube (among color distractors) and carry
it up to a marked height; the object is chosen to fit the jaw and the lift IS the
grasp-and-raise. Here the red cube is judged on exactly the seed's outcome — ending up
HIGH — but it can never be grasped or lifted: it rides in a walled, ROOFED cage on the
cargo end of a two-sided lever (a seesaw beam resting by its axle in an open cradle),
and every cage opening is smaller than the cube. The only way to raise it is to operate
the MACHINE: fetch heavy BLUE steel blocks scattered on the floor and load at least
three of them into the open BASKET on the opposite end of the beam, until the ballast
torque overcomes the cargo torque and the beam tips onto its raised stop, hoisting the
cage ~12 cm. A WHITE block of the same size is foam (25 g vs 260 g) — visually a decoy
pair to the seed's color-distractor cubes, but here the discrimination is about MASS:
foam in the basket contributes nothing. A solver therefore needs a different PLAN
(indirect lift via counterweight loading; never touching the judged object) and a
different code structure (torque-driven mechanism state, containment-in-a-moving-frame
predicates, count-latched credit) — not a grasp-and-raise checker.

success() iff, settled: the beam rests RAISED (cargo-end pitch sin >= `raised_sin`,
near its +13 deg stop), with >= `need_k` (3) STEEL blocks contained in the basket
(beam-frame containment — the basket is a moving target), the red cargo still contained
in its cage tray, and the beam still seated on its cradle. score() latches every
physics substep: 0.10 * best steel-block approach toward the basket (normalized by its
own spawn distance) + 0.15 per steel block ever settled in the basket (up to 3) + 0.15
* best raise fraction GATED on >= 3 steel in the basket at that instant (so pressing
the beam down by hand farms nothing), capped at 0.85; exactly 1.0 iff success(). Doing
nothing scores ~0.

TORQUE WINDOW (asserted in `__post_init__`, worst-case block/cargo positions inside
their trays): 3 steel blocks ALWAYS tip the beam, 2 steel + the foam NEVER do. Arms are
kept tight by making both trays narrow along the beam axis, so where a block rests
inside the basket barely changes its lever arm.

Assets are fully procedural (no external files):
  - cradle (KINEMATIC): ground slab, two notch posts (wide 40 mm throat — the axle can
    roll through its full swing without pinching against a prong), two tilt-stop posts
    under the beam ends (+/- ~13.4 deg travel);
  - beam (DYNAMIC compound, explicit COM pinned at the axle so the empty beam is
    neutral): 640 x 90 mm plate, 24 mm axle + end caps (axial retention), cargo tray
    with a barred CAGE roof (openings <= 30 mm < the 60 mm cube), ballast basket
    (open top, 60 mm throat along the beam so a 40 mm block drops in with 10 mm slack);
  - cargo: RED 60 mm cube, 660 g, spawned inside the cage;
  - ballast: four BLUE 40 mm steel cubes, 260 g each, scattered on the floor;
  - decoy: one WHITE 40 mm foam cube, 25 g, scattered among them.

Per-episode randomization (verified by readback in smoke): machine xy + yaw (the
basket end swings — read the scene), the five floor blocks spawn on a jittered arc with
their SLOT ASSIGNMENT PERMUTED (which slot holds the foam varies), free block yaw,
cargo xy jitter + small yaw inside the cage. Heavy imports (isaaclab, pxr) are deferred
so importing this module — and registering the scene — stays app-free.
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


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl_y(stage, path: str, radius: float, height: float, center, color,
           contact_offset: float) -> None:
    """Cylinder with its axis along local Y (the beam's pivot axis)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Y")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -height / 2, -radius),
                          Gf.Vec3f(radius, height / 2, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float, com=None) -> None:
    """Dynamic rigid-body armor on a compound root: mass (+ optional EXPLICIT centre of
    mass — the beam pins its COM at the axle so the empty lever is neutral), damping so
    the swing settles promptly, no sleeping while we judge velocities, depenetration cap."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.20)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC cradle at `prim_path`. Origin = the point on the GROUND
    directly under the axle; the notch posts straddle the beam at local +/- y, the two
    tilt-stop posts sit under the beam ends at local +/- x."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/slab", (cfg.slab_x, cfg.slab_y, cfg.slab_t),
         (0.0, 0.0, cfg.slab_t / 2), cfg.slab_color, co)
    notch_floor = cfg.axle_h - cfg.axle_r  # the axle rests here
    for tag, sgn in (("post_yp", 1.0), ("post_yn", -1.0)):
        y = sgn * cfg.post_y
        _box(stage, f"{prim_path}/{tag}_col", (cfg.post_wx, cfg.post_wy, notch_floor),
             (0.0, y, notch_floor / 2), cfg.post_color, co)
        for ptag, psgn in (("xp", 1.0), ("xn", -1.0)):
            px_c = psgn * (cfg.throat / 2 + cfg.prong_t / 2)
            _box(stage, f"{prim_path}/{tag}_prong_{ptag}",
                 (cfg.prong_t, cfg.post_wy, cfg.prong_h),
                 (px_c, y, notch_floor + cfg.prong_h / 2), cfg.post_color, co)
    for tag, sgn in (("stop_xp", 1.0), ("stop_xn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.stop_w, cfg.stop_w, cfg.stop_h),
             (sgn * cfg.arm, 0.0, cfg.stop_h / 2), cfg.stop_color, co)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC beam at `prim_path`. Origin = the AXLE CENTRE; cargo tray +
    cage at local +x, ballast basket at local -x; explicit COM at the origin."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass, com=(0.0, 0.0, 0.0))

    co = cfg.contact_offset
    pz = cfg.plate_z  # plate centre z (top = pz + t/2)
    _box(stage, f"{prim_path}/plate", (cfg.plate_l, cfg.plate_w, cfg.plate_t),
         (0.0, 0.0, pz), cfg.plate_color, co)
    _cyl_y(stage, f"{prim_path}/axle", cfg.axle_r, cfg.axle_len, (0.0, 0.0, 0.0),
           cfg.axle_color, co)
    for tag, sgn in (("cap_yp", 1.0), ("cap_yn", -1.0)):
        _cyl_y(stage, f"{prim_path}/{tag}", cfg.cap_r, cfg.cap_t,
               (0.0, sgn * (cfg.axle_len / 2 + cfg.cap_t / 2), 0.0), cfg.axle_color, co)

    wt, wh = cfg.wall_t, cfg.wall_h
    wz = pz + cfg.plate_t / 2 + wh / 2  # wall centre z (walls stand on the plate top)

    # ---- cargo tray + cage at +arm ----
    cx = cfg.arm
    tix, tiy = cfg.tray_in_x, cfg.tray_in_y
    _box(stage, f"{prim_path}/tray_floor", (tix + 2 * wt + 0.010, tiy + 2 * wt + 0.010,
         cfg.plate_t), (cx, 0.0, pz), cfg.plate_color, co)
    for tag, sgn in (("xp", 1.0), ("xn", -1.0)):
        _box(stage, f"{prim_path}/tray_w{tag}", (wt, tiy + 2 * wt, wh),
             (cx + sgn * (tix / 2 + wt / 2), 0.0, wz), cfg.tray_color, co)
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/tray_w{tag}", (tix, wt, wh),
             (cx, sgn * (tiy / 2 + wt / 2), wz), cfg.tray_color, co)
    pil_z0 = wz + wh / 2
    for xtag, xs in (("xp", 1.0), ("xn", -1.0)):
        for ytag, ys in (("yp", 1.0), ("yn", -1.0)):
            _box(stage, f"{prim_path}/cage_pil_{xtag}{ytag}",
                 (0.008, 0.008, cfg.cage_gap),
                 (cx + xs * (tix / 2 + wt / 2), ys * (tiy / 2 + wt / 2),
                  pil_z0 + cfg.cage_gap / 2), cfg.cage_color, co)
    roof_z = pil_z0 + cfg.cage_gap + cfg.roof_t / 2
    for tag, sgn in (("xp", 1.0), ("xn", -1.0)):
        _box(stage, f"{prim_path}/cage_roof_{tag}", (cfg.roof_bar_w, tiy + 2 * wt, cfg.roof_t),
             (cx + sgn * cfg.roof_bar_x, 0.0, roof_z), cfg.cage_color, co)

    # ---- ballast basket at -arm ----
    bx = -cfg.arm
    bix, biy = cfg.basket_in_x, cfg.basket_in_y
    _box(stage, f"{prim_path}/basket_floor", (bix + 2 * wt + 0.010, biy + 2 * wt + 0.010,
         cfg.plate_t), (bx, 0.0, pz), cfg.plate_color, co)
    for tag, sgn in (("xp", 1.0), ("xn", -1.0)):
        _box(stage, f"{prim_path}/basket_w{tag}", (wt, biy + 2 * wt, wh),
             (bx + sgn * (bix / 2 + wt / 2), 0.0, wz), cfg.basket_color, co)
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/basket_w{tag}", (bix, wt, wh),
             (bx, sgn * (biy / 2 + wt / 2), wz), cfg.basket_color, co)
    return root


def _cradle_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cradle" not in _SPAWNER_CACHE:

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            slab_x: float = 0.50
            slab_y: float = 0.26
            slab_t: float = 0.030
            axle_h: float = 0.160
            axle_r: float = 0.012
            post_y: float = 0.075
            post_wx: float = 0.070
            post_wy: float = 0.030
            throat: float = 0.040
            prong_t: float = 0.012
            prong_h: float = 0.030
            stop_w: float = 0.050
            stop_h: float = 0.075
            arm: float = 0.26
            slab_color: tuple = (0.35, 0.35, 0.38)
            post_color: tuple = (0.25, 0.25, 0.28)
            stop_color: tuple = (0.45, 0.42, 0.30)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["cradle"] = CradleSpawnerCfg

    return _SPAWNER_CACHE["cradle"](
        mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        slab_x=c.slab_x, slab_y=c.slab_y, slab_t=c.slab_t, axle_h=c.axle_h,
        axle_r=c.axle_r, post_y=c.post_y, post_wx=c.post_wx, post_wy=c.post_wy,
        throat=c.throat, prong_t=c.prong_t, prong_h=c.prong_h, stop_w=c.stop_w,
        stop_h=c.stop_h, arm=c.arm, slab_color=c.cradle_slab_color,
        post_color=c.cradle_post_color, stop_color=c.cradle_stop_color,
        contact_offset=c.contact_offset,
    )


def _beam_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "beam" not in _SPAWNER_CACHE:

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            mass: float = 1.2
            plate_l: float = 0.64
            plate_w: float = 0.09
            plate_t: float = 0.014
            plate_z: float = -0.019
            axle_r: float = 0.012
            axle_len: float = 0.20
            cap_r: float = 0.022
            cap_t: float = 0.008
            arm: float = 0.26
            wall_t: float = 0.008
            wall_h: float = 0.045
            tray_in_x: float = 0.070
            tray_in_y: float = 0.090
            cage_gap: float = 0.030
            roof_t: float = 0.008
            roof_bar_w: float = 0.015
            roof_bar_x: float = 0.0225
            basket_in_x: float = 0.060
            basket_in_y: float = 0.130
            plate_color: tuple = (0.55, 0.55, 0.58)
            axle_color: tuple = (0.20, 0.20, 0.22)
            tray_color: tuple = (0.60, 0.30, 0.10)
            cage_color: tuple = (0.15, 0.15, 0.17)
            basket_color: tuple = (0.85, 0.65, 0.10)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg

    return _SPAWNER_CACHE["beam"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass=c.beam_mass, plate_l=c.plate_l, plate_w=c.plate_w, plate_t=c.plate_t,
        plate_z=c.plate_z, axle_r=c.axle_r, axle_len=c.axle_len, cap_r=c.cap_r,
        cap_t=c.cap_t, arm=c.arm, wall_t=c.wall_t, wall_h=c.wall_h,
        tray_in_x=c.tray_in_x, tray_in_y=c.tray_in_y, cage_gap=c.cage_gap,
        roof_t=c.roof_t, roof_bar_w=c.roof_bar_w, roof_bar_x=c.roof_bar_x,
        basket_in_x=c.basket_in_x, basket_in_y=c.basket_in_y,
        plate_color=c.beam_plate_color, axle_color=c.beam_axle_color,
        tray_color=c.tray_color, cage_color=c.cage_color, basket_color=c.basket_color,
        contact_offset=c.contact_offset,
    )


def _cube_cfg(size: float, mass: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=(size, size, size),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=contact_offset, rest_offset=0.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.10),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastLeverSceneCfg(BaseCfg):
    """Config for `BallastLeverScene`. Honesty knobs asserted in `__post_init__`: the
    torque window (3 steel blocks ALWAYS tip the beam, 2 steel + foam NEVER do, at
    worst-case positions inside the trays), the cage openings all smaller than the
    cargo cube (the seed's grasp-and-lift is physically unavailable), the basket throat
    wide enough to drop a block in, and the axle free to roll its full swing without
    pinching in the notch throat."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    raised_deg: float = tunable(10.0)  # cargo-end pitch for "raised" (stop is ~13.4 deg)
    need_k: int = tunable(3)  # steel blocks required in the basket
    settle_lin: float = tunable(0.08)  # max |lin vel| of cargo + counted blocks (m/s)
    settle_beam_ang: float = tunable(0.25)  # max beam |ang vel| when judging (rad/s)
    block_slow: float = tunable(0.10)  # a block must be slower than this to latch "in basket"

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    machine_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the whole machine (m)
    machine_yaw_deg: float = tunable(45.0)  # uniform +/- machine yaw (basket end swings)
    slot_jitter: float = tunable(0.045)  # uniform +/- xy jitter per block slot (m)
    block_yaw_deg: float = tunable(180.0)  # uniform +/- block yaw (free)
    cargo_jx: float = tunable(0.004)  # cargo xy jitter inside the cage (m)
    cargo_jy: float = tunable(0.010)
    cargo_yaw_deg: float = tunable(4.0)  # cargo yaw jitter (tray confines larger yaws)

    # --- tunable: placement ------------------------------------------------------------------
    machine_pos: tuple = tunable((0.10, 0.0))  # cradle origin (under the axle), nominal
    # five spawn slots for the 4 steel + 1 foam blocks (assignment permuted per episode)
    slots: tuple = tunable(((-0.52, -0.30), (-0.60, -0.15), (-0.63, 0.0),
                            (-0.60, 0.15), (-0.52, 0.30)))

    # --- info: machine structure (beam local frame: origin = axle centre) --------------------
    arm: float = info(0.26)  # tray/basket centres at local x = +/- arm
    beam_mass: float = info(1.2)  # explicit COM at the axle -> empty beam is neutral
    plate_l: float = info(0.64)
    plate_w: float = info(0.09)
    plate_t: float = info(0.014)
    plate_z: float = info(-0.019)  # plate centre below the axle (top -0.012, bottom -0.026)
    axle_r: float = info(0.012)
    axle_len: float = info(0.20)
    cap_r: float = info(0.022)  # axle end caps: axial retention against the posts
    cap_t: float = info(0.008)
    wall_t: float = info(0.008)
    wall_h: float = info(0.045)
    tray_in_x: float = info(0.070)  # cargo tray interior (cube 60 -> +/-5 mm x slack)
    tray_in_y: float = info(0.090)
    cage_gap: float = info(0.030)  # side opening between wall top and roof (< cube)
    roof_t: float = info(0.008)
    roof_bar_w: float = info(0.015)  # two roof bars -> central gap 30 mm (< cube)
    roof_bar_x: float = info(0.0225)
    basket_in_x: float = info(0.060)  # narrow throat along the beam: block arm stays tight
    basket_in_y: float = info(0.130)  # three blocks fit side by side across the beam
    # cradle
    slab_x: float = info(0.50)
    slab_y: float = info(0.26)
    slab_t: float = info(0.030)
    axle_h: float = info(0.160)  # axle centre height when seated in the notch
    post_y: float = info(0.075)
    post_wx: float = info(0.070)
    post_wy: float = info(0.030)
    throat: float = info(0.040)  # notch throat: 24 mm axle + 16 mm roll room (no pinch)
    prong_t: float = info(0.012)
    prong_h: float = info(0.030)
    stop_w: float = info(0.050)
    stop_h: float = info(0.075)  # tilt stops under the beam ends -> +/- ~13.4 deg travel
    # payloads
    cargo_size: float = info(0.060)
    cargo_mass: float = info(0.66)
    block_size: float = info(0.040)
    steel_mass: float = info(0.26)
    foam_mass: float = info(0.025)
    n_steel: int = info(4)
    # colors
    cradle_slab_color: tuple = info((0.35, 0.35, 0.38))
    cradle_post_color: tuple = info((0.25, 0.25, 0.28))
    cradle_stop_color: tuple = info((0.45, 0.42, 0.30))
    beam_plate_color: tuple = info((0.55, 0.55, 0.58))
    beam_axle_color: tuple = info((0.20, 0.20, 0.22))
    tray_color: tuple = info((0.60, 0.30, 0.10))
    cage_color: tuple = info((0.15, 0.15, 0.17))
    basket_color: tuple = info((0.85, 0.65, 0.10))
    cargo_color: tuple = info((0.90, 0.08, 0.08))
    steel_color: tuple = info((0.15, 0.35, 0.85))
    foam_color: tuple = info((0.93, 0.93, 0.93))
    # Explicit small offsets: a default ~2 cm offset would eat the basket drop clearance.
    contact_offset: float = info(0.001)
    # spawn keep-outs
    keepout_machine: float = info(0.50)  # blocks spawn at least this from the cradle origin
    keepout_block: float = info(0.075)  # pairwise block separation

    # Derived (filled in __post_init__).
    sin_stop: float = field(default=None, init=False)  # |sin(pitch)| at the tilt stop
    raised_sin: float = field(default=None, init=False)
    tray_floor_z: float = field(default=None, init=False)  # beam-local tray/basket floor top
    reset_pitch: float = field(default=None, init=False)  # beam written just short of the stop

    def __post_init__(self) -> None:
        plate_bot = self.plate_z - self.plate_t / 2
        self.tray_floor_z = self.plate_z + self.plate_t / 2
        # underside of the beam meets the stop top: axle_h + plate_bot*cos - arm*sin = stop_h
        s = (self.axle_h + plate_bot - self.stop_h) / self.arm  # cos~1 first pass
        for _ in range(4):  # tiny fixed-point refinement with the cos term
            s = (self.axle_h + plate_bot * math.sqrt(max(1 - s * s, 0.0)) - self.stop_h) / self.arm
        self.sin_stop = s
        self.raised_sin = math.sin(math.radians(self.raised_deg))
        self.reset_pitch = math.asin(s) - math.radians(1.0)
        assert self.raised_sin < self.sin_stop - 0.03, "raised threshold must sit below the stop"

        # ---- torque window (worst-case block/cargo positions inside the trays) ----
        bh = self.block_size / 2
        b_slack = self.basket_in_x / 2 - bh  # how far a block centre can shift in the basket
        c_slack = self.tray_in_x / 2 - self.cargo_size / 2
        tilt_z = (self.tray_floor_z + bh + self.block_size) * self.sin_stop  # stacked-block tilt arm
        arm_b_min = self.arm - b_slack - 0.002
        arm_b_max = self.arm + b_slack + tilt_z
        arm_c_min = self.arm - c_slack - 0.002
        arm_c_max = self.arm + c_slack + 0.002
        t3_min = 3 * self.steel_mass * arm_b_min
        t_cargo_max = self.cargo_mass * arm_c_max
        assert t3_min > 1.06 * t_cargo_max, (
            f"3 steel blocks must ALWAYS tip the beam ({t3_min:.4f} vs {t_cargo_max:.4f})")
        t2_max = 2 * self.steel_mass * arm_b_max + self.foam_mass * arm_b_max
        t_cargo_min = self.cargo_mass * arm_c_min
        assert t2_max < 0.94 * t_cargo_min, (
            f"2 steel + foam must NEVER tip the beam ({t2_max:.4f} vs {t_cargo_min:.4f})")

        # ---- cage honesty: every opening smaller than the cargo cube ----
        roof_gap_mid = 2 * (self.roof_bar_x - self.roof_bar_w / 2)
        roof_gap_out = self.tray_in_x / 2 + self.wall_t - (self.roof_bar_x + self.roof_bar_w / 2)
        assert roof_gap_mid < self.cargo_size - 0.020, "cage roof central gap must trap the cube"
        assert roof_gap_out < self.cargo_size - 0.020, "cage roof outer gaps must trap the cube"
        assert self.cage_gap < self.cargo_size - 0.020, "cage side openings must trap the cube"
        head = self.cage_gap + self.wall_h - self.cargo_size  # cube headroom under the roof
        assert 0.008 < head < self.cargo_size, "cube must fit inside the cage with headroom"

        # ---- drop + pivot clearances ----
        assert self.basket_in_x >= self.block_size + 0.016, "basket throat: block drop slack"
        assert self.basket_in_y >= 3 * self.block_size + 0.008, "three blocks fit across"
        assert self.throat >= 2 * self.axle_r + 0.012, "axle must roll without pinching"
        roll = self.axle_r * 2 * math.asin(self.sin_stop)  # rolling excursion over full swing
        assert self.throat / 2 - self.axle_r > roll / 2, "full swing fits inside the throat"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_lever_lift")
class BallastLeverScene(BaseScene):
    cfg: BallastLeverSceneCfg

    def __init__(self, cfg: BallastLeverSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastLeverSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=_cradle_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.machine_pos[0], c.machine_pos[1], 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=_beam_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.machine_pos[0], c.machine_pos[1], c.axle_h)),
            ),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=_cube_cfg(c.cargo_size, c.cargo_mass, c.cargo_color, c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.machine_pos[0] + c.arm, c.machine_pos[1],
                         c.axle_h + c.tray_floor_z + c.cargo_size / 2 + 0.002)),
            ),
        }
        for i in range(c.n_steel):
            sx, sy = c.slots[i]
            out[f"steel_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Steel_" + str(i),
                spawn=_cube_cfg(c.block_size, c.steel_mass, c.steel_color, c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, c.block_size / 2 + 0.002)),
            )
        fx, fy = c.slots[c.n_steel]
        out["foam"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Foam",
            spawn=_cube_cfg(c.block_size, c.foam_mass, c.foam_color, c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, c.block_size / 2 + 0.002)),
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
        self.cradle: RigidObject = env.iscene["cradle"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.steel: list[RigidObject] = [env.iscene[f"steel_{i}"] for i in range(c.n_steel)]
        self.foam: RigidObject = env.iscene["foam"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n, c.n_steel), 0.60, device=dev)  # block-spawn -> basket dist
        self.approach_latch = torch.zeros(n, device=dev)
        self.k_latch = torch.zeros(n, device=dev)  # best settled steel-in-basket count
        self.raise_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cradle with xy jitter + yaw, beam seated on the axle tipped
        just short of its cargo-down stop, cargo inside the cage (jitter + small yaw),
        the five floor blocks dealt onto PERMUTED, jittered slots with free yaw and
        keep-out resampling; latches zeroed and per-block approach baselines captured."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- machine pose ---
        mach_xy = torch.tensor(c.machine_pos, device=dev).expand(m, 2).clone()
        mach_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.machine_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.machine_yaw_deg)
        half = yaw / 2
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = mach_xy
        st[:, 3:7] = q_yaw
        st[:, 0:3] += origin
        self.cradle.write_root_state_to_sim(st, env_ids)

        # --- beam: seated on the axle, tipped just short of the cargo-down stop ---
        th = c.reset_pitch  # Ry(+th): +x (cargo end) pitches DOWN
        ct, stt = math.cos(th / 2), math.sin(th / 2)
        q_pitch = torch.tensor([ct, 0.0, stt, 0.0], device=dev).expand(m, 4)
        q_beam = quat_mul(q_yaw, q_pitch)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = mach_xy
        st[:, 2] = c.axle_h + 0.001
        st[:, 3:7] = q_beam
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- cargo: inside the cage, beam-local, jitter + small yaw ---
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.arm + (torch.rand(m, device=dev) * 2 - 1) * c.cargo_jx
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.cargo_jy
        loc[:, 2] = c.tray_floor_z + c.cargo_size / 2 + 0.0015
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cargo_yaw_deg) / 2
        q_c = torch.zeros(m, 4, device=dev)
        q_c[:, 0] = torch.cos(cyaw)
        q_c[:, 3] = torch.sin(cyaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = mach_xy
        st[:, 2] = c.axle_h + 0.001
        st[:, 0:3] += quat_apply(q_beam, loc)
        st[:, 3:7] = quat_mul(q_beam, q_c)
        st[:, 0:3] += origin
        self.cargo.write_root_state_to_sim(st, env_ids)

        # --- blocks: permuted slots, jitter, free yaw, keep-out resampled ---
        slots = torch.tensor(c.slots, device=dev)  # (5, 2)
        perm = torch.rand(m, len(c.slots), device=dev).argsort(dim=1)  # body -> slot
        placed: list[torch.Tensor] = []
        bodies = [*self.steel, self.foam]
        for j, body in enumerate(bodies):
            base = slots[perm[:, j]]
            xy = base + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            for _ in range(12):
                bad = (xy - mach_xy).norm(dim=-1) < c.keepout_machine
                for q in placed:
                    bad |= (xy - q).norm(dim=-1) < c.keepout_block
                if not bad.any():
                    break
                k = int(bad.sum())
                xy[bad] = base[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * c.slot_jitter
            placed.append(xy)
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.block_size / 2 + 0.002
            st[:, 3] = torch.cos(byaw)
            st[:, 6] = torch.sin(byaw)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- baselines + latches ---
        basket_w = torch.zeros(m, 3, device=dev)
        basket_w[:, 0] = -c.arm
        basket_xy = mach_xy + quat_apply(q_beam, basket_w)[:, :2]
        for j in range(c.n_steel):
            self.d0[env_ids, j] = (placed[j] - basket_xy).norm(dim=-1).clamp(min=0.10)
        self.approach_latch[env_ids] = 0.0
        self.k_latch[env_ids] = 0.0
        self.raise_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cradle": self.cradle.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "steel": [b.data.root_state_w[env_ids].clone() for b in self.steel],
            "foam": self.foam.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "k_latch": self.k_latch[env_ids].clone(),
            "raise_latch": self.raise_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cradle.write_root_state_to_sim(state["cradle"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        for b, s in zip(self.steel, state["steel"]):
            b.write_root_state_to_sim(s, env_ids)
        self.foam.write_root_state_to_sim(state["foam"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.k_latch[env_ids] = state["k_latch"]
        self.raise_latch[env_ids] = state["raise_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A seesaw machine stands on the floor: a gray BEAM "
            f"({c.plate_l * 1000:.0f} mm long) rests by its dark axle in an open cradle, "
            f"free to tip like a balance between two travel stops. On one end of the beam "
            f"sits an orange-brown tray covered by a barred black CAGE, and locked inside "
            f"the cage is a RED cube ({c.cargo_size * 1000:.0f} mm) — every opening in the "
            f"cage is smaller than the cube, so the cube can never be taken out or touched "
            f"usefully. On the other end of the beam is an open-topped YELLOW BASKET. At "
            f"the start the cage end rests DOWN on its stop. Scattered on the floor away "
            f"from the machine lie five loose {c.block_size * 1000:.0f} mm cubes: four "
            f"BLUE steel blocks (~{c.steel_mass * 1000:.0f} g each — heavy) and one WHITE "
            f"foam block (~{c.foam_mass * 1000:.0f} g — almost weightless; it looks the "
            f"same size but it is a decoy). The red cube weighs about as much as two and a "
            f"half steel blocks.\n"
            f"Goal: RAISE the red cube. Since it cannot be grasped, load BLUE steel blocks "
            f"into the yellow basket one by one — drop each block in over the open top — "
            f"until their weight overbalances the red cube and the beam tips the other "
            f"way, lifting the cage to its raised stop (about "
            f"{2 * c.arm * c.sin_stop * 100:.0f} cm of lift). At least {c.need_k} steel "
            f"blocks IN the basket are required; the white foam block is far too light to "
            f"help and counts for nothing. Load the blocks in any order. Judged only when "
            f"settled: beam resting tipped with the cage end up (>= {c.raised_deg:.0f} "
            f"deg), >= {c.need_k} blue blocks inside the basket, and the red cube still in "
            f"its cage. Blocks resting on the beam outside the basket, on the ground, or "
            f"held against the machine count for nothing; pressing the beam down by hand "
            f"achieves nothing lasting — only ballast IN the basket keeps the cube up."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Raise the caged red cube by tipping the seesaw: drop at least three of the "
            "heavy BLUE steel blocks into the yellow basket on the far end of the beam so "
            "their weight lifts the cage to its raised stop. The white foam block is too "
            "light to help, and the red cube itself cannot be grasped."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _beam_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> beam body frame (origin = axle centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w, p_w - self.beam.data.root_pos_w)

    def raised_sin_now(self) -> torch.Tensor:
        """(N,) sin of the cargo-end pitch: +1-ish when the CAGE end is up. Beam local
        +x is the cargo end; raised means its world z-component is NEGATIVE pitch of
        Ry, i.e. (R @ ex).z > 0."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.beam.data.root_quat_w, ex)[:, 2]

    def beam_seated(self) -> torch.Tensor:
        """(N,) bool: the beam's axle still rides in the cradle notch."""
        c = self.cfg
        d_xy = (self.beam.data.root_pos_w - self.cradle.data.root_pos_w)[:, :2].norm(dim=-1)
        z = (self.beam.data.root_pos_w - self.env_origins)[:, 2]
        return (d_xy < 0.06) & ((z - c.axle_h).abs() < 0.025)

    def steel_pos_w(self) -> torch.Tensor:
        """(N, S, 3) world positions of the steel blocks."""
        return torch.stack([b.data.root_pos_w for b in self.steel], dim=1)

    def _in_basket(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,) bool for a batch of world points: inside the basket interior, beam frame."""
        c = self.cfg
        loc = self._beam_local(p_w)
        return ((loc[:, 0] + c.arm).abs() < c.basket_in_x / 2 + 0.005) \
            & (loc[:, 1].abs() < c.basket_in_y / 2 + 0.005) \
            & (loc[:, 2] > c.tray_floor_z - 0.010) & (loc[:, 2] < c.tray_floor_z + 0.12)

    def steel_in_basket(self) -> torch.Tensor:
        """(N, S) bool: steel block contained in the basket (a MOVING target — judged
        in the beam's body frame)."""
        return torch.stack([self._in_basket(b.data.root_pos_w) for b in self.steel], dim=1)

    def foam_in_basket(self) -> torch.Tensor:
        """(N,) bool (informational — foam earns nothing either way)."""
        return self._in_basket(self.foam.data.root_pos_w)

    def cargo_in_cage(self) -> torch.Tensor:
        """(N,) bool: the red cube still inside its cage tray, beam frame."""
        c = self.cfg
        loc = self._beam_local(self.cargo.data.root_pos_w)
        return ((loc[:, 0] - c.arm).abs() < c.tray_in_x / 2 + 0.008) \
            & (loc[:, 1].abs() < c.tray_in_y / 2 + 0.008) \
            & (loc[:, 2] > c.tray_floor_z - 0.010) & (loc[:, 2] < c.tray_floor_z + 0.09)

    def settled(self) -> torch.Tensor:
        """(N,) bool: beam angularly quiet, cargo + every in-basket steel block slow."""
        c = self.cfg
        beam_ok = (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_beam_ang) \
            & (self.beam.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        cargo_ok = self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.steel], dim=1)
        blocks_ok = ((vel < c.settle_lin) | ~self.steel_in_basket()).all(dim=1)
        return beam_ok & cargo_ok & blocks_ok

    # ----- graded progress --------------------------------------------------------------------
    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: best steel-block progress toward the basket, each normalized
        by its own spawn distance."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        basket_l = torch.tensor([-c.arm, 0.0, 0.0], device=self.env.device)
        basket_w = self.beam.data.root_pos_w + quat_apply(
            self.beam.data.root_quat_w, basket_l.expand(self.env.num_envs, 3))
        d = (self.steel_pos_w()[:, :, :2] - basket_w[:, None, :2]).norm(dim=-1)
        return (1.0 - d / self.d0).clamp(0.0, 1.0).max(dim=1).values

    def raise_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: beam raise progress from the cargo-down stop to the raised
        threshold, GATED on >= need_k steel blocks in the basket at this instant (a
        hand pressing the beam down farms nothing)."""
        c = self.cfg
        k = self.steel_in_basket().sum(dim=1)
        frac = (self.raised_sin_now() + c.sin_stop) / (c.raised_sin + c.sin_stop)
        return frac.clamp(0.0, 1.0) * (k >= c.need_k).float()

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best approach, best settled in-basket steel count, and gated raise
        progress each physics substep, so transient progress keeps its credit."""
        self.approach_latch = torch.maximum(self.approach_latch, self.approach_frac())
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.steel], dim=1)
        k_now = (self.steel_in_basket() & (vel < self.cfg.block_slow)).sum(dim=1).float()
        self.k_latch = torch.maximum(self.k_latch, k_now)
        self.raise_latch = torch.maximum(self.raise_latch, self.raise_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beam resting RAISED with >= need_k steel blocks in the basket,
        cargo still caged, beam still on its cradle, settled."""
        c = self.cfg
        k = self.steel_in_basket().sum(dim=1)
        return (self.raised_sin_now() >= c.raised_sin) & (k >= c.need_k) \
            & self.cargo_in_cage() & self.beam_seated() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched best approach + 0.15 per steel block ever
        settled in the basket (up to 3) + 0.15 * latched gated raise progress, capped at
        0.85; exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy
        (grasp the red cube and lift it) is physically unavailable and earns nothing."""
        base = (0.10 * self.approach_latch
                + 0.15 * self.k_latch.clamp(max=float(self.cfg.need_k))
                + 0.15 * self.raise_latch).clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="ballast_lever_lift", robot="null"))
