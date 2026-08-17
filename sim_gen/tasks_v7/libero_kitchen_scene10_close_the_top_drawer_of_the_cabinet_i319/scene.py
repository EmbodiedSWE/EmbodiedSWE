"""BallastPressCabinetScene — close a cabinet drawer by LOADING BALLAST onto a counterweighted
press lever (sim_gen task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i319`).

Derived from libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet, but
STRATEGICALLY different: the seed's whole skill is a guided hand-push on the drawer front
along its prismatic travel — the robot actuates the judged part directly and continuously,
and its hand supplies the closing energy. Here the robot redistributes MASS instead: a
bell-crank PRESS LEVER pivots on a real revolute joint above the cabinet mouth; its long
arm carries an open-top 3-cell BALLAST HOPPER, its short arm is a steel COUNTERWEIGHT, and
a PRESS BLADE hangs straight down from the hinge with a rounded nose in front of the
drawer's face. The counterweight is sized so the empty lever — and the lever with only ONE
ballast block aboard — rests firmly UP against its raised stop; loading a SECOND block
reverses the torque balance, the lever heels over under gravity, and the blade sweeps the
drawer to its rear hard stop. The robot's entire contribution is picking 40 mm steel
blocks off the deck and dropping them into open hopper cells; the closing force is the
WEIGHT of the ballast, delivered through the lever — never the hand. Success demands the
whole machine state, judged LIVE on the settled scene: drawer seated in its channel AND
the lever heeled past its press angle AND at least two ballast blocks riding INSIDE the
hopper cells. Pushing the drawer shut by hand still leaves the lever up and the hopper
empty (~0.55, never success); pressing the lever down by hand and letting go just swings
it back up (the counterweight restores it) — only resident ballast holds the press.

The machine (all station-local; z=0 on the ground, +x runs OUT of the cabinet toward the
hopper and the block docks; the deck = plinth top at z = plinth_h):
  - the drawer is a free rigid body captured in a channel (floor slab, side walls, rear
    hard stop, roof) — "closed" = front face at q_stop = +3 mm, inside the 12 mm success
    tolerance; the channel mouth is OPEN (a hand may push the drawer — it earns partial
    credit only);
  - the lever is one rigid compound (axle + arm + hopper + counterweight + blade) hung on
    a spawn-authored revolute joint (rotY free [-0.5 deg, +36 deg], all else locked) whose
    body0 is the heavy DYNAMIC base — so the whole station can be yawed and jittered per
    episode and the hinge follows;
  - the TORQUE BUDGET is asserted in `__post_init__` from the same parts table the
    spawner authors (per-child density -> real CoM): one block anywhere in a cell leaves
    >= 0.4 N m of restoring margin (the lever provably stays up), two blocks anywhere in
    cells give >= 0.6 N m of driving surplus at lift-off and >= 2.5x the drawer's sliding
    friction torque at the seated angle — the press cannot stall by construction;
  - all sliding surfaces are slick (mu ~0.08-0.10, min combine, restitution 0): blocks
    slide OUTWARD in their tilted cells (raising drive torque — the safe direction) and
    the blade nose skids on the drawer face instead of grabbing.

Per-episode randomization (readback-verified by smoke): station yaw FREE (+/-180 deg) +
xy jitter, the drawer's initial opening q0, and the three block dock poses (rejection-
sampled with a deterministic fallback lattice).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15 x 2  ballast aboard — per-block latch: block rode inside a hopper cell (streak-
            gated so a bounce-through cannot latch), max two blocks credited
  0.55      drawer progress — latched max closure fraction of the initial opening,
            counted ONLY while the drawer genuinely rides its channel
base capped at 0.85; exactly 1.0 iff success(): drawer seated AND lever heeled past
press_min_deg AND >= 2 blocks LIVE inside cells, settled and finite. Null policy ~0. The
seed's end state — drawer pushed shut by hand, lever up, hopper empty — earns the drawer
credit only (~0.55) and can never succeed.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable, density: float | None = None):
    """One axis-aligned box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl_y(stage, path: str, *, center, radius, length, color, collide: Callable,
               density: float | None = None):
    """One y-axis cylinder child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Y")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _lever_parts(c: Any) -> list:
    """Shared parts table for the LEVER compound (lever-local frame, origin ON the hinge
    axis, +x toward the hopper, blade hanging along -z). The spawner authors exactly
    these prims with per-child density and the cfg torque audit integrates exactly the
    same table — the mass model and the authored body cannot drift apart.
    Entries: (tag, kind, center, dims, density, color_key). kind box: dims=(sx,sy,sz);
    kind cyl (y-axis): dims=(radius, length)."""
    xi, xo = c.tray_x0, c.tray_x1                      # cell interior x span
    fx = (xi + xo) / 2                                  # tray centre x
    t, dt = c.tray_wall_t, c.tray_div_t
    yi = c.tray_hw                                      # interior half-width
    z0, z1 = -c.cube_s / 2, c.tray_wall_top             # floor top / wall top
    wall_h = z1 - z0
    wall_zc = (z0 + z1) / 2
    dvy = c.cell_hw + dt / 2                            # divider centre |y|
    blade_l = c.press_r - c.nose_r
    rho, rho_cw = c.struct_density, c.cw_density
    return [
        ("axle", "cyl", (0.0, 0.0, 0.0), (c.axle_r, c.axle_len), rho, "steel"),
        ("arm", "box", ((c.arm_x0 + c.arm_x1) / 2, 0.0, 0.0),
         (c.arm_x1 - c.arm_x0, 0.040, 0.016), rho, "lever"),
        ("tray_floor", "box", (fx, 0.0, z0 - c.tray_floor_t / 2),
         (xo - xi + 2 * t, 2 * (yi + t), c.tray_floor_t), rho, "lever"),
        ("tray_wxi", "box", (xi - t / 2, 0.0, wall_zc), (t, 2 * (yi + t), wall_h), rho, "lever"),
        ("tray_wxo", "box", (xo + t / 2, 0.0, wall_zc), (t, 2 * (yi + t), wall_h), rho, "lever"),
        ("tray_wyn", "box", (fx, -(yi + t / 2), wall_zc), (xo - xi, t, wall_h), rho, "lever"),
        ("tray_wyp", "box", (fx, yi + t / 2, wall_zc), (xo - xi, t, wall_h), rho, "lever"),
        ("tray_dn", "box", (fx, -dvy, wall_zc), (xo - xi, dt, wall_h), rho, "lever"),
        ("tray_dp", "box", (fx, dvy, wall_zc), (xo - xi, dt, wall_h), rho, "lever"),
        ("blade", "box", (0.0, 0.0, -blade_l / 2), (0.016, c.blade_w, blade_l), rho, "steel"),
        ("nose", "cyl", (0.0, 0.0, -c.press_r), (c.nose_r, c.blade_w), rho, "nose"),
        ("cw", "box", (-c.cw_x, 0.0, 0.0), c.cw_size, rho_cw, "steel"),
    ]


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station BASE at `prim_path`: heavy DYNAMIC compound (dynamic because a
    joint anchored to a kinematic body0 stays world-fixed when the body is teleported at
    reset). Local frame: origin on the GROUND under the cabinet face plane (x=0 at the
    face, channel centre y=0); the deck (plinth top) is at z=plinth_h."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.base_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    prb.CreateLinearDampingAttr(0.5)
    prb.CreateAngularDampingAttr(0.5)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    D = c.plinth_h
    Y = c.chan_hw + c.chan_wall_t                       # channel outer half-width

    def span(tag, x, y, z, color):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2),
                 size=(x[1] - x[0], y[1] - y[0], z[1] - z[0]), color=color, collide=collide)

    # plinth (raises everything to Franka-friendly heights; also the block deck)
    span("plinth", (-0.22, 0.48), (-0.31, 0.31), (0.0, D), c.plinth_color)
    # channel floor slab (the drawer's sliding surface; extends under the open travel)
    span("slab", (-c.chan_len - c.back_t, c.slab_x1), (-0.085, 0.085),
         (D, D + c.slab_t), c.body_color)
    # channel side walls, rear hard stop, roof (drawer captured, mouth OPEN at +x)
    zw = (D + c.slab_t, D + 0.135)
    span("wall_p", (-c.chan_len - c.back_t, 0.010), (c.chan_hw, Y), zw, c.body_color)
    span("wall_n", (-c.chan_len - c.back_t, 0.010), (-Y, -c.chan_hw), zw, c.body_color)
    span("rear", (-c.chan_len - c.back_t, -c.chan_len), (-c.chan_hw, c.chan_hw), zw, c.body_color)
    span("roof", (-c.chan_len - c.back_t, 0.010), (-Y, Y), (D + 0.135, D + 0.150), c.body_color)
    # cradle posts flanking the lever axle (visual pivot; the joint itself is authored
    # on the lever and anchored to this base body)
    for sgn, tag in ((1.0, "post_p"), (-1.0, "post_n")):
        span(tag, (c.hinge_x - 0.025, c.hinge_x + 0.025),
             (sgn * c.post_y - 0.012, sgn * c.post_y + 0.012),
             (D, D + c.hinge_z), c.post_color)

    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_lever(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the PRESS LEVER at `prim_path`: DYNAMIC compound (per-child density so the
    hinge inertia and CoM are real), hung on a spawn-authored revolute joint (generic D6:
    rotY free in [tilt_min_deg, tilt_max_deg], every other axis locked) anchored to the
    sibling `{base}/Base` body at the hinge point. Lever origin ON the hinge axis."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(0.0)
    # DASHPOT: a block dropped into a hopper cell delivers an impulsive torque spike far
    # above the static threshold; heavy angular damping (a velocity-decay RATE, 1/s) kills
    # the transient within ~3-4 deg of swing — well short of the ~8 deg the nose needs to
    # reach the drawer — while the sustained 2-block press torque still heels the lever
    # over in ~1-2 s (quasi-static, exactly the story the task tells).
    prb.CreateAngularDampingAttr(float(cfg.lever_ang_damping))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    colors = {"lever": c.lever_color, "steel": c.steel_color, "nose": c.nose_color}
    for tag, kind, center, dims, density, ckey in _lever_parts(c):
        if kind == "box":
            _add_box(stage, f"{prim_path}/{tag}", center=center, size=dims,
                     color=colors[ckey], collide=collide, density=density)
        else:
            _add_cyl_y(stage, f"{prim_path}/{tag}", center=center, radius=dims[0],
                       length=dims[1], color=colors[ckey], collide=collide, density=density)

    # ---- the HINGE: generic D6 joint to the sibling base. rotY = the press heel
    # (+ = hopper down / blade toward the drawer). Limits in UsdPhysics convention:
    # low > high locks an axis; rotational limits in DEGREES.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Base"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # lever<->base contact stays FILTERED (the joint stops bound the sweep; nothing on
    # the lever should ever touch the base within its travel — asserted in cfg)
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), 0.0, float(c.plinth_h + c.hinge_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim = j.GetPrim()
    for axis in ("transX", "transY", "transZ", "rotX", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)                        # low > high = locked
    lim = UsdPhysics.LimitAPI.Apply(prim, "rotY")
    lim.CreateLowAttr(float(c.tilt_min_deg))
    lim.CreateHighAttr(float(c.tilt_max_deg))

    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    from isaaclab.sim.utils import bind_physics_material

    bind_physics_material(prim_path, slick)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at `prim_path`: DYNAMIC compound open-top box. Local frame:
    origin at the xy CENTRE with z=0 at the BOTTOM face; +x is the front."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L2, B2 = c.drawer_l / 2, c.drawer_w / 2
    t, zt = c.drawer_wall_t, c.drawer_h
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.drawer_floor_t / 2),
             size=(2 * L2, 2 * B2, c.drawer_floor_t), color=c.drawer_color, collide=collide)
    zc, zh = (c.drawer_floor_t + zt) / 2, zt - c.drawer_floor_t
    _add_box(stage, f"{prim_path}/w_rear", center=(-L2 + t / 2, 0.0, zc),
             size=(t, 2 * B2, zh), color=c.drawer_color, collide=collide)
    _add_box(stage, f"{prim_path}/w_yn", center=(0.0, -B2 + t / 2, zc),
             size=(2 * L2, t, zh), color=c.drawer_color, collide=collide)
    _add_box(stage, f"{prim_path}/w_yp", center=(0.0, B2 - t / 2, zc),
             size=(2 * L2, t, zh), color=c.drawer_color, collide=collide)
    # the front face — full height, what the press blade bears on
    _add_box(stage, f"{prim_path}/w_front", center=(L2 - t / 2, 0.0, zc),
             size=(t, 2 * B2, zh), color=c.drawer_face_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.drawer_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(c.drawer_damping))
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class BaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            plinth_h: float = 0.300
            chan_hw: float = 0.069
            chan_wall_t: float = 0.012
            chan_len: float = 0.157
            back_t: float = 0.013
            slab_t: float = 0.020
            slab_x1: float = 0.155
            hinge_x: float = 0.146
            hinge_z: float = 0.290
            post_y: float = 0.130
            base_mass: float = 70.0
            body_color: tuple = (0.45, 0.38, 0.30)
            plinth_color: tuple = (0.22, 0.22, 0.24)
            post_color: tuple = (0.30, 0.34, 0.42)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class LeverSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lever)
            plinth_h: float = 0.300
            hinge_x: float = 0.146
            hinge_z: float = 0.290
            press_r: float = 0.250
            nose_r: float = 0.010
            blade_w: float = 0.080
            arm_x0: float = 0.020
            arm_x1: float = 0.260
            tray_x0: float = 0.272
            tray_x1: float = 0.328
            tray_hw: float = 0.090
            cell_hw: float = 0.028
            tray_wall_t: float = 0.008
            tray_div_t: float = 0.006
            tray_floor_t: float = 0.010
            tray_wall_top: float = 0.035
            cube_s: float = 0.040
            axle_r: float = 0.012
            axle_len: float = 0.220
            cw_x: float = 0.100
            cw_size: tuple = (0.100, 0.070, 0.067)
            struct_density: float = 900.0
            cw_density: float = 7800.0
            tilt_min_deg: float = -0.5
            tilt_max_deg: float = 36.0
            lever_ang_damping: float = 15.0
            lever_color: tuple = (0.13, 0.50, 0.22)
            steel_color: tuple = (0.55, 0.57, 0.60)
            nose_color: tuple = (0.85, 0.30, 0.10)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            drawer_l: float = 0.160
            drawer_w: float = 0.130
            drawer_wall_t: float = 0.010
            drawer_floor_t: float = 0.010
            drawer_h: float = 0.095
            drawer_mass: float = 0.90
            drawer_damping: float = 1.0
            drawer_color: tuple = (0.66, 0.53, 0.35)
            drawer_face_color: tuple = (0.72, 0.60, 0.40)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        _SPAWNER_CACHE["base"] = BaseSpawnerCfg
        _SPAWNER_CACHE["lever"] = LeverSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastPressCabinetSceneCfg(BaseCfg):
    """Config for `BallastPressCabinetScene`. The press contract is asserted in
    `__post_init__` from the SAME parts table the spawner authors: the torque budget
    (one block provably rests up, two blocks provably press the drawer home with >= 2.5x
    friction), every clearance the mechanism relies on, and the rear-stop-equals-seated
    geometry."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    q_closed_tol: float = tunable(0.012)    # drawer front face within this of the face plane
    press_min_deg: float = tunable(28.0)    # lever heel angle that counts as "pressing"
    settle_drawer: float = tunable(0.05)    # max drawer |lin vel| when judging (m/s)
    settle_block: float = tunable(0.15)     # max block |lin vel| when judging (m/s)
    settle_lever_w: float = tunable(0.40)   # max lever |ang vel| when judging (rad/s)
    lane_y_tol: float = tunable(0.020)      # drawer centred in its channel when judged

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)         # station yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)        # station xy jitter (+/- m)
    q0_range: tuple = tunable((0.090, 0.130))   # initial drawer opening (m)
    dock_x_range: tuple = tunable((0.21, 0.41))     # block dock strip: x band on the deck
    dock_y_range: tuple = tunable((0.15, 0.26))     # block dock strip: |y| band (both sides)
    dock_min_sep: float = tunable(0.060)    # min pairwise block separation at reset

    # --- info: base / station (local frame: face plane x=0, channel centre y=0, ground z=0) ------
    plinth_h: float = info(0.300)           # deck height (station stands on a plinth)
    chan_hw: float = info(0.069)            # channel interior half-width (4 mm/side clearance)
    chan_wall_t: float = info(0.012)
    chan_len: float = info(0.157)           # rear hard stop inner face at x=-chan_len
    back_t: float = info(0.013)
    slab_t: float = info(0.020)             # drawer sliding surface height above the deck
    slab_x1: float = info(0.155)            # slab front edge (supports the open drawer)
    hinge_x: float = info(0.146)            # hinge axis x (above/before the mouth)
    hinge_z: float = info(0.290)            # hinge axis height above the DECK
    post_y: float = info(0.130)             # cradle post centres (visual pivot)
    base_mass: float = info(70.0)
    # --- info: lever (lever-local, origin ON the hinge axis) -------------------------------------
    press_r: float = info(0.250)            # hinge axis -> nose centre (blade length)
    nose_r: float = info(0.010)             # rounded press nose radius
    blade_w: float = info(0.080)            # blade / nose width (y)
    arm_x0: float = info(0.020)
    arm_x1: float = info(0.260)
    tray_x0: float = info(0.272)            # hopper cell interior x span
    tray_x1: float = info(0.328)
    tray_hw: float = info(0.090)            # hopper interior half-width (3 cells along y)
    cell_hw: float = info(0.028)            # centre cell interior half-width
    tray_wall_t: float = info(0.008)
    tray_div_t: float = info(0.006)
    tray_floor_t: float = info(0.010)
    tray_wall_top: float = info(0.035)      # wall top above the hinge plane (floor top = -cube_s/2:
    #                                         a resting block's CoM sits AT hinge height, making the
    #                                         torque margins angle-invariant)
    axle_r: float = info(0.012)
    axle_len: float = info(0.220)
    cw_x: float = info(0.100)               # counterweight centre at lever-local -cw_x
    cw_size: tuple = info((0.100, 0.070, 0.067))
    struct_density: float = info(900.0)
    cw_density: float = info(7800.0)
    tilt_min_deg: float = info(-0.5)        # joint travel: raised stop ...
    tilt_max_deg: float = info(36.0)        # ... to just past the seated press angle
    lever_ang_damping: float = info(15.0)   # hinge dashpot (velocity-decay rate, 1/s): kills
    #                                         drop-impact jolts in ~3-4 deg of swing while the
    #                                         sustained 2-block press still heels over in ~1-2 s
    # --- info: drawer ----------------------------------------------------------------------------
    drawer_l: float = info(0.160)
    drawer_w: float = info(0.130)
    drawer_wall_t: float = info(0.010)
    drawer_floor_t: float = info(0.010)
    drawer_h: float = info(0.095)
    drawer_mass: float = info(0.90)
    drawer_damping: float = info(1.0)
    # --- info: ballast blocks --------------------------------------------------------------------
    cube_s: float = info(0.040)             # 40 mm steel block (parallel-jaw graspable)
    cube_mass: float = info(0.60)
    n_blocks: int = info(3)
    # --- info: materials -------------------------------------------------------------------------
    slide_mu_s: float = info(0.10)          # slick everywhere sliding (min combine)
    slide_mu_d: float = info(0.08)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (2 x 0.15 + 0.55 = 0.85 = the non-success cap) ---------------------
    w_block: float = info(0.15)
    w_drawer: float = info(0.55)

    # Derived (filled in __post_init__).
    q_stop: float = field(default=None, init=False)     # drawer front x pressed on the rear stop
    theta_seat_deg: float = field(default=None, init=False)  # lever angle with the drawer seated

    def cell_centers_y(self) -> tuple:
        """Lever-local y centres of the three hopper cells."""
        cy = self.cell_hw + self.tray_div_t + (self.tray_hw - self.cell_hw - self.tray_div_t) / 2
        return (-cy, 0.0, cy)

    def __post_init__(self) -> None:
        c = self
        g = 9.81
        # ---- geometry: the rear hard stop IS the seated pose, inside the tolerance --------------
        self.q_stop = c.drawer_l - c.chan_len
        assert 0.0 < self.q_stop < c.q_closed_tol - 0.005, \
            "rear stop must seat the drawer inside q_closed_tol with margin"
        assert abs(c.chan_hw * 2 - c.drawer_w - 0.008) < 1e-9      # 4 mm/side channel clearance
        assert 0.135 > c.slab_t + c.drawer_h + 0.015, "drawer must ride free under the roof"
        # ---- geometry: the press stroke ----------------------------------------------------------
        # nose front surface at heel angle a: x = hinge_x - press_r*sin(a) - nose_r
        sin_seat = (c.hinge_x - self.q_stop - c.nose_r) / c.press_r
        self.theta_seat_deg = math.degrees(math.asin(sin_seat))
        assert c.press_min_deg + 1.5 < self.theta_seat_deg < c.tilt_max_deg - 2.0, \
            "seated press angle must clear press_min and stay inside the joint travel"
        sin_tol = (c.hinge_x - c.q_closed_tol - c.nose_r) / c.press_r
        assert math.degrees(math.asin(sin_tol)) > c.press_min_deg + 1.0, \
            "a drawer at the closed tolerance must already read as pressed"
        # raised blade clears the widest-open drawer (reset band vs wall clearance)
        assert c.hinge_x - c.nose_r >= c.q0_range[1] + 0.005, \
            "raised nose must clear the widest-open drawer face"
        # nose meets the drawer FACE band (10 mm inside its edges) at both stroke ends
        for a in (math.asin((c.hinge_x - c.q0_range[1] - c.nose_r) / c.press_r),
                  math.asin(sin_seat)):
            zc = c.hinge_z - c.press_r * math.cos(a)
            assert c.slab_t + c.nose_r + 0.005 < zc < c.slab_t + c.drawer_h - c.nose_r - 0.005, \
                "press nose must bear on the drawer front face throughout the stroke"
        # raised blade tip clears the slab; fully-heeled nose stays inside the open channel mouth
        assert c.hinge_z - c.press_r - c.nose_r > c.slab_t + 0.008
        a_max = math.radians(c.tilt_max_deg)
        z_max = c.hinge_z - c.press_r * math.cos(a_max)
        assert c.slab_t + c.nose_r + 0.003 < z_max < 0.135 - c.nose_r - 0.003, \
            "at the joint's high stop the nose must hang inside the open mouth, touching nothing"
        # ---- geometry: hopper cells admit and retain the blocks ----------------------------------
        assert 2 * c.cell_hw - c.cube_s >= 0.012, "cell must admit a block with margin (y)"
        assert (c.tray_x1 - c.tray_x0) - c.cube_s >= 0.012, "cell must admit a block (x)"
        assert c.tray_wall_top - c.cube_s / 2 >= 0.010, \
            "a resting block's top must sit below the wall tops (retained at any heel angle)"
        assert (c.tray_hw - c.cell_hw - c.tray_div_t) - c.cube_s >= 0.012  # outer cells (y)
        # dock strips clear the hopper footprint and stay on the plinth
        assert c.dock_y_range[0] > c.tray_hw + c.tray_wall_t + 0.040, \
            "block docks must never sit under the hopper"
        assert c.dock_x_range[1] + 0.030 < 0.48 and c.dock_y_range[1] + 0.030 < 0.31
        assert c.dock_x_range[0] > c.slab_x1 + 0.045, "block docks clear the drawer slab"
        # fallback lattice (used if rejection sampling fails) respects the separation
        assert (c.dock_x_range[1] - c.dock_x_range[0]) / 2 > c.dock_min_sep
        # ---- TORQUE AUDIT (from the authored parts table; + heel = press) ------------------------
        Sx = Sz = 0.0
        for _tag, kind, center, dims, density, _ck in _lever_parts(c):
            vol = dims[0] * dims[1] * dims[2] if kind == "box" \
                else math.pi * dims[0] ** 2 * dims[1]
            m = vol * density
            Sx += m * center[0]
            Sz += m * center[2]
        mb = c.cube_mass
        r_in = c.tray_x0 + c.cube_s / 2                 # block CoM against the inner cell wall
        r_out = c.tray_x1 - c.cube_s / 2                # ... against the outer cell wall
        # one block ANYWHERE in a cell: the lever provably rests UP (>= 0.4 N m margin)
        assert g * (Sx + mb * r_out) < -0.40, \
            f"one-block restoring margin too thin ({g * (Sx + mb * r_out):.3f} N m)"
        # two blocks ANYWHERE in cells: provable lift-off surplus (>= 0.6 N m)
        assert g * (Sx + 2 * mb * r_in) > 0.60, \
            f"two-block driving surplus too thin ({g * (Sx + 2 * mb * r_in):.3f} N m)"
        # ... and at the seated angle the drive still beats sliding friction by >= 2.5x
        a = math.radians(self.theta_seat_deg)
        drive_seat = g * ((Sx + 2 * mb * r_in) * math.cos(a) + Sz * math.sin(a))
        tau_fric = c.drawer_mass * g * c.slide_mu_s * c.press_r * math.cos(a)
        assert drive_seat > 2.5 * tau_fric, \
            f"press must overpower drawer friction ({drive_seat:.3f} vs {tau_fric:.3f} N m)"
        # blade weight must RESTORE (Sz < 0): the press is powered by the ballast, and the
        # empty lever cannot hang at the pressed stop
        assert Sz < -0.02
        assert abs(2 * c.w_block + c.w_drawer - 0.85) < 1e-9


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
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    """(N,) yaw angle of quats (wxyz)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_press_cabinet")
class BallastPressCabinetScene(BaseScene):
    cfg: BallastPressCabinetSceneCfg

    def __init__(self, cfg: BallastPressCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastPressCabinetSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        base_spawn = cls["base"](
            plinth_h=c.plinth_h, chan_hw=c.chan_hw, chan_wall_t=c.chan_wall_t,
            chan_len=c.chan_len, back_t=c.back_t, slab_t=c.slab_t, slab_x1=c.slab_x1,
            hinge_x=c.hinge_x, hinge_z=c.hinge_z, post_y=c.post_y, base_mass=c.base_mass,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        lever_spawn = cls["lever"](
            plinth_h=c.plinth_h, hinge_x=c.hinge_x, hinge_z=c.hinge_z, press_r=c.press_r,
            nose_r=c.nose_r, blade_w=c.blade_w, arm_x0=c.arm_x0, arm_x1=c.arm_x1,
            tray_x0=c.tray_x0, tray_x1=c.tray_x1, tray_hw=c.tray_hw, cell_hw=c.cell_hw,
            tray_wall_t=c.tray_wall_t, tray_div_t=c.tray_div_t, tray_floor_t=c.tray_floor_t,
            tray_wall_top=c.tray_wall_top, cube_s=c.cube_s, axle_r=c.axle_r,
            axle_len=c.axle_len, cw_x=c.cw_x, cw_size=c.cw_size,
            struct_density=c.struct_density, cw_density=c.cw_density,
            tilt_min_deg=c.tilt_min_deg, tilt_max_deg=c.tilt_max_deg,
            lever_ang_damping=c.lever_ang_damping,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        drawer_spawn = cls["drawer"](
            drawer_l=c.drawer_l, drawer_w=c.drawer_w, drawer_wall_t=c.drawer_wall_t,
            drawer_floor_t=c.drawer_floor_t, drawer_h=c.drawer_h, drawer_mass=c.drawer_mass,
            drawer_damping=c.drawer_damping, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        # template poses: the lever MUST spawn consistent with its authored joint frames
        # (base at the origin, yaw 0; lever origin at the hinge point, heel angle 0)
        hinge = (c.hinge_x, 0.0, c.plinth_h + c.hinge_z)

        def block_cfg(tag: str, y: float) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + tag,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,   # kills GPU phantom creep
                        max_depenetration_velocity=0.5,
                        linear_damping=0.02, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0,
                        friction_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.38, 0.38, 0.44), metallic=0.8)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, y, 0.05)))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base", spawn=base_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "lever": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lever", spawn=lever_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=hinge)),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.6, 0.05))),
            "block_a": block_cfg("BlockA", 0.9),
            "block_b": block_cfg("BlockB", 1.0),
            "block_c": block_cfg("BlockC", 1.1),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.base: RigidObject = env.iscene["base"]
        self.lever: RigidObject = env.iscene["lever"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.blocks: list[RigidObject] = [
            env.iscene["block_a"], env.iscene["block_b"], env.iscene["block_c"]]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.q0 = torch.zeros(n, device=dev)            # initial drawer opening
        # latches (partial credit survives transients; success is judged live)
        self._fblock = torch.zeros(n, 3, device=dev)    # per-block "rode in a cell" latch
        self._streak = torch.zeros(n, 3, dtype=torch.long, device=dev)
        self._fdrawer = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the base (free yaw + xy jitter) and hang the lever on its
        hinge at heel 0 — the whole jointed pair is written together with consistent
        poses (the lever origin sits ON the hinge axis, so any heel angle is a pure pose
        write). Then seat the drawer at a sampled opening q0 and scatter the three
        ballast blocks over the dock strips (rejection-sampled separation with a
        deterministic fallback lattice)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, device=dev)                   # burn the degenerate first draw
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_b = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        bp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_b
        self.base.write_root_state_to_sim(st, env_ids)

        # lever: heel 0 (it settles onto its raised stop at tilt_min_deg)
        hinge = torch.zeros(m, 3, device=dev)
        hinge[:, 0], hinge[:, 2] = c.hinge_x, c.plinth_h + c.hinge_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin + quat_apply(q_b, hinge)
        st[:, 3:7] = q_b
        self.lever.write_root_state_to_sim(st, env_ids)

        # sampled knob: drawer opening q0 (front face x in the station frame)
        q0 = c.q0_range[0] + torch.rand(m, device=dev) * (c.q0_range[1] - c.q0_range[0])
        self.q0[env_ids] = q0
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = q0 - c.drawer_l / 2
        loc[:, 2] = c.plinth_h + c.slab_t + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin + quat_apply(q_b, loc)
        st[:, 3:7] = q_b
        self.drawer.write_root_state_to_sim(st, env_ids)

        # ballast blocks: rejection-sample xy on the two dock strips (deterministic
        # lattice fallback), free per-block yaw, resting on the deck
        x0, x1 = c.dock_x_range
        y0, y1 = c.dock_y_range
        xy = torch.zeros(m, 3, 2, device=dev)
        for _try in range(20):
            xs = x0 + torch.rand(m, 3, device=dev) * (x1 - x0)
            side = torch.where(torch.rand(m, 3, device=dev) < 0.5, -1.0, 1.0)
            ys = side * (y0 + torch.rand(m, 3, device=dev) * (y1 - y0))
            cand = torch.stack([xs, ys], dim=-1)
            d01 = (cand[:, 0] - cand[:, 1]).norm(dim=-1)
            d02 = (cand[:, 0] - cand[:, 2]).norm(dim=-1)
            d12 = (cand[:, 1] - cand[:, 2]).norm(dim=-1)
            ok = (d01 > c.dock_min_sep) & (d02 > c.dock_min_sep) & (d12 > c.dock_min_sep)
            if _try == 0:
                xy = cand
                pend = ~ok
            else:
                xy = torch.where(pend[:, None, None], cand, xy)
                pend = pend & ~ok
            if not pend.any():
                break
        if pend.any():                                   # deterministic by-construction fallback
            xm = (x0 + x1) / 2
            lat = torch.tensor([[x0 + 0.02, y0 + 0.04], [xm, -(y0 + 0.04)], [x1 - 0.02, y0 + 0.04]],
                               device=dev).expand(m, 3, 2)
            xy = torch.where(pend[:, None, None], lat, xy)
        byaw = (torch.rand(m, 3, device=dev) * 2 - 1) * math.pi
        for k, blk in enumerate(self.blocks):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0], loc[:, 1] = xy[:, k, 0], xy[:, k, 1]
            loc[:, 2] = c.plinth_h + c.cube_s / 2 + 0.002
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = bp + origin + quat_apply(q_b, loc)
            st[:, 3:7] = _qmul(q_b, _qz(byaw[:, k]))
            blk.write_root_state_to_sim(st, env_ids)

        self._fblock[env_ids] = 0.0
        self._streak[env_ids] = 0
        self._fdrawer[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "lever": self.lever.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "blocks": [b.data.root_state_w[env_ids].clone() for b in self.blocks],
            "q0": self.q0[env_ids].clone(),
            "fblock": self._fblock[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
            "fdrawer": self._fdrawer[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.lever.write_root_state_to_sim(state["lever"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        for b, s in zip(self.blocks, state["blocks"]):
            b.write_root_state_to_sim(s, env_ids)
        self.q0[env_ids] = state["q0"]
        self._fblock[env_ids] = state["fblock"]
        self._streak[env_ids] = state["streak"]
        self._fdrawer[env_ids] = state["fdrawer"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden CABINET stands on a dark plinth. Its single drawer (light wood, "
            f"{c.drawer_w * 1000:.0f} mm wide) sticks OUT of the cabinet face by roughly "
            f"{c.q0_range[0] * 100:.0f}-{c.q0_range[1] * 100:.0f} cm — the opening varies by "
            f"episode, and the whole station's position and heading also vary. Above and just "
            f"in front of the cabinet mouth, a PRESS LEVER pivots between two steel-blue "
            f"cradle posts: its long GREEN arm reaches out over the deck and ends in an "
            f"open-top BALLAST HOPPER with three square cells; its short arm is a solid steel "
            f"COUNTERWEIGHT; and a steel PRESS BLADE with an orange rounded nose hangs "
            f"straight down from the pivot, poised in front of the drawer's face. Three "
            f"{c.cube_s * 1000:.0f} mm STEEL BLOCKS ({c.cube_mass:.2f} kg each) rest on the "
            f"deck beside the hopper.\n"
            f"The machine is a WEIGHT-POWERED PRESS: the counterweight holds the empty lever "
            f"up against its stop, and one block in the hopper is not enough to tip it. TWO "
            f"blocks loaded into the hopper cells overcome the counterweight; the lever heels "
            f"over and its blade sweeps the drawer to its rear stop — the ballast's weight, "
            f"not the hand, supplies the closing force, and the press holds only while the "
            f"ballast stays aboard (a hand-pressed lever swings back up when released).\n"
            f"Goal: load at least two steel blocks into the hopper cells so the lever presses "
            f"the drawer shut. Success is the settled end state: the drawer front within "
            f"{c.q_closed_tol * 1000:.0f} mm of the cabinet face, the lever heeled past "
            f"{c.press_min_deg:.0f} degrees, and at least two blocks riding inside the hopper "
            f"cells. All three are required — a drawer pushed shut by hand with the hopper "
            f"empty does NOT succeed, and blocks parked anywhere but inside the cells do not "
            f"count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick the steel blocks off the deck and drop at least two of them into the "
            "green hopper cells on the press lever, so the ballast heels the lever over "
            "and its blade presses the cabinet drawer shut. Finish with the drawer flush, "
            "the lever heeled, and the blocks riding in the hopper."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.base.data.root_quat_w,
                                  pos_w - self.base.data.root_pos_w)

    def _lever_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.lever.data.root_quat_w,
                                  pos_w - self.lever.data.root_pos_w)

    def drawer_q(self) -> torch.Tensor:
        """(N,) drawer opening: the front face's x in the station frame
        (0 = flush with the cabinet face plane)."""
        return self._station_local(self.drawer.data.root_pos_w)[:, 0] + self.cfg.drawer_l / 2

    def drawer_in_channel(self) -> torch.Tensor:
        """(N,) bool: drawer genuinely riding in its channel (guards every drawer clause
        and the drawer-progress latch against a drawer stolen out of the cabinet)."""
        c = self.cfg
        loc = self._station_local(self.drawer.data.root_pos_w)
        return (loc[:, 1].abs() < c.lane_y_tol) \
            & ((loc[:, 2] - c.plinth_h - c.slab_t).abs() < 0.030) \
            & (loc[:, 0] + c.drawer_l / 2 > -0.020) \
            & (loc[:, 0] + c.drawer_l / 2 < 0.180)

    def drawer_closed(self) -> torch.Tensor:
        """(N,) bool: drawer seated — front face within q_closed_tol of the cabinet face
        plane, riding in its channel."""
        return (self.drawer_q() < self.cfg.q_closed_tol) & self.drawer_in_channel()

    def lever_theta_deg(self) -> torch.Tensor:
        """(N,) lever heel angle in degrees (0 = raised template pose, + = pressing)."""
        qb = self.base.data.root_quat_w
        qb_inv = qb * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qb.device)
        rel = _qmul(qb_inv, self.lever.data.root_quat_w)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return ang

    def blocks_in_cells(self) -> torch.Tensor:
        """(N, 3) bool: each block riding INSIDE a hopper cell, in the LEVER frame (the
        test rides with the heel angle)."""
        c = self.cfg
        res = []
        for blk in self.blocks:
            loc = self._lever_local(blk.data.root_pos_w)
            res.append((loc[:, 0] > c.tray_x0 + c.cube_s / 2 - 0.012)
                       & (loc[:, 0] < c.tray_x1 - c.cube_s / 2 + 0.012)
                       & (loc[:, 1].abs() < c.tray_hw + 0.002)
                       & (loc[:, 2] > -c.cube_s / 2 - 0.012)
                       & (loc[:, 2] < c.tray_wall_top))
        return torch.stack(res, dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer, lever and blocks at rest (thresholds sit above the GPU
        phantom-velocity band)."""
        c = self.cfg
        ok = (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_drawer) \
            & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < 0.5) \
            & (self.lever.data.root_ang_vel_w.norm(dim=-1) < c.settle_lever_w)
        for blk in self.blocks:
            ok = ok & (blk.data.root_lin_vel_w.norm(dim=-1) < c.settle_block)
        return ok

    def _finite(self) -> torch.Tensor:
        ps = [self.drawer.data.root_pos_w, self.lever.data.root_pos_w] \
            + [b.data.root_pos_w for b in self.blocks]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        inc = self.blocks_in_cells() & fin[:, None]
        self._streak = torch.where(inc, self._streak + 1, torch.zeros_like(self._streak))
        self._fblock = torch.maximum(self._fblock, (self._streak >= 30).float())
        q = self.drawer_q()
        span = (self.q0 - self.cfg.q_stop).clamp(min=1e-6)
        fd = ((self.q0 - q) / span).clamp(0.0, 1.0)
        fd = torch.where(self.drawer_in_channel() & fin, fd, torch.zeros_like(fd))
        self._fdrawer = torch.maximum(self._fdrawer, fd)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the whole machine state, LIVE on the settled scene — drawer seated
        in its channel AND the lever heeled past press_min_deg AND at least two ballast
        blocks riding inside the hopper cells, settled and finite. The counterweight
        makes the conjunction honest: only resident ballast can HOLD the heel, so a
        hand-pressed lever (released) or a hand-closed drawer (lever up, hopper empty)
        can never satisfy all three clauses at rest."""
        self._update_latches()
        return self.drawer_closed() \
            & (self.lever_theta_deg() >= self.cfg.press_min_deg) \
            & (self.blocks_in_cells().sum(dim=1) >= 2) \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 per latched ballast block (max two) + 0.55 *
        latched drawer-closure fraction (channel-guarded), capped at 0.85; exactly 1.0
        iff success() holds live. Doing nothing scores ~0. The seed's end state — the
        drawer pushed shut by hand, hopper empty — earns the drawer credit only (~0.55):
        the ballast clauses never fire."""
        c = self.cfg
        self._update_latches()
        nb = self._fblock.sum(dim=1).clamp(max=2.0)
        base = (c.w_block * nb + c.w_drawer * self._fdrawer).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ballast_press_cabinet", robot="null"))
