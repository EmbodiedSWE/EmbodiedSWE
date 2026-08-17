"""WeightCrankScene — close the cabinet's top drawer by LOADING A COUNTERWEIGHT
(sim_gen task `libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i163`).

Derived from libero_90/libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet,
but STRATEGICALLY different: in the seed the hand pushes the drawer front along
its prismatic travel — the robot's arm supplies both the guidance and the energy,
directly on the judged part. Here the robot never has to touch the drawer OR
drive any mechanism through its travel: it performs a PICK-AND-PLACE of a free
object. A gravity-biased BELL CRANK hangs on a revolute pivot in a gallows frame
in front of the cabinet: a horizontal PAN ARM (with a deep receiving pocket)
reaches out toward the robot, and a hanging PRESS ARM ends in a red roller nose
facing the drawer's front plate. Dropping the heavy BRASS cube into the pan
pocket makes the payload's own weight overpower the crank's return bias: the
crank rotates down its arc, the nose cam-presses the drawer front, and the
drawer translates shut — the ENERGY comes from the payload's gravity, not from
the robot's hand. A visually similar light FOAM cube (the decoy) cannot
overpower the bias: mass discrimination is part of the task. The goal state is
the drawer seated AND the weighted press holding it (crank at its down-stop with
the brass weight riding the pan) — the seed's exact skill (hand-push the drawer
shut) leaves the pan empty and the crank up, and can never succeed.

No stored energy against the goal: the crank's authored CoM offset biases it
UP (against its lower joint stop) when unloaded, the drawer rides a horizontal
prismatic joint and holds any position, and the loaded terminal state is a
static force balance (weight-on-pan vs crank-on-stop + nose-on-drawer), so the
outcome persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body
never self-collide): cabinet + apron bench + gallows frame (KINEMATIC, fixed
pose — bind-time joint anchors on a kinematic body are world-fixed), drawer
(DYNAMIC: open box + smooth front plate, on a prismatic joint whose limits are
the hard stops), bell crank (DYNAMIC: hanging press plank + roller nose +
pan arm + deep pocket, on a revolute joint through the gallows axle), plus the
brass weight and the foam decoy (free cubes on the apron). The nose->plate cam
contact carries a bound slick material (mu ~0.06, min combine) — PhysX's
default ~0.5 friction would drag the plate down and load the joint instead of
sliding cleanly.

Per-episode randomization (readback-verified by smoke): the drawer's initial
opening q0, the two cubes' apron poses (x, yaw) and WHICH SIDE each cube
spawns on (weight left vs right).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.30  loaded — brass weight ever riding the pan assembly (latched)
  0.25  x latched max crank closure fraction theta/theta_max (on-pivot guarded)
  0.35  x latched max drawer closure fraction (q0-q)/q0 (channel guarded)
capped at 0.90; exactly 1.0 iff success(): brass weight riding the pan, crank
at its down-stop (>= crank_goal_deg, on its pivot), drawer seated
(q <= q_goal, in its channel), all settled and finite — live. Null policy ~0.
The seed's strategy (hand-push the drawer shut) earns ~0.35 and never succeeds;
hand-cranking the press down without the weight earns <= 0.60 and never
succeeds (the bias returns the crank up the moment the hand lets go).

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

_G = 9.81


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


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _cyl(stage, path: str, *, axis: str, radius: float, half: float, center, color,
         collide: Callable):
    """Cylinder child (native PhysX cylinder collision), axis-aligned."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(2 * half))
    r, h = float(radius), float(half)
    ext = {"X": [(-h, -r, -r), (h, r, r)], "Y": [(-r, -h, -r), (r, h, r)],
           "Z": [(-r, -r, -h), (r, r, h)]}[axis]
    cyl.CreateExtentAttr([Gf.Vec3f(*ext[0]), Gf.Vec3f(*ext[1])])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float, com=None) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station at `prim_path`: KINEMATIC compound. Local frame: x=0 is
    the cabinet FRONT face plane (+x = out of the cabinet, toward the robot),
    z=0 is the ground. One drawer cavity opens at the front; an apron bench
    stands in front below the drawer; a gallows frame (two outboard posts + a
    high crossbeam + axle stubs) straddles the crank's pivot line."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # cabinet box (single cavity y +/- wall_y0, z cav_z0..cav_z1)
    _span(stage, f"{prim_path}/side_p", x=(c.back_x0, 0.0), y=(c.wall_y0, c.wall_y1),
          z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.wall_y1, -c.wall_y0),
          z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.back_x0 - 0.02, c.back_x0),
          y=(-c.wall_y1, c.wall_y1), z=(0.0, c.top_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/base", x=(c.back_x0, 0.0), y=(-c.wall_y0, c.wall_y0),
          z=(0.0, c.cav_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/top", x=(c.back_x0, 0.0), y=(-c.wall_y0, c.wall_y0),
          z=(c.cav_z1, c.top_z1), color=c.trim_color, collide=collide)
    # apron bench in front, below the drawer travel (cube staging area)
    _span(stage, f"{prim_path}/apron", x=(0.0, c.apron_x1), y=(-c.wall_y0, c.wall_y0),
          z=(0.0, c.apron_z1), color=c.trim_color, collide=collide)
    # gallows: outboard posts + crossbeam, well clear of the drawer and the swing
    _span(stage, f"{prim_path}/post_p", x=(c.post_x0, c.post_x1),
          y=(c.post_y0, c.post_y1), z=(0.0, c.beam_z1), color=c.steel_color, collide=collide)
    _span(stage, f"{prim_path}/post_n", x=(c.post_x0, c.post_x1),
          y=(-c.post_y1, -c.post_y0), z=(0.0, c.beam_z1), color=c.steel_color, collide=collide)
    _span(stage, f"{prim_path}/beam", x=(c.post_x0, c.post_x1),
          y=(-c.post_y1, c.post_y1), z=(c.beam_z0, c.beam_z1),
          color=c.steel_color, collide=collide)
    # axle stubs from the posts to the crank hub (crank<->station collisions are
    # filtered by the revolute joint, so the stub may reach the plank flank)
    px, pz = c.pivot_x, c.pivot_z
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        y0, y1 = sgn * (c.crank_hy + 0.002), sgn * c.post_y0
        _cyl(stage, f"{prim_path}/axle_{tag}", axis="Y", radius=0.012,
             half=abs(y1 - y0) / 2, center=(px, (y0 + y1) / 2, pz),
             color=c.steel_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at its CLOSED pose, root origin = station origin (so the
    prismatic joint anchors coincide and the joint coordinate IS the opening).
    Open box + smooth front PLATE (no handle — the plate is the cam surface the
    crank's roller nose presses)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/plate", x=(0.0, c.lip_t), y=(-c.lip_hy, c.lip_hy),
          z=(c.lip_z0, c.lip_z1), color=c.plate_color, collide=collide)
    _span(stage, f"{prim_path}/floor", x=(c.body_x0, 0.0), y=(-c.body_hy, c.body_hy),
          z=(c.floor_z0, c.floor_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(c.body_x0, c.body_x0 + 0.015),
          y=(-c.body_hy, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(c.body_x0, 0.0), y=(c.body_hy - 0.015, c.body_hy),
          z=(c.floor_z1, c.wall_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(c.body_x0, 0.0), y=(-c.body_hy, -c.body_hy + 0.015),
          z=(c.floor_z1, c.wall_z1), color=c.box_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_crank(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bell crank with root origin AT the pivot. Hanging press plank
    (z in [-hang_len, 0]) ending in a roller NOSE (cylinder, axis Y — the sole
    cam contact, proud of the plank face everywhere); horizontal pan arm (+x)
    carrying a deep receiving POCKET at its tip. MassAPI CoM is authored at
    (-com_x, 0, 0): gravity biases the crank UP against its lower joint stop
    when unloaded (no springs, no stored energy against the goal)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # hanging press arm + roller nose
    _span(stage, f"{prim_path}/hang", x=(-0.015, 0.015), y=(-c.crank_hy, c.crank_hy),
          z=(-c.hang_len, 0.015), color=c.arm_color, collide=collide)
    _cyl(stage, f"{prim_path}/nose", axis="Y", radius=c.nose_r, half=c.nose_hy,
         center=(0.0, 0.0, -c.hang_len), color=c.nose_color, collide=collide)
    # pan arm + deep pocket (floor + 4 walls; inner well is pocket_in_*)
    _span(stage, f"{prim_path}/pan_arm", x=(-0.015, c.pkt_x0), y=(-c.crank_hy, c.crank_hy),
          z=(-0.015, 0.015), color=c.arm_color, collide=collide)
    _span(stage, f"{prim_path}/pkt_floor", x=(c.pkt_x0, c.pkt_x1), y=(-c.crank_hy, c.crank_hy),
          z=(-0.015, -0.005), color=c.pkt_color, collide=collide)
    _span(stage, f"{prim_path}/pkt_wxn", x=(c.pkt_x0, c.pkt_in_x0), y=(-c.crank_hy, c.crank_hy),
          z=(-0.005, c.pkt_top), color=c.pkt_color, collide=collide)
    _span(stage, f"{prim_path}/pkt_wxp", x=(c.pkt_in_x1, c.pkt_x1), y=(-c.crank_hy, c.crank_hy),
          z=(-0.005, c.pkt_top), color=c.pkt_color, collide=collide)
    _span(stage, f"{prim_path}/pkt_wyp", x=(c.pkt_in_x0, c.pkt_in_x1),
          y=(c.pkt_in_hy, c.crank_hy), z=(-0.005, c.pkt_top),
          color=c.pkt_color, collide=collide)
    _span(stage, f"{prim_path}/pkt_wyn", x=(c.pkt_in_x0, c.pkt_in_x1),
          y=(-c.crank_hy, -c.pkt_in_hy), z=(-0.005, c.pkt_top),
          color=c.pkt_color, collide=collide)
    _dyn_body(root, c.mass, 0.2, c.ang_damp, com=(-c.com_x, 0.0, 0.0))
    struct = _mk_material(prim_path, "struct", c.struct_mu_s, c.struct_mu_d, "average")
    bind_physics_material(prim_path, struct)
    # the roller nose alone is slick ("min" combine wins the pair with the plate)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(f"{prim_path}/nose", slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            wall_y0: float = 0.13
            wall_y1: float = 0.16
            back_x0: float = -0.44
            cav_z0: float = 0.49
            cav_z1: float = 0.67
            top_z1: float = 0.70
            apron_x1: float = 0.30
            apron_z1: float = 0.40
            post_x0: float = 0.16
            post_x1: float = 0.22
            post_y0: float = 0.165
            post_y1: float = 0.195
            beam_z0: float = 0.95
            beam_z1: float = 0.97
            pivot_x: float = 0.19
            pivot_z: float = 0.845
            crank_hy: float = 0.05
            body_color: tuple = (0.52, 0.38, 0.24)
            trim_color: tuple = (0.42, 0.30, 0.19)
            steel_color: tuple = (0.25, 0.27, 0.30)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.60
            wood_mu_d: float = 0.55

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            lip_t: float = 0.018
            lip_hy: float = 0.145
            lip_z0: float = 0.485
            lip_z1: float = 0.675
            body_x0: float = -0.20
            body_hy: float = 0.115
            floor_z0: float = 0.492
            floor_z1: float = 0.500
            wall_z1: float = 0.63
            mass: float = 0.60
            lin_damp: float = 6.0
            ang_damp: float = 4.0
            plate_color: tuple = (0.42, 0.44, 0.50)
            box_color: tuple = (0.68, 0.56, 0.40)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        @configclass
        class CrankSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crank)
            hang_len: float = 0.30
            nose_r: float = 0.02
            nose_hy: float = 0.04
            crank_hy: float = 0.05
            pkt_x0: float = 0.20
            pkt_x1: float = 0.32
            pkt_in_x0: float = 0.21
            pkt_in_x1: float = 0.31
            pkt_in_hy: float = 0.04
            pkt_top: float = 0.055
            mass: float = 0.50
            com_x: float = 0.07
            ang_damp: float = 1.5
            arm_color: tuple = (0.30, 0.45, 0.70)
            nose_color: tuple = (0.85, 0.15, 0.10)
            pkt_color: tuple = (0.90, 0.65, 0.15)
            contact_offset: float = 0.0015
            struct_mu_s: float = 0.50
            struct_mu_d: float = 0.45
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
        _SPAWNER_CACHE["crank"] = CrankSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WeightCrankSceneCfg(BaseCfg):
    """Config for `WeightCrankScene`. The transmission contract is asserted in
    `__post_init__`: at the crank's down-stop the roller nose really presses the
    drawer inside the success tolerance; at rest the nose really clears the
    drawer at every sampled (and reachable) opening; the nose really stays on
    the front plate through the whole arc; the brass weight's torque really
    overpowers the bias with margin while the foam decoy's never can; the
    pocket really captures a cube at full tilt; and every clearance
    (posts/drawer, beam/pocket, apron/drawer, nose/apron) is positive."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    q_goal: float = tunable(0.010)        # drawer opening for "seated" (m)
    crank_goal_deg: float = tunable(27.0)  # crank angle for "down" (deg; stop at theta_max)
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    settle_ang: float = tunable(0.30)     # max |ang vel| of the crank when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    q0_lo: float = tunable(0.085)         # drawer initial opening, low (m)
    q0_hi: float = tunable(0.122)         # drawer initial opening, high (m)
    cube_x: tuple = tunable((0.055, 0.245))   # cube spawn x band on the apron
    cube_y: tuple = tunable((0.045, 0.080))   # cube spawn |y| band (sides swap randomly)
    weight_mass: float = tunable(0.90)    # brass cube mass (kg)
    decoy_mass: float = tunable(0.045)    # foam cube mass (kg)

    # --- info: station (local frame: x=0 cabinet face, +x = out, z=0 ground) ---------------------
    wall_y0: float = info(0.13)
    wall_y1: float = info(0.16)
    back_x0: float = info(-0.44)
    cav_z0: float = info(0.49)
    cav_z1: float = info(0.67)
    top_z1: float = info(0.70)
    apron_x1: float = info(0.30)
    apron_z1: float = info(0.40)          # apron top: the cube staging surface
    post_x0: float = info(0.16)
    post_x1: float = info(0.22)
    post_y0: float = info(0.165)
    post_y1: float = info(0.195)
    beam_z0: float = info(0.95)
    beam_z1: float = info(0.97)
    # --- info: drawer (authored closed; prismatic axis X, limits = hard stops) -------------------
    lip_t: float = info(0.018)            # front plate thickness (its face = q + lip_t)
    lip_hy: float = info(0.145)
    lip_z0: float = info(0.485)
    lip_z1: float = info(0.675)
    floor_z0: float = info(0.492)
    stroke: float = info(0.142)           # prismatic upper limit (m)
    drawer_mass: float = info(0.60)
    # --- info: crank (root/pivot at (pivot_x, 0, pivot_z); revolute axis Y) ----------------------
    pivot_x: float = info(0.19)
    pivot_z: float = info(0.845)
    hang_len: float = info(0.30)          # pivot -> nose centre
    nose_r: float = info(0.02)
    theta_max_deg: float = info(30.0)     # revolute upper limit (down-stop; lower limit 0)
    crank_hy: float = info(0.05)
    crank_mass: float = info(0.50)
    com_x: float = info(0.07)             # authored CoM offset (-x): the UP bias
    pkt_x0: float = info(0.20)            # pocket outer span (crank-local)
    pkt_x1: float = info(0.32)
    pkt_in_x0: float = info(0.21)         # pocket inner well
    pkt_in_x1: float = info(0.31)
    pkt_in_hy: float = info(0.04)
    pkt_top: float = info(0.055)          # pocket wall top (crank-local z)
    # --- info: cubes ------------------------------------------------------------------------------
    cube_s: float = info(0.05)            # cube edge (both cubes)
    # --- info: "riding the pan" box (crank-local; load clause + latch) ---------------------------
    load_x: tuple = info((0.05, 0.33))
    load_hy: float = info(0.055)
    load_z: tuple = info((-0.02, 0.10))
    # --- info: materials --------------------------------------------------------------------------
    slide_mu_s: float = info(0.06)        # nose<->plate cam: default ~0.5 would drag/jam
    slide_mu_d: float = info(0.05)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.30 + 0.25 + 0.35 = 0.90 = the non-success cap) ------------------
    w_load: float = info(0.30)
    w_crank: float = info(0.25)
    w_drawer: float = info(0.35)

    def __post_init__(self) -> None:
        th = math.radians(self.theta_max_deg)
        # cam relation: nose surface plane x at angle t is
        #   pivot_x - hang_len*sin(t) - nose_r  (nose centre local (0,0,-hang_len))
        # so the drawer is pressed to q(t) = pivot_x - hang_len*sin(t) - nose_r - lip_t.
        q_rest = self.pivot_x - self.nose_r - self.lip_t
        q_press = q_rest - self.hang_len * math.sin(th)
        assert 0.001 <= q_press <= self.q_goal - 0.005, \
            f"down-stop must press the drawer inside tolerance (q_press={q_press:.4f})"
        # at the crank-goal threshold the press must already be plausible: the stop
        # rest angle (theta_max) is > crank_goal, and at crank_goal+1.5deg q < q_goal
        qg = q_rest - self.hang_len * math.sin(math.radians(self.crank_goal_deg + 1.5))
        assert self.crank_goal_deg <= self.theta_max_deg - 2.0
        assert qg <= self.q_goal + 0.008, "goal angle band must sit near the seated band"
        # the drawer's own lower stop (0) is DEEPER than the press: terminal chain is
        # crank-on-its-stop + nose-on-plate, a static force balance
        assert q_press > 0.0005
        # at rest the nose clears the drawer at every reachable opening
        assert q_rest >= self.q0_hi + 0.02, "nose must clear the widest sampled opening"
        assert self.stroke <= q_rest - 0.008, "nose must clear the drawer even at full stroke"
        assert self.q0_lo < self.q0_hi and self.q0_lo >= 0.06
        assert self.q0_hi <= self.stroke - 0.015
        # nose stays on the front plate through the whole arc (with radius margin)
        z_lo = self.pivot_z - self.hang_len - self.nose_r
        z_hi = self.pivot_z - self.hang_len * math.cos(th) + self.nose_r
        assert z_lo >= self.lip_z0 + 0.005, "nose must start on the plate"
        assert z_hi <= self.lip_z1 - 0.005, "nose must end on the plate at full swing"
        # torque contract (about the pivot): brass overpowers the bias 3x even at the
        # innermost pocket radius and full tilt; foam never reaches half the bias
        tau_bias = self.crank_mass * _G * self.com_x
        r_min = self.pkt_in_x0 + self.cube_s / 2
        tau_w = self.weight_mass * _G * r_min * math.cos(th)
        assert tau_w >= 3.0 * (tau_bias + 0.10), \
            f"brass must overpower the bias 3x (tau_w={tau_w:.3f}, bias={tau_bias:.3f})"
        tau_d = self.decoy_mass * _G * self.load_x[1]
        assert tau_d <= 0.5 * tau_bias, \
            f"foam decoy must never reach half the bias (tau_d={tau_d:.3f})"
        # pocket captivity: a seated cube's top stays below the wall top; clearances
        assert self.pkt_top - (-0.005 + self.cube_s) >= 0.008, "pocket must capture the cube"
        assert (self.pkt_in_x1 - self.pkt_in_x0) - self.cube_s >= 0.03
        assert 2 * self.pkt_in_hy - self.cube_s >= 0.02
        # load box contains the pocket well
        assert self.load_x[0] < self.pkt_in_x0 and self.load_x[1] > self.pkt_in_x1
        assert self.load_hy > self.pkt_in_hy and self.load_z[1] > self.pkt_top
        # gallows clearances: beam above the pocket walls; drop corridor clear in x;
        # posts outboard of the drawer plate; axle line inside the posts span
        assert self.beam_z0 - (self.pivot_z + self.pkt_top) >= 0.03
        assert (self.pivot_x + self.pkt_x0) - self.post_x1 >= 0.10, "drop corridor clear of beam"
        assert self.post_y0 - self.lip_hy >= 0.015, "posts clear the drawer plate"
        assert self.post_x0 <= self.pivot_x <= self.post_x1
        # apron: cubes stand clear below the drawer travel and the nose swing
        assert self.apron_z1 + self.cube_s <= self.floor_z0 - 0.03, "cubes clear the drawer"
        assert z_lo - (self.apron_z1 + self.cube_s) >= 0.05, "nose swing clears staged cubes"
        assert self.cube_x[1] + self.cube_s / 2 <= self.apron_x1 - 0.02
        assert self.cube_y[1] + self.cube_s / 2 <= self.wall_y0 - 0.02
        assert self.cube_y[0] >= 0.03, "side split must separate the two cubes"
        assert 2 * self.cube_y[0] >= self.cube_s * math.sqrt(2.0) + 0.015, \
            "cubes can never spawn overlapping across the centreline"
        # cube spawn band clear of the posts in y and never under the nose rest point
        assert self.cube_y[1] + self.cube_s / 2 <= self.post_y0 - 0.02
        assert abs(self.w_load + self.w_crank + self.w_drawer - 0.90) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("weight_crank_cabinet")
class WeightCrankScene(BaseScene):
    cfg: WeightCrankSceneCfg

    def __init__(self, cfg: WeightCrankSceneCfg | None = None) -> None:
        super().__init__(cfg or WeightCrankSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            wall_y0=c.wall_y0, wall_y1=c.wall_y1, back_x0=c.back_x0, cav_z0=c.cav_z0,
            cav_z1=c.cav_z1, top_z1=c.top_z1, apron_x1=c.apron_x1, apron_z1=c.apron_z1,
            post_x0=c.post_x0, post_x1=c.post_x1, post_y0=c.post_y0, post_y1=c.post_y1,
            beam_z0=c.beam_z0, beam_z1=c.beam_z1, pivot_x=c.pivot_x, pivot_z=c.pivot_z,
            crank_hy=c.crank_hy, contact_offset=c.contact_offset)
        drawer_spawn = cls["drawer"](
            mass=c.drawer_mass, lip_t=c.lip_t, lip_hy=c.lip_hy, lip_z0=c.lip_z0,
            lip_z1=c.lip_z1, floor_z0=c.floor_z0, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        crank_spawn = cls["crank"](
            hang_len=c.hang_len, nose_r=c.nose_r, crank_hy=c.crank_hy, pkt_x0=c.pkt_x0,
            pkt_x1=c.pkt_x1, pkt_in_x0=c.pkt_in_x0, pkt_in_x1=c.pkt_in_x1,
            pkt_in_hy=c.pkt_in_hy, pkt_top=c.pkt_top, mass=c.crank_mass, com_x=c.com_x,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        def cube(color, mass):
            return sim_utils.CuboidCfg(
                size=(c.cube_s, c.cube_s, c.cube_s),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=rigid, collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.60, dynamic_friction=0.55, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # Drawer/crank are authored IN PLACE at their closed/rest poses: the
            # bind-time joints anchor at these authored positions.
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "crank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crank", spawn=crank_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pivot_x, 0.0, c.pivot_z))),
            "weight": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Weight",
                spawn=cube((0.78, 0.60, 0.16), c.weight_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.6, 0.05))),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=cube((0.93, 0.93, 0.90), c.decoy_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, -0.6, 0.05))),
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
        self.station: RigidObject = env.iscene["station"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.crank: RigidObject = env.iscene["crank"]
        self.weight: RigidObject = env.iscene["weight"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.q0 = torch.zeros(n, device=dev)          # initial drawer opening
        # latches (partial credit survives transients; success is judged live)
        self._floaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._fcrank = torch.zeros(n, device=dev)     # max theta/theta_max fraction
        self._fdrawer = torch.zeros(n, device=dev)    # max (q0-q)/q0 fraction
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED poses
        (drawer root coincides with the station root; the crank root is its
        pivot). Joint LIMITS are the hard stops (drawer 0..stroke; crank
        0..theta_max down). Joint collision filtering only disables the
        station<->follower pair, so nose<->drawer cam contact still collides."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.stroke))
            r = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/crank_pivot")
            r.CreateBody0Rel().SetTargets([f"{base}/Station"])
            r.CreateBody1Rel().SetTargets([f"{base}/Crank"])
            r.CreateCollisionEnabledAttr(False)
            r.CreateAxisAttr("Y")
            r.CreateLocalPos0Attr(Gf.Vec3f(float(c.pivot_x), 0.0, float(c.pivot_z)))
            r.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            r.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # +angle about +Y = pan arm down / nose toward the drawer. Degrees.
            r.CreateLowerLimitAttr(0.0)
            r.CreateUpperLimitAttr(float(c.theta_max_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: station re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed — the station must never move), crank at rest
        against its up-stop (0 deg), drawer at a SAMPLED opening q0, brass
        weight and foam decoy at sampled apron poses with RANDOM side
        assignment, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, dx, dy, dz, yaw=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if yaw is None:
                s[:, 3] = 1.0
            else:
                s[:, 3] = torch.cos(yaw / 2)
                s[:, 6] = torch.sin(yaw / 2)
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.station, zeros, zeros, zeros)
        place(self.crank, zeros + c.pivot_x, zeros, zeros + c.pivot_z)

        q0 = c.q0_lo + torch.rand(m, device=dev) * (c.q0_hi - c.q0_lo)
        self.q0[env_ids] = q0
        place(self.drawer, q0, zeros, zeros)

        # cubes: independent x and |y| draws; random side split (torch.rand-based —
        # the first randint after a manual seed is degenerate across seeds)
        u = torch.rand(m, 7, device=dev)
        side = torch.where(u[:, 6] < 0.5, torch.ones(m, device=dev), -torch.ones(m, device=dev))
        zc = c.apron_z1 + c.cube_s / 2 + 0.002
        wx = c.cube_x[0] + u[:, 0] * (c.cube_x[1] - c.cube_x[0])
        wy = side * (c.cube_y[0] + u[:, 1] * (c.cube_y[1] - c.cube_y[0]))
        place(self.weight, wx, wy, zeros + zc, yaw=(u[:, 2] * 2 - 1) * math.pi)
        dx = c.cube_x[0] + u[:, 3] * (c.cube_x[1] - c.cube_x[0])
        dy = -side * (c.cube_y[0] + u[:, 4] * (c.cube_y[1] - c.cube_y[0]))
        place(self.decoy, dx, dy, zeros + zc, yaw=(u[:, 5] * 2 - 1) * math.pi)

        self._floaded[env_ids] = False
        self._fcrank[env_ids] = 0.0
        self._fdrawer[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "crank": self.crank.data.root_state_w[env_ids].clone(),
            "weight": self.weight.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "q0": self.q0[env_ids].clone(),
            "floaded": self._floaded[env_ids].clone(),
            "fcrank": self._fcrank[env_ids].clone(),
            "fdrawer": self._fdrawer[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.crank.write_root_state_to_sim(state["crank"], env_ids)
        self.weight.write_root_state_to_sim(state["weight"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.q0[env_ids] = state["q0"]
        self._floaded[env_ids] = state["floaded"]
        self._fcrank[env_ids] = state["fcrank"]
        self._fdrawer[env_ids] = state["fdrawer"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden KITCHEN CABINET stands on the ground with one drawer at chest "
            f"height. The drawer starts pulled OUT (the exact opening varies by "
            f"episode); its grey front plate has NO handle. In front of the cabinet "
            f"a dark steel GALLOWS FRAME (two outboard posts and a high crossbeam) "
            f"carries a blue BELL CRANK on a horizontal axle: a hanging PRESS ARM "
            f"ends in a red roller nose facing the drawer's front plate, and a "
            f"horizontal PAN ARM reaches out toward you, ending in an amber "
            f"RECEIVING POCKET (a deep open-topped well, "
            f"{(c.pkt_in_x1 - c.pkt_in_x0) * 100:.0f} x {2 * c.pkt_in_hy * 100:.0f} cm "
            f"inside). The crank is counter-biased: unloaded it rests with the pan "
            f"arm level and the press arm hanging clear of the drawer.\n"
            f"On the low apron bench below sit two {c.cube_s * 100:.0f} cm cubes: a "
            f"solid BRASS cube (gold, heavy, ~{c.weight_mass:.1f} kg) and a FOAM "
            f"cube (white, feather-light). Their bench positions and sides vary by "
            f"episode. Drop the BRASS cube into the receiving pocket: its weight "
            f"overpowers the crank's bias, the pan arm sinks, and the red roller "
            f"cam-presses the drawer shut along its slide. The foam cube is too "
            f"light to move the crank at all — putting it in the pocket does "
            f"nothing. Pushing the drawer shut by hand also fails the task: the "
            f"goal is the drawer SEATED AND HELD by the weighted press (pan arm "
            f"down on its stop with the brass cube riding it). Everything must "
            f"come to rest.\n"
            f"Goal: brass cube in the pan pocket, pan arm down on its stop, drawer "
            f"seated within {c.q_goal * 1000:.0f} mm — all at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Close the cabinet drawer using the counterweight press: pick up the "
            "heavy brass cube (not the white foam one) and drop it into the "
            "crank's receiving pocket so the press arm swings down and holds the "
            "drawer shut. Finish with everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the station frame (station fixed at the
        env origin with identity heading)."""
        return body.data.root_pos_w - self.env_origins

    def drawer_open(self) -> torch.Tensor:
        """(N,) drawer opening (its prismatic coordinate; 0 = seated/closed)."""
        return self._local(self.drawer)[:, 0]

    def crank_theta(self) -> torch.Tensor:
        """(N,) crank angle about +Y in rad (0 = up-stop rest, +down)."""
        q = self.crank.data.root_quat_w
        th = 2.0 * torch.atan2(q[:, 2], q[:, 0])
        return (th + math.pi) % (2.0 * math.pi) - math.pi

    def crank_on_pivot(self) -> torch.Tensor:
        """(N,) bool: crank root still at its pivot (guards teleported fakes)."""
        c = self.cfg
        tgt = self.env_origins.clone()
        tgt[:, 0] += c.pivot_x
        tgt[:, 2] += c.pivot_z
        return (self.crank.data.root_pos_w - tgt).norm(dim=-1) < 0.02

    def drawer_in_channel(self) -> torch.Tensor:
        """(N,) bool: drawer root on its slide (guards stolen-drawer fakes)."""
        p = self._local(self.drawer)
        return (p[:, 1].abs() < 0.03) & (p[:, 2].abs() < 0.03) \
            & (p[:, 0] > -0.02) & (p[:, 0] < self.cfg.stroke + 0.02)

    def _crank_local(self, body) -> torch.Tensor:
        """(N,3) body root position in the CRANK frame (rides the swing)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.crank.data.root_quat_w,
            body.data.root_pos_w - self.crank.data.root_pos_w)

    def weight_on_pan(self) -> torch.Tensor:
        """(N,) bool: the BRASS cube rides the pan assembly (crank-local box
        spanning the pan arm and pocket; excludes the ground, the apron, the
        hanging arm and anything off the crank)."""
        c = self.cfg
        p = self._crank_local(self.weight)
        return (p[:, 0] > c.load_x[0]) & (p[:, 0] < c.load_x[1]) \
            & (p[:, 1].abs() < c.load_hy) \
            & (p[:, 2] > c.load_z[0]) & (p[:, 2] < c.load_z[1]) \
            & self.crank_on_pivot()

    def settled(self) -> torch.Tensor:
        """(N,) bool: cubes and drawer slow; crank judged on ANGULAR velocity
        (its CoM sits near the pivot, so linear velocity is small even
        mid-swing — angular is the honest observable)."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.weight.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.crank.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.crank.data.root_pos_w,
                         self.weight.data.root_pos_w, self.decoy.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._floaded |= self.weight_on_pan() & fin
        th_max = math.radians(c.theta_max_deg)
        fc = (self.crank_theta() / th_max).clamp(0.0, 1.0)
        fc = torch.where(self.crank_on_pivot() & fin, fc, torch.zeros_like(fc))
        self._fcrank = torch.maximum(self._fcrank, fc)
        fd = ((self.q0 - self.drawer_open()) / self.q0.clamp(min=1e-4)).clamp(0.0, 1.0)
        fd = torch.where(self.drawer_in_channel() & fin, fd, torch.zeros_like(fd))
        self._fdrawer = torch.maximum(self._fdrawer, fd)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, judged LIVE on the settled physical state: the brass
        weight rides the pan assembly, the crank stands at its down-stop
        (>= crank_goal_deg, on its pivot), and the drawer is seated
        (<= q_goal, in its channel) — all settled and finite. The unloaded
        crank's bias makes 'crank down without the weight' physically
        unsustainable, and the load clause refuses it instantly in constructed
        states; the foam decoy can never overpower the bias."""
        c = self.cfg
        self._update_latches()
        return self.weight_on_pan() \
            & (self.crank_theta() >= math.radians(c.crank_goal_deg)) \
            & self.crank_on_pivot() \
            & (self.drawer_open() <= c.q_goal) \
            & self.drawer_in_channel() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 loaded (latched) + 0.25 x crank closure
        fraction (latched max) + 0.35 x drawer closure fraction (latched max),
        capped at 0.90; exactly 1.0 iff success() holds live. Null ~0. The
        seed's strategy (hand-push the drawer shut) earns ~0.35 and can never
        succeed: the pan stays empty and the crank stays up."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._floaded.float() + c.w_crank * self._fcrank
                + c.w_drawer * self._fdrawer).clamp(max=0.90)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="weight_crank_cabinet", robot="null"))
