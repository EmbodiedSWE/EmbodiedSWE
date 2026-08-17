"""GimbalServiceScene — build a two-bowl ball-and-socket gimbal on a slick tilted shelf
so the dessert cube can rest WORLD-LEVEL on an incline nothing rests on directly.

Derived from libero_90 kitchen_scene2 "stack the middle black bowl on the back black
bowl". The seed is ONE unconstrained grasp-carry-place: pick a black bowl, set it on
another black bowl on a flat table, success = an xy/z proximity relation between two
bowl origins. Here the two black bowls are still stacked one inside the other — but the
stack is a MACHINE, not a goal pose:

  * the goal surface is a SERVICE SHELF tilted 16-20 deg (re-randomized each episode,
    with fresh yaw and position) and coated slick (mu 0.02). Nothing survives on it
    bare: the cube slides off (pair mu 0.21 << tan 16 deg = 0.29) and so does the
    socket bowl itself (pair 0.235 << 0.29). The ONLY hold is a low octagonal RING
    curb bolted to the shelf: the socket bowl's lower wall drops inside it with ~6 mm
    play and the curb arrests the slide;
  * the SOCKET BOWL (black bowl A) is a two-tier open bowl with a POLISHED interior
    seat (base + lower wall slick, mu 0.04). Seated in the ring it is rigidly tilted
    WITH the shelf — anything set flat inside it rests face-flat ON the tilted floor,
    contained by the lower wall, so its tilt EQUALS the shelf pitch (>= 16 deg,
    verified by a smoke probe), which the rubric rejects (level gates are 8-10 deg,
    far below the 16 deg minimum pitch);
  * the GIMBAL BOWL (black bowl B) is a cup on a polished BALL: a 42 mm slick sphere
    bottom (mu 0.04) whose CoM is authored (and runtime-enforced) 34 mm BELOW the
    sphere centre. Nested into the socket bowl, the ball wedges against the polished
    seat; leveling is a rotation about the sphere centre that must SLIDE at the
    contacts, so the seat/ball pairing is deliberately slick (pair mu 0.04: friction
    stick angle ~3.6 deg, righting margin 2.2x at the 8 deg gate) and the deep
    ballast rights the cup — it self-levels under gravity at any sampled pitch. That
    passive articulation, not a pose, is what the task builds;
  * the golden dessert cube is served INTO the gimbal cup and must rest WORLD-level
    (within 10 deg) while sitting on a 16-20 deg incline — impossible without the
    gimbal, trivial with it.

Success (simultaneous, settled):
  * socket bowl seated in the ring curb (station-frame xy/z bands, face-aligned to
    the shelf within 8 deg);
  * gimbal bowl nested in the socket bowl (socket-frame xy/z bands) AND world-level
    within 8 deg — on a >=16 deg shelf only the ball articulation can satisfy both;
  * cube at rest inside the gimbal cup (cup-frame bands) and world-level within
    10 deg;
  * every dynamic body settled.

Rubric (graded 0..1, latched in post_step, additive; 1.0 iff success()):
  0.00  nothing happened
  +0.20 ringed — the socket bowl has rested seated in the ring curb
  +0.30 nested — gimbal bowl in the socket AND world-level, WHILE the socket is
        seated in the ring (a level gimbal parked anywhere else latches nothing)
  +0.30 served — cube at rest, level, inside the gimbal cup, WHILE nested and
        ringed both hold (a cube in a level cup on the bench latches nothing)
  1.00  success() (overrides the 0.80 partial sum)

Honesty of the gates:
  * `nested` cannot be satisfied by a rigid stack: a body fixed to the socket rides
    at the shelf pitch (>= 16 deg), and the level gate is 8 deg. Only the sphere
    articulation passes. Teleport-writing B level inside a tilted A is exactly the
    physical rest state, so there is nothing to fake — the physics produces it;
  * the DECLARED structure clause: the gimbal must sit IN the socket bowl and the
    socket bowl IN the ring. A gimbal dropped straight into the ring (physically
    stable, cube level) latches nothing — `nested` demands the socket-frame bands
    and `served` demands nested. This ordering clause is stated in describe();
  * every latch carries a `latch_speed` velocity gate so a body flying through a
    band latches nothing; latched credit survives later mishaps.

Assets are fully procedural (no external files):
  * station (kinematic, ONE body, pose re-randomized per episode — no joints, so
    unlike a jointed fixture it CAN move): support column + slick shelf slab
    360x300x30 mm + fixture-mu octagonal ring curb (inner apothem 67 mm, 16 mm tall)
    centred 50 mm downhill of the slab centre;
  * socket bowl A (free, black): 110 mm base disc + lower octagon wall (inner apothem
    48 mm), both POLISHED slick (the gimbal seat), flared upper wall at bowl mu
    (mouth apothem 62 mm, 8 mm rim — pinchable);
  * gimbal bowl B (free, black): 42 mm polished-slick sphere bottom + 10 mm cup floor
    + octagon cup wall at bowl mu (inner apothem 40 mm, 8 mm rim); mass 0.55 kg, CoM
    34 mm below the sphere centre, authored AND runtime-enforced via set_coms +
    readback;
  * cube: 30 mm golden dessert cube.

Per-episode randomization (verified by readback in the smoke): shelf pitch
U[16, 20] deg, shelf yaw U[0, 2pi), shelf centre +/-30 mm, socket-bowl park side +
jitter + yaw, gimbal-bowl park (opposite side) + jitter, cube park side + jitter +
yaw.

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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ EXPLICIT MassAPI mass, CoM and
# diagonal inertia on dynamics — custom spawners get no auto-computed mass properties, and
# the gimbal's below-centre CoM is the whole mechanism). Physics materials are authored and
# bound per child (custom spawn funcs apply no cfg schemas; PhysX pair friction is the MEAN
# of the two prims' mu, which the slick-shed / self-leveling margins below are computed from).
# Authored through `clone()` so per-env replication is idempotent.

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float | None, com=(0.0, 0.0, 0.0), inertia=None,
                ang_damp: float = 0.10, lin_damp: float = 0.05):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        mapi = UsdPhysics.MassAPI.Apply(root)
        mapi.CreateMassAttr(float(mass))
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
        if inertia is not None:
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
        px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        # Cap the contact-solver pop on landings; light damping so landed bodies cross
        # the settle gate promptly. Solver iterations raised: the sphere-in-socket
        # contact otherwise shows the GPU curved-collider creep artifact.
        px_rb.CreateMaxDepenetrationVelocityAttr(0.5)
        px_rb.CreateLinearDampingAttr(float(lin_damp))
        px_rb.CreateAngularDampingAttr(float(ang_damp))
        px_rb.CreateSolverPositionIterationCountAttr(8)
        px_rb.CreateSolverVelocityIterationCountAttr(4)
    return root


def _phys_material(prim_path: str, name: str, static: float, dynamic: float):
    """A physics material prim under the compound root (restitution 0 everywhere)."""
    import omni.usd
    from pxr import UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, f"{prim_path}/{name}")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _child_cyl(prim_path: str, name: str, *, radius: float, height: float,
               tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
               color: tuple, contact_offset: float | None):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return cyl


def _child_box(prim_path: str, name: str, *, tx: float, ty: float, tz: float,
               sx: float, sy: float, sz: float, yaw_deg: float = 0.0,
               color: tuple, contact_offset: float | None):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    if yaw_deg != 0.0:
        bxf.AddRotateZOp().Set(float(yaw_deg))
    bxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return box


def _child_sphere(prim_path: str, name: str, *, radius: float,
                  tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
                  color: tuple, contact_offset: float | None):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/{name}")
    sph.CreateRadiusAttr(radius)
    sph.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(sph.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return sph


def _octagon_wall(prim_path: str, tag: str, *, cx: float, cy: float, z_center: float,
                  r_mid: float, thick: float, height: float, yaw0_deg: float,
                  color: tuple, contact_offset: float):
    """8 box segments forming a closed octagon wall of mid-line radius `r_mid`.
    Each segment's local x axis is radial (so its size is (thick, seg_len, height))."""
    segs = []
    seg_len = 2.0 * r_mid * math.tan(math.pi / 8.0) + 0.004  # slight overlap: no slits
    for k in range(8):
        ang = yaw0_deg + k * 45.0
        rad = math.radians(ang)
        segs.append(_child_box(prim_path, f"{tag}{k}",
                               tx=cx + r_mid * math.cos(rad), ty=cy + r_mid * math.sin(rad),
                               tz=z_center, sx=thick, sy=seg_len, sz=height, yaw_deg=ang,
                               color=color, contact_offset=contact_offset))
    return segs


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The kinematic service station, ONE body: support column, slick shelf slab,
    fixture-mu octagonal ring curb. Root origin at the slab BOTTOM-face centre; local
    +x is the downhill direction once the root is pitched about local y."""
    root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=None)
    slick = _phys_material(prim_path, "slickmat", cfg.mu_slab[0], cfg.mu_slab[1])
    grip = _phys_material(prim_path, "gripmat", cfg.mu_fixture[0], cfg.mu_fixture[1])
    col = _child_box(prim_path, "column", tx=0.0, ty=0.0, tz=-cfg.col_h / 2,
                     sx=cfg.col_w, sy=cfg.col_w, sz=cfg.col_h,
                     color=cfg.wood, contact_offset=cfg.contact_offset)
    _bind_material(col, grip)
    slab = _child_box(prim_path, "slab", tx=0.0, ty=0.0, tz=cfg.slab_t / 2,
                      sx=cfg.slab_x, sy=cfg.slab_y, sz=cfg.slab_t,
                      color=cfg.ice, contact_offset=cfg.contact_offset)
    _bind_material(slab, slick)
    ring = _octagon_wall(prim_path, "ring", cx=cfg.ring_x, cy=0.0,
                         z_center=cfg.slab_t + cfg.ring_h / 2,
                         r_mid=cfg.ring_rmid, thick=cfg.ring_t, height=cfg.ring_h,
                         yaw0_deg=22.5, color=cfg.wood_dark,
                         contact_offset=cfg.contact_offset)
    for seg in ring:
        _bind_material(seg, grip)
    return root


def _spawn_socket_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Socket bowl A: base disc + narrow lower octagon wall (drops inside the ring
    curb) + flared upper wall (the mouth that admits the gimbal ball). Root origin at
    the base bottom centre. The base and the lower wall — the polished gimbal SEAT —
    are slick (leveling the wedged ball is a rotation about the sphere centre that
    must SLIDE at these contacts; grippy faces would freeze the cup off-level). The
    upper wall keeps bowl-mu. Slick base also sheds on the bare slab even harder."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, com=(0.0, 0.0, cfg.com_z),
                       inertia=cfg.inertia)
    mat = _phys_material(prim_path, "bowlmat", cfg.mu_bowl[0], cfg.mu_bowl[1])
    seat = _phys_material(prim_path, "seatmat", cfg.mu_seat[0], cfg.mu_seat[1])
    seat_kids = [_child_cyl(prim_path, "base", radius=cfg.base_r, height=cfg.base_t,
                            tz=cfg.base_t / 2, color=cfg.color,
                            contact_offset=cfg.contact_offset)]
    seat_kids += _octagon_wall(prim_path, "low", cx=0.0, cy=0.0,
                               z_center=cfg.base_t + cfg.low_h / 2,
                               r_mid=cfg.low_rmid, thick=cfg.low_t, height=cfg.low_h,
                               yaw0_deg=0.0, color=cfg.color,
                               contact_offset=cfg.contact_offset)
    kids = _octagon_wall(prim_path, "top", cx=0.0, cy=0.0,
                         z_center=cfg.base_t + cfg.low_h + cfg.top_h / 2,
                         r_mid=cfg.top_rmid, thick=cfg.top_t, height=cfg.top_h,
                         yaw0_deg=22.5, color=cfg.color,
                         contact_offset=cfg.contact_offset)
    for k in seat_kids:
        _bind_material(k, seat)
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_gimbal_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Gimbal bowl B: polished sphere bottom + cup floor disc + octagon cup wall.
    Root origin at the SPHERE CENTRE (so the authored below-origin CoM is below the
    pivot). The ball is slick — self-leveling slides at the ball contacts, so the
    ball/seat pair mu must stay tiny; the cup interior keeps bowl-mu for the cube."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, com=cfg.com,
                       inertia=cfg.inertia, ang_damp=cfg.ang_damp)
    mat = _phys_material(prim_path, "bowlmat", cfg.mu_bowl[0], cfg.mu_bowl[1])
    ballmat = _phys_material(prim_path, "ballmat", cfg.mu_ball[0], cfg.mu_ball[1])
    ball = _child_sphere(prim_path, "ball", radius=cfg.sphere_r, color=cfg.color,
                         contact_offset=cfg.contact_offset)
    kids = [_child_cyl(prim_path, "floor", radius=cfg.floor_r, height=cfg.floor_t,
                       tz=cfg.floor_z, color=cfg.color,
                       contact_offset=cfg.contact_offset)]
    kids += _octagon_wall(prim_path, "cup", cx=0.0, cy=0.0, z_center=cfg.wall_z,
                          r_mid=cfg.wall_rmid, thick=cfg.wall_t, height=cfg.wall_h,
                          yaw0_deg=0.0, color=cfg.color,
                          contact_offset=cfg.contact_offset)
    _bind_material(ball, ballmat)
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_cfgs() -> dict[str, Any]:
    """Lazily-built @configclass spawner cfg types (heavy imports deferred)."""
    if "station" not in _SPAWNER_CACHE:
        from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils import configclass

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            col_w: float = 0.12
            col_h: float = 0.16
            slab_x: float = 0.36
            slab_y: float = 0.30
            slab_t: float = 0.030
            ring_x: float = 0.05
            ring_rmid: float = 0.072
            ring_t: float = 0.010
            ring_h: float = 0.016
            mu_slab: tuple = (0.02, 0.02)
            mu_fixture: tuple = (0.5, 0.4)
            ice: tuple = (0.62, 0.70, 0.78)
            wood: tuple = (0.55, 0.42, 0.25)
            wood_dark: tuple = (0.38, 0.28, 0.16)
            contact_offset: float = 0.002

        @configclass
        class SocketBowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_socket_bowl)
            base_r: float = 0.055
            base_t: float = 0.010
            low_rmid: float = 0.052
            low_t: float = 0.008
            low_h: float = 0.028
            top_rmid: float = 0.066
            top_t: float = 0.008
            top_h: float = 0.020
            com_z: float = 0.023
            inertia: tuple = (0.0006, 0.0006, 0.0009)
            mu_bowl: tuple = (0.45, 0.40)
            mu_seat: tuple = (0.04, 0.03)  # polished interior seat: base + lower wall
            color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        @configclass
        class GimbalBowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gimbal_bowl)
            sphere_r: float = 0.042
            floor_r: float = 0.044
            floor_t: float = 0.010
            floor_z: float = 0.040
            wall_rmid: float = 0.044
            wall_t: float = 0.008
            wall_h: float = 0.040
            wall_z: float = 0.065
            com: tuple = (0.0, 0.0, -0.034)
            inertia: tuple = (0.0005, 0.0005, 0.0005)
            ang_damp: float = 1.50  # damps the ballast pendulum after each drop
            mu_bowl: tuple = (0.45, 0.40)
            mu_ball: tuple = (0.04, 0.03)  # polished ball: leveling slides here
            color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(station=StationSpawnerCfg, socket=SocketBowlSpawnerCfg,
                              gimbal=GimbalBowlSpawnerCfg)
    return _SPAWNER_CACHE


def _compound_spawner_cfg(kind: str, defaults: dict, *, mass: float, kinematic: bool) -> Any:
    import isaaclab.sim as sim_utils

    return _spawner_cfgs()[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        **defaults,
    )


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class GimbalServiceSceneCfg(BaseCfg):
    """Config for `GimbalServiceScene`. The margin budget behind the numbers (PhysX
    pair friction = mean of the two prims' static mu):
      * cube on bare slab:  (0.40+0.02)/2 = 0.210, tan(16 deg) = 0.287 -> slides off
        at every sampled pitch (1.37x margin);
      * socket bowl on bare slab: polished base -> (0.04+0.02)/2 = 0.03 << 0.287, it
        slides; only the ring curb (16 mm tall, catching the 28 mm lower wall)
        arrests it;
      * cube inside the TILTED socket: it rests FACE-FLAT on the tilted floor
        (contained by the 28 mm lower wall, which reaches above the cube's
        mid-height), so its tilt equals the shelf pitch >= 16 deg — failing the
        10 deg level gate by GEOMETRY, whatever the friction. The gimbal is
        necessary, not decorative;
      * gimbal self-levels: CoM 34 mm below the sphere pivot; nested, the ball wedges
        on the polished seat and leveling is a rotation about the sphere centre that
        SLIDES at the ball contacts. Ball/seat pair mu (0.04+0.04)/2 = 0.04 gives a
        friction stick angle ~3.6 deg — the righting torque at the 8 deg level gate
        exceeds the worst-case friction torque >= 2x (asserted below). Level gates
        (8/10 deg) sit >= 6 deg below the minimum pitch (16 deg), so a rigid ride can
        never pass them;
      * ring retention: sphere-vs-curb escape angle atan(0.033/0.034) = 44 deg and
        seated-stack tip angle atan(0.067/0.06) = 48 deg, both >> 20 deg."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_xy_tol: float = tunable(0.015)  # socket-bowl origin to ring centre, station frame (m)
    seat_z_lo: float = tunable(0.022)  # socket-bowl origin z band, station frame (seated = 0.030)
    seat_z_hi: float = tunable(0.042)
    seat_align_max_deg: float = tunable(8.0)  # socket-bowl axis vs shelf NORMAL (rides the shelf)
    nest_xy_tol: float = tunable(0.030)  # gimbal origin in socket frame (rest offset <= ~8 mm)
    nest_z_lo: float = tunable(0.042)  # gimbal origin z band, socket frame (rest = 0.052)
    nest_z_hi: float = tunable(0.068)
    level_max_deg: float = tunable(8.0)  # gimbal axis vs WORLD up — the gimbal proof
    cube_xy_tol: float = tunable(0.030)  # cube centre in gimbal-cup frame (inner apothem 0.040)
    cube_z_tol: float = tunable(0.009)  # |cube centre - cube_seat_z| in cup frame (rim perch ~ +0.040)
    cube_level_max_deg: float = tunable(10.0)  # cube face vs WORLD up
    settle_speed: float = tunable(0.08)  # max |v| of every dynamic body when judging (m/s)
    settle_ang: float = tunable(0.60)  # max |omega| (rad/s; GPU phantom band peaks ~0.47)
    settle_streak: int = tunable(30)  # settled = still for this many CONSECUTIVE steps (~0.25 s)
    latch_speed: float = tunable(0.10)  # milestones latch only while the moving body is this slow

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pitch_lo_deg: float = tunable(16.0)  # shelf pitch range, re-sampled per episode
    pitch_hi_deg: float = tunable(20.0)
    center_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the station centre
    park_jitter: float = tunable(0.020)  # uniform +/- xy jitter of the three parks
    mirror_a: bool = tunable(True)  # per-episode: socket bowl parks +y or -y (gimbal opposite)
    mirror_cube: bool = tunable(True)

    # --- tunable: placement (counter frame; intended arm base at (-0.42, 0, surface_z)) ---------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    station_center: tuple = tunable((0.10, 0.02))  # nominal station centre (jittered per episode)
    mount_h: float = tunable(0.075)  # slab bottom-face height above the counter
    a_park: tuple = tunable((-0.22, 0.26))  # socket-bowl park (y side mirrored; gimbal at -y side)
    cube_park: tuple = tunable((0.02, 0.40))  # cube park (y side mirrored)

    # --- info: station geometry -------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))
    slab_x: float = info(0.36)
    slab_y: float = info(0.30)
    slab_t: float = info(0.030)
    col_w: float = info(0.12)
    col_h: float = info(0.16)
    ring_x: float = info(0.05)  # ring-curb centre, station frame (downhill = +x of centre)
    ring_rmid: float = info(0.072)  # inner apothem 0.067: socket lower wall (circum 0.0606) drops in
    ring_t: float = info(0.010)
    ring_h: float = info(0.016)
    mu_slab: tuple = info((0.02, 0.02))  # slick shelf coat (static, dynamic)
    mu_fixture: tuple = info((0.5, 0.4))  # column + ring curb
    mu_bowl: tuple = info((0.45, 0.40))  # bowl bodies (A upper wall, B cup interior)
    mu_seat: tuple = info((0.04, 0.03))  # A's polished interior seat: base + lower wall
    mu_ball: tuple = info((0.04, 0.03))  # B's polished sphere bottom
    mu_cube: tuple = info((0.40, 0.35))
    # socket bowl A
    a_base_r: float = info(0.055)
    a_base_t: float = info(0.010)
    a_low_rmid: float = info(0.052)  # lower wall: inner apothem 0.048 (holds the 42 mm ball)
    a_low_t: float = info(0.008)
    a_low_h: float = info(0.028)
    a_top_rmid: float = info(0.066)  # upper wall: mouth apothem 0.062 (admits the ball, 10 mm play)
    a_top_t: float = info(0.008)
    a_top_h: float = info(0.020)
    a_mass: float = info(0.30)
    a_com_z: float = info(0.023)
    a_inertia: tuple = info((0.0006, 0.0006, 0.0009))
    # gimbal bowl B
    b_sphere_r: float = info(0.042)
    b_floor_r: float = info(0.044)
    b_floor_t: float = info(0.010)
    b_floor_z: float = info(0.040)  # cup floor disc centre (spans 0.035..0.045; top proud of ball crown)
    b_wall_rmid: float = info(0.044)  # cup wall: inner apothem 0.040
    b_wall_t: float = info(0.008)
    b_wall_h: float = info(0.040)
    b_wall_z: float = info(0.065)  # cup wall centre (spans 0.045..0.085)
    b_mass: float = info(0.55)
    b_com: tuple = info((0.0, 0.0, -0.034))  # 34 mm BELOW the sphere pivot: authored AND
    # runtime-enforced — the self-leveling the whole task hangs on
    b_inertia: tuple = info((0.0005, 0.0005, 0.0005))
    b_ang_damp: float = info(1.50)  # damps the ballast pendulum after each drop
    # cube
    cube_size: float = info(0.030)
    cube_mass: float = info(0.030)
    black: tuple = info((0.05, 0.05, 0.06))  # matte black, like the seed's akita bowls
    ice: tuple = info((0.62, 0.70, 0.78))
    gold: tuple = info((0.93, 0.74, 0.20))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    a_h: float = field(default=None, init=False)  # socket bowl total height
    a_mouth_apothem: float = field(default=None, init=False)
    a_low_apothem: float = field(default=None, init=False)
    a_low_circum: float = field(default=None, init=False)  # lower wall OUTER circumradius
    a_top_circum: float = field(default=None, init=False)
    ring_in_apothem: float = field(default=None, init=False)
    ring_out_circum: float = field(default=None, init=False)
    cup_in_apothem: float = field(default=None, init=False)
    b_cup_circum: float = field(default=None, init=False)  # cup wall OUTER circumradius
    b_rest_z_in_a: float = field(default=None, init=False)  # gimbal origin, socket frame, at rest
    cube_seat_z: float = field(default=None, init=False)  # cube centre, cup frame, at rest
    seat_z_local: float = field(default=None, init=False)  # socket origin, station frame, seated

    def __post_init__(self) -> None:
        coso = math.cos(math.pi / 8.0)
        self.a_h = round(self.a_base_t + self.a_low_h + self.a_top_h, 4)  # 0.058
        self.a_mouth_apothem = self.a_top_rmid - self.a_top_t / 2  # 0.062
        self.a_low_apothem = self.a_low_rmid - self.a_low_t / 2  # 0.048
        self.a_low_circum = (self.a_low_rmid + self.a_low_t / 2) / coso  # 0.0606
        self.a_top_circum = (self.a_top_rmid + self.a_top_t / 2) / coso  # 0.0758
        self.ring_in_apothem = self.ring_rmid - self.ring_t / 2  # 0.067
        self.ring_out_circum = (self.ring_rmid + self.ring_t / 2) / coso  # 0.0833
        self.cup_in_apothem = self.b_wall_rmid - self.b_wall_t / 2  # 0.040
        self.b_cup_circum = (self.b_wall_rmid + self.b_wall_t / 2) / coso  # 0.052
        self.b_rest_z_in_a = self.a_base_t + self.b_sphere_r  # 0.052: ball on the socket floor
        self.cube_seat_z = self.b_floor_z + self.b_floor_t / 2 + self.cube_size / 2  # 0.060
        self.seat_z_local = self.slab_t  # 0.030: socket base bottom on the slab top

        # ---- honesty asserts: the physics the rubric leans on must actually hold ----------------
        t_lo = math.tan(math.radians(self.pitch_lo_deg))
        t_hi = math.tan(math.radians(self.pitch_hi_deg))
        cube_slab = (self.mu_cube[0] + self.mu_slab[0]) / 2
        assert cube_slab * 1.25 < t_lo, \
            f"cube must slide off the bare slab at min pitch: pair mu {cube_slab:.3f} vs tan {t_lo:.3f}"
        a_slab = (self.mu_seat[0] + self.mu_slab[0]) / 2  # polished base on the slab
        assert a_slab * 1.15 < t_lo, \
            f"socket bowl must slide on the bare slab at min pitch: pair mu {a_slab:.3f} vs tan {t_lo:.3f}"
        # cube inside the TILTED socket rests face-flat on the tilted floor: the lower
        # wall must CONTAIN it (reach above its mid-height so it cannot climb out) —
        # then its tilt equals the shelf pitch and the level gate rejects by geometry
        assert self.a_base_t + self.a_low_h > self.a_base_t + self.cube_size / 2 + 0.010, \
            "socket lower wall must reach well above the contained cube's mid-height"
        assert self.a_low_apothem > self.cube_size * math.sqrt(2) / 2 + 0.003, \
            "socket lower cavity must admit the cube at any yaw (it rests ON the tilted floor)"
        # self-leveling friction budget: righting torque at the level gate must beat the
        # worst-case friction torque at the wedged ball contacts (rotation about the
        # sphere centre slides at BOTH contacts with lever = sphere radius) >= 2x
        pair_ball_seat = (self.mu_ball[0] + self.mu_seat[0]) / 2
        a_hi = math.radians(self.pitch_hi_deg)
        t_fric = pair_ball_seat * (math.cos(a_hi) + math.sin(a_hi)) * self.b_sphere_r
        t_right = -self.b_com[2] * math.sin(math.radians(self.level_max_deg))
        assert t_right > 2.0 * t_fric, \
            f"gimbal must self-level past the gate: righting {t_right:.5f} vs 2x friction {t_fric:.5f}"
        # ring admits the lower wall with play, and the curb overlaps it vertically
        assert self.ring_in_apothem > self.a_low_circum + 0.004, \
            "ring curb must admit the socket bowl's lower wall with play"
        assert self.slab_t + self.ring_h > self.slab_t + self.a_base_t + 0.004, \
            "ring curb must stand taller than the socket base disc to catch the lower wall"
        # gimbal ball passes the mouth, seats in the lower cavity, on the socket floor
        assert self.a_mouth_apothem > self.b_sphere_r + 0.008, "mouth must admit the ball with play"
        assert self.a_low_apothem > self.b_sphere_r + 0.004, "lower cavity must hold the ball with play"
        assert self.nest_z_lo < self.b_rest_z_in_a < self.nest_z_hi, "nest z band must bracket the ball rest"
        # cup admits the cube; cube seat inside the served band; floor top proud of the ball crown
        assert self.cup_in_apothem > self.cube_size * math.sqrt(2) / 2 + 0.003, \
            "cup must admit the cube at any yaw"
        assert self.b_floor_z + self.b_floor_t / 2 > self.b_sphere_r + 0.002, \
            "cup floor top must sit proud of the ball crown (cube rests on the disc, not the ball)"
        assert self.b_com[2] < -0.020, "gimbal CoM must sit well below the sphere pivot (righting torque)"
        # level gates must be UNREACHABLE by a rigid ride at any sampled pitch
        assert self.level_max_deg <= self.pitch_lo_deg - 6.0, \
            "gimbal level gate must sit >= 6 deg below the minimum shelf pitch"
        assert self.cube_level_max_deg <= self.pitch_lo_deg - 6.0, \
            "cube level gate must sit >= 6 deg below the minimum shelf pitch"
        # retention margins: seated stack cannot tip over the curb; ball cannot escape it
        tip = math.degrees(math.atan2(self.ring_in_apothem, 0.060))  # stack CoM <= 60 mm above slab
        assert tip > 2.0 * self.pitch_hi_deg, f"seated-stack tip angle {tip:.1f} must dwarf the pitch"
        off = math.sqrt(self.b_sphere_r**2 - (self.b_sphere_r - self.ring_h) ** 2)
        esc = math.degrees(math.atan2(off, -self.b_com[2]))
        assert esc > 2.0 * self.pitch_hi_deg, f"ball-vs-curb escape angle {esc:.1f} must dwarf the pitch"
        # the tilted slab's downhill edge must stay clear of the counter
        assert self.mount_h - (self.slab_x / 2) * math.sin(math.radians(self.pitch_hi_deg)) > 0.010, \
            "slab downhill edge must clear the counter at max pitch"
        # parks must clear the station footprint (slab circumradius, worst-case jittered centre)
        foot = math.hypot(self.slab_x / 2, self.slab_y / 2)  # 0.234
        jit = self.center_jitter * math.sqrt(2) + self.park_jitter * math.sqrt(2)
        for name, park, obj_r in (("a_park", self.a_park, self.a_top_circum),
                                  ("b_park", (self.a_park[0], -self.a_park[1]), self.b_cup_circum),
                                  ("cube_park", self.cube_park, self.cube_size * 0.71)):
            for sy in (1.0, -1.0):
                px, py = park[0], park[1] * sy
                d = math.hypot(px - self.station_center[0], py - self.station_center[1])
                assert d - jit > foot + obj_r + 0.010, \
                    f"{name} (side {sy:+.0f}) too close to the station footprint: {d:.3f}"


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("gimbal_service")
class GimbalServiceScene(BaseScene):
    cfg: GimbalServiceSceneCfg

    def __init__(self, cfg: GimbalServiceSceneCfg | None = None) -> None:
        super().__init__(cfg or GimbalServiceSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, service station, the two black bowls, the cube,
        at nominal poses (reset() re-poses everything and samples the episode)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        cx, cy = c.station_center

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            ),
        }

        station_defaults = dict(col_w=c.col_w, col_h=c.col_h, slab_x=c.slab_x,
                                slab_y=c.slab_y, slab_t=c.slab_t, ring_x=c.ring_x,
                                ring_rmid=c.ring_rmid, ring_t=c.ring_t, ring_h=c.ring_h,
                                mu_slab=c.mu_slab, mu_fixture=c.mu_fixture,
                                contact_offset=c.contact_offset)
        out["station"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Station",
            spawn=_compound_spawner_cfg("station", station_defaults, mass=1.0, kinematic=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, z0 + c.mount_h)),
        )

        out["bowl_a"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl_a",
            spawn=_compound_spawner_cfg(
                "socket",
                dict(base_r=c.a_base_r, base_t=c.a_base_t, low_rmid=c.a_low_rmid,
                     low_t=c.a_low_t, low_h=c.a_low_h, top_rmid=c.a_top_rmid,
                     top_t=c.a_top_t, top_h=c.a_top_h, com_z=c.a_com_z,
                     inertia=c.a_inertia, mu_bowl=c.mu_bowl, mu_seat=c.mu_seat,
                     color=c.black, contact_offset=c.contact_offset),
                mass=c.a_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.a_park[0], c.a_park[1], z0 + 0.001)),
        )

        out["bowl_b"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl_b",
            spawn=_compound_spawner_cfg(
                "gimbal",
                dict(sphere_r=c.b_sphere_r, floor_r=c.b_floor_r, floor_t=c.b_floor_t,
                     floor_z=c.b_floor_z, wall_rmid=c.b_wall_rmid, wall_t=c.b_wall_t,
                     wall_h=c.b_wall_h, wall_z=c.b_wall_z, com=c.b_com,
                     inertia=c.b_inertia, ang_damp=c.b_ang_damp, mu_bowl=c.mu_bowl,
                     mu_ball=c.mu_ball, color=c.black, contact_offset=c.contact_offset),
                mass=c.b_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.a_park[0], -c.a_park[1], z0 + c.b_sphere_r + 0.001)),
        )

        out["cube"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(c.cube_size, c.cube_size, c.cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.gold),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.mu_cube[0], dynamic_friction=c.mu_cube[1],
                    restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cube_park[0], c.cube_park[1], z0 + c.cube_size / 2 + 0.001)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, enforce the gimbal CoM, allocate layout readbacks + latches."""
        super().bind(env)
        c = self.cfg
        self.station: RigidObject = env.iscene["station"]
        self.bowl_a: RigidObject = env.iscene["bowl_a"]
        self.bowl_b: RigidObject = env.iscene["bowl_b"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._enforce_gimbal_mass_props()
        # station pose readback (world), refreshed by reset(); defaults = authored spawn
        self.station_pos = self.env_origins + torch.tensor(
            [c.station_center[0], c.station_center[1], c.surface_z + c.mount_h], device=dev)
        self.station_quat = torch.zeros(n, 4, device=dev)
        self.station_quat[:, 0] = 1.0
        self.station_pitch_deg = torch.zeros(n, device=dev)
        self.station_yaw = torch.zeros(n, device=dev)
        self.a_side = torch.ones(n, device=dev)
        self.cube_side = torch.ones(n, device=dev)
        # progress latches (post_step; cleared per reset)
        self.ever_ringed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_nested = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_served = torch.zeros(n, dtype=torch.bool, device=dev)
        # consecutive-steps-still counter behind settled() (post_step; cleared per reset)
        self.rest_streak = torch.zeros(n, dtype=torch.long, device=dev)

    def _enforce_gimbal_mass_props(self) -> None:
        """Write the gimbal bowl's below-pivot CoM into the PhysX view and VERIFY by
        readback. The self-leveling — the entire mechanism — lives on this CoM; a CoM
        silently left at the body origin makes the cup a neutral ball that rests at
        any angle, voiding the task. Enforced at runtime on top of the USD authoring
        and printed once for the log."""
        c = self.cfg
        view = self.bowl_b.root_physx_view
        n = self.env.num_envs
        coms = view.get_coms().clone()
        flat = coms.view(n, 7)
        flat[:, :3] = torch.tensor(c.b_com, dtype=flat.dtype)
        view.set_coms(coms, torch.arange(n))
        com_back = view.get_coms().view(n, 7)[0, :3].tolist()
        mass_back = float(view.get_masses().view(n, -1)[0, 0])
        print(f"[gimbal_service] gimbal readback: mass={mass_back:.3f} "
              f"(authored {c.b_mass}) com={[round(v, 4) for v in com_back]} "
              f"(authored {c.b_com})", flush=True)
        if abs(mass_back - c.b_mass) > 0.02 or abs(com_back[2] - c.b_com[2]) > 0.004:
            print("[gimbal_service] GIMBAL MASS/COM APPLY FAILED — self-leveling void",
                  flush=True)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: station re-posed (fresh pitch, yaw, centre), socket bowl and
        cube parked on random sides with jitter (gimbal bowl opposite the socket),
        latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z

        # burn one draw: the FIRST post-seed sample is near-constant across seeds
        torch.rand(m, device=dev)

        def u(amp: float, shape=(1,)) -> torch.Tensor:
            return (torch.rand(m, *shape, device=dev) * 2 - 1) * amp

        def side(enabled: bool) -> torch.Tensor:
            if not enabled:
                return torch.ones(m, device=dev)
            return torch.where(torch.rand(m, device=dev) > 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))

        def write(body: RigidObject, xy: torch.Tensor, z: float, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # station: fresh pitch, yaw, centre; quat = Rz(yaw) * Ry(pitch)
        pitch = (torch.rand(m, device=dev) * (c.pitch_hi_deg - c.pitch_lo_deg)
                 + c.pitch_lo_deg)
        yaw = torch.rand(m, device=dev) * (2 * math.pi)
        cxy = torch.tensor(c.station_center, device=dev).expand(m, 2) + u(c.center_jitter, (2,))
        h1, h2 = yaw / 2, torch.deg2rad(pitch) / 2
        c1, s1, c2, s2 = torch.cos(h1), torch.sin(h1), torch.cos(h2), torch.sin(h2)
        quat = torch.stack([c1 * c2, -s1 * s2, c1 * s2, s1 * c2], dim=1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cxy
        st[:, 2] = z0 + c.mount_h
        st[:, 3:7] = quat
        st[:, 0:3] += origin
        self.station.write_root_state_to_sim(st, env_ids)
        self.station_pos[env_ids] = st[:, 0:3]
        self.station_quat[env_ids] = quat
        self.station_pitch_deg[env_ids] = pitch
        self.station_yaw[env_ids] = yaw

        # socket bowl + gimbal bowl on OPPOSITE sides; cube on its own random side
        asd = side(c.mirror_a)
        self.a_side[env_ids] = asd
        axy = torch.stack([torch.full((m,), c.a_park[0], device=dev),
                           c.a_park[1] * asd], dim=1) + u(c.park_jitter, (2,))
        write(self.bowl_a, axy, z0 + 0.001, u(math.pi).squeeze(-1))

        bxy = torch.stack([torch.full((m,), c.a_park[0], device=dev),
                           -c.a_park[1] * asd], dim=1) + u(c.park_jitter, (2,))
        write(self.bowl_b, bxy, z0 + c.b_sphere_r + 0.001, u(math.pi).squeeze(-1))

        cs = side(c.mirror_cube)
        self.cube_side[env_ids] = cs
        cxy2 = torch.stack([torch.full((m,), c.cube_park[0], device=dev),
                            c.cube_park[1] * cs], dim=1) + u(c.park_jitter, (2,))
        write(self.cube, cxy2, z0 + c.cube_size / 2 + 0.001, u(math.pi).squeeze(-1))

        for latch in (self.ever_ringed, self.ever_nested, self.ever_served):
            latch[env_ids] = False
        self.rest_streak[env_ids] = 0

    # ----- frames + readings ----------------------------------------------------------------------
    def st_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> station frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station_quat, p_w - self.station_pos)

    def station_up_w(self) -> torch.Tensor:
        """(N,3) shelf normal in world."""
        from isaaclab.utils.math import quat_apply

        n = self.station_quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.station_quat.device).expand(n, 3)
        return quat_apply(self.station_quat, ez)

    def seat_center_w(self) -> torch.Tensor:
        """(N,3) world point where the seated socket bowl's origin belongs."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.station_quat.shape[0]
        loc = torch.tensor([c.ring_x, 0.0, c.seat_z_local],
                           device=self.station_quat.device).expand(n, 3)
        return self.station_pos + quat_apply(self.station_quat, loc)

    # ----- geometric predicates -----------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world, for tilt gates."""
        from isaaclab.utils.math import quat_apply

        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n, 3)
        return quat_apply(quat, ez)[:, 2]

    def _axis_up(self, quat: torch.Tensor, axis_w: torch.Tensor) -> torch.Tensor:
        """dot(body local +z in world, axis_w) — alignment to an arbitrary axis."""
        from isaaclab.utils.math import quat_apply

        n = quat.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n, 3)
        return (quat_apply(quat, ez) * axis_w).sum(dim=-1)

    def a_in_ring(self) -> torch.Tensor:
        """(N,) bool: socket bowl seated in the ring curb — station-frame xy within
        `seat_xy_tol` of the ring centre, origin z in the seated band, face aligned
        to the SHELF normal (it rides the shelf: world-up would be dishonest here)."""
        c = self.cfg
        loc = self.st_local(self.bowl_a.data.root_pos_w)
        tgt = torch.tensor([c.ring_x, 0.0], device=loc.device)
        in_xy = (loc[:, :2] - tgt).norm(dim=-1) < c.seat_xy_tol
        in_z = (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)
        aligned = self._axis_up(self.bowl_a.data.root_quat_w, self.station_up_w()) \
            .clamp(-1, 1) >= math.cos(math.radians(c.seat_align_max_deg))
        return in_xy & in_z & aligned

    def b_in_a(self) -> torch.Tensor:
        """(N,) bool: gimbal bowl nested in the socket bowl (socket-frame xy/z bands
        around the ball-on-floor rest) AND world-level within `level_max_deg` — on a
        >= 16 deg shelf only the ball articulation satisfies both at once."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q = self.bowl_a.data.root_quat_w
        d = self.bowl_b.data.root_pos_w - self.bowl_a.data.root_pos_w
        loc = quat_apply_inverse(q, d)
        in_xy = loc[:, :2].norm(dim=-1) < c.nest_xy_tol
        in_z = (loc[:, 2] > c.nest_z_lo) & (loc[:, 2] < c.nest_z_hi)
        level = self._up_z(self.bowl_b.data.root_quat_w).clamp(-1, 1) \
            >= math.cos(math.radians(c.level_max_deg))
        return in_xy & in_z & level

    def cube_in_b(self) -> torch.Tensor:
        """(N,) bool: cube at rest INSIDE the gimbal cup (cup frame; a rim perch is
        ~40 mm too high) and world-level within `cube_level_max_deg` (any face up)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        q = self.bowl_b.data.root_quat_w
        d = self.cube.data.root_pos_w - self.bowl_b.data.root_pos_w
        loc = quat_apply_inverse(q, d)
        in_xy = loc[:, :2].norm(dim=-1) < c.cube_xy_tol
        in_z = (loc[:, 2] - c.cube_seat_z).abs() < c.cube_z_tol
        cq = self.cube.data.root_quat_w
        n = cq.shape[0]
        best = torch.zeros(n, device=cq.device)
        for ax in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            v = torch.tensor(ax, device=cq.device).expand(n, 3)
            best = torch.maximum(best, quat_apply(cq, v)[:, 2].abs())
        level = best.clamp(max=1.0) >= math.cos(math.radians(c.cube_level_max_deg))
        return in_xy & in_z & level

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: every dynamic body slow RIGHT NOW (lin + ang)."""
        c = self.cfg
        still = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.bowl_a, self.bowl_b, self.cube):
            still &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            still &= b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return still

    def settled(self) -> torch.Tensor:
        """(N,) bool: still for `settle_streak` CONSECUTIVE steps (counter maintained
        in post_step) — a rocking gimbal is instantaneously slow at every turning
        point, so a single-instant read would call a moving scene settled."""
        return self.rest_streak >= self.cfg.settle_streak

    def success(self) -> torch.Tensor:
        """(N,) bool: socket seated in the ring, gimbal nested and level, cube served
        level in the cup, everything at rest — a HELD articulated state."""
        return self.a_in_ring() & self.b_in_a() & self.cube_in_b() & self.settled()

    # ----- progress latching ----------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch milestones at sim rate, velocity-gated and ORDER-gated: `nested`
        demands the socket seated in the ring at the same instant (a level gimbal
        parked on the bench or dropped straight into the ring latches nothing) and
        `served` demands nested AND ringed (a cube in a level cup anywhere else
        latches nothing)."""
        c = self.cfg
        now = self._still_now()
        self.rest_streak = torch.where(now, self.rest_streak + 1,
                                       torch.zeros_like(self.rest_streak))
        ring_now = self.a_in_ring()
        a_slow = self.bowl_a.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_ringed |= ring_now & a_slow
        nest_now = self.b_in_a()
        b_slow = self.bowl_b.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_nested |= nest_now & ring_now & b_slow
        cube_slow = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_served |= self.cube_in_b() & nest_now & ring_now & cube_slow

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along a correct
        run): +0.20 ringed, +0.30 nested, +0.30 served; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.20 * self.ever_ringed.float()
        s = s + 0.30 * self.ever_nested.float()
        s = s + 0.30 * self.ever_served.float()
        return torch.where(self.success(), torch.ones_like(s), s)

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"station": self.station, "bowl_a": self.bowl_a,
                  "bowl_b": self.bowl_b, "cube": self.cube}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "station_pose": torch.cat([self.station_pos[env_ids],
                                       self.station_quat[env_ids],
                                       self.station_pitch_deg[env_ids, None],
                                       self.station_yaw[env_ids, None]], dim=1),
            "sides": torch.stack([self.a_side[env_ids], self.cube_side[env_ids]], dim=1),
            "latches": torch.stack([self.ever_ringed[env_ids], self.ever_nested[env_ids],
                                    self.ever_served[env_ids]], dim=1),
            "rest_streak": self.rest_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"station": self.station, "bowl_a": self.bowl_a,
                  "bowl_b": self.bowl_b, "cube": self.cube}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        sp = state["station_pose"]
        self.station_pos[env_ids] = sp[:, 0:3]
        self.station_quat[env_ids] = sp[:, 3:7]
        self.station_pitch_deg[env_ids] = sp[:, 7]
        self.station_yaw[env_ids] = sp[:, 8]
        sides = state["sides"]
        self.a_side[env_ids] = sides[:, 0]
        self.cube_side[env_ids] = sides[:, 1]
        lat = state["latches"]
        self.ever_ringed[env_ids] = lat[:, 0]
        self.ever_nested[env_ids] = lat[:, 1]
        self.ever_served[env_ids] = lat[:, 2]
        self.rest_streak[env_ids] = state["rest_streak"]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wide gray kitchen counter. Rising from it on a wooden column is a SERVICE "
            f"SHELF: a pale ice-slick slab ({c.slab_x * 1000:.0f} x {c.slab_y * 1000:.0f} mm) "
            f"tilted {c.pitch_lo_deg:.0f}-{c.pitch_hi_deg:.0f} deg (the pitch, the facing "
            f"direction and the exact spot change every episode). The coating is so slick "
            f"that NOTHING rests on it bare — a cube or a bowl set on the open slab slides "
            f"straight off the downhill edge. The only purchase is a low dark RING CURB "
            f"({c.ring_h * 1000:.0f} mm tall, inner span ~{2 * c.ring_in_apothem * 1000:.0f} mm) "
            f"fixed to the slab, downhill of its centre.\n"
            f"On the counter wait two matte-BLACK bowls and a small GOLDEN dessert cube "
            f"({c.cube_size * 1000:.0f} mm), parked on sides that change every episode. The "
            f"SOCKET bowl ({2 * c.a_top_circum * 1000:.0f} mm across, {c.a_h * 1000:.0f} mm "
            f"tall) has a narrow lower wall that drops inside the ring curb and a flared "
            f"mouth (~{2 * c.a_mouth_apothem * 1000:.0f} mm). The GIMBAL bowl is a cup "
            f"riding on a {2 * c.b_sphere_r * 1000:.0f} mm BALL bottom, heavily ballasted "
            f"below the ball's centre: nested in the socket, the ball contact lets the cup "
            f"swivel and its low ballast rights it, so the cup finds WORLD-LEVEL on its own "
            f"no matter how the shelf tilts.\n"
            f"Goal, in this structure: seat the SOCKET bowl in the ring curb (it must sit "
            f"flat against the slab, held by the curb); NEST the gimbal bowl into the "
            f"socket so its cup self-levels; then serve the golden cube INTO the gimbal cup "
            f"so it rests level (within {c.cube_level_max_deg:.0f} deg of world-horizontal) "
            f"on the tilted shelf. The full stack is required: a gimbal set straight into "
            f"the ring, or a level stack built anywhere off the shelf, does not count — the "
            f"cup must sit IN the socket bowl and the socket bowl IN the ring. A cube laid "
            f"straight into the socket bowl just rests at the shelf angle and fails the "
            f"level test; a cube on the bare slab slides away. Everything must come to rest."
        )

    def instruction(self) -> str:
        return (
            "Seat the wide black socket bowl inside the ring curb on the tilted slick "
            "shelf, nest the ball-bottomed black gimbal bowl into it so the cup "
            "self-levels, then place the golden dessert cube into the gimbal cup so it "
            "rests level while the whole stack stays seated on the shelf."
        )


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="gimbal_service", robot="null"))
