"""VaultLauncherScene — deliver the steel ball INTO the sealed delivery vault by
machine, not by hand: swivel the yaw TURRET until its gravity chute points at the
vault's snout mouth, drop the ball into the chute's high end, and let it roll,
launch off the lip, fly the gap, thread the mouth, and climb the one-way ridge
into the vault's interior — where a 72 mm drop keeps it forever.

Derived from maniskill/pick_single_egad ("pick the loose object, lift it 7.5 cm":
one grasp plus a straight-up position-shift check, judged by a z displacement).
Here the seed's entire plan is insufficient by construction, not re-parameterized:

  1. the goal is not a lift — the ball must END UP inside a fully SEALED vault
     (floor, walls, roof; the only opening is a low snout tunnel facing the
     launcher). Picking the ball up and putting it down anywhere reachable scores
     nothing; the smoke battery lifts it 10 cm and sets it down to prove it.
  2. the hand cannot deliver it: the snout tunnel is 124 mm deep but a Franka
     finger (~55 mm) holding the 32 mm ball reaches barely ~90 mm in, short of
     the ridge — and the tunnel floor is a rising incline, so anything RELEASED
     inside the snout rolls straight back out (the smoke battery does exactly
     that). Entry demands ~1 m/s of arrival speed through a 90 mm window.
  3. the only tool that produces that arrival is the LAUNCHER: a chute on a yaw
     turret. The task is therefore an AIMING task — read where the vault stands
     this episode (its bearing is randomized over a 100 deg fan and the turret
     always starts misaimed by 18-55 deg), swivel the turret by its rear handle
     until the chute points at the mouth (~5 deg of physical tolerance), then
     load the ball high in the chute. Gravity does the throw: the ball exits the
     lip at ~1.4 m/s, flies the 109 mm gap, drops ~30 mm, threads the mouth
     between apron and ceiling, and carries the 12 mm uphill ridge with 3x the
     required energy. A gentle or misaimed launch fails recoverably (the ball
     ends on the open floor and can be fetched and re-loaded).

Execution order is forced physically: aiming after loading is too late (the ball
is already gone — it leaves the chute in ~0.4 s), so aim-then-load is the only
sequence that works; the smoke battery launches at a misaimed turret to show the
aim is load-bearing.

Assets are fully procedural (compound spawners; memory: custom spawners apply no
cfg schemas, so mass/damping/collision are authored in the funcs):
  - base: heavy DYNAMIC pedestal (root MassAPI 40 kg, CoM at ground). Dynamic,
    not kinematic: a joint anchored to a kinematic body0 stays world-fixed when
    the body is teleported at reset.
  - turret: DYNAMIC, body origin ON the vertical axle (pure-quat yaw writes),
    carrying the inclined chute (40 mm channel, 35 mm rails, backstop), the flat
    launch run and lip, and the rear handle post with a graspable ball knob at
    185 mm lever radius; spawn-authored RevoluteJoint to the base (axis Z,
    limits +/-87 deg — clear of the PhysX 180 wrap; joint pair filtered).
  - vault: KINEMATIC sealed box — interior 140x160 mm under a full roof, snout
    tunnel with rising floor (one-way ridge), flare wings and a low apron at the
    mouth; solid plinths seal every under-floor cavity.
  - tray: KINEMATIC shallow dish where the ball starts, off behind the launcher.
  - ball: DYNAMIC 32 mm steel-ish sphere (density 2700, ~46 g), solver velocity
    iterations 4 (GPU sphere-creep fix).

Per-episode randomization (readback-verifiable): vault bearing +/-50 deg at
0.51 m; turret initial yaw = bearing -/+ 18..55 deg (always misaimed toward the
center); base xy jitter; tray bearing 180 +/- 40 deg at 0.35..0.50 m plus ball
jitter in the tray.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.20 * aimed   — turret ever settled within 4 deg of the vault bearing
                   (3-step persistence with the turret slow)
  0.20 * loaded  — ball ever riding the chute floor between the rails (3-step
                   persistence; a hover write above the chute does not count)
  0.30 * entered — ball ever inside the snout tunnel / interior airspace
  1.0 iff success() — ball inside the vault interior (vault-local box under the
                   ridge height), settled, finite. Non-success capped at 0.70;
                   null policy ~0 (the ball rests in its tray and the turret
                   starts >= 18 deg off aim).

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


# ----- USD authoring helpers --------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             orient=None, density: float | None = None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable | None,
             axis: str = "Z", density: float | None = None):
    """One cylinder child along `axis`."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    h2 = height / 2
    ext = {"Z": (Gf.Vec3f(-radius, -radius, -h2), Gf.Vec3f(radius, radius, h2)),
           "Y": (Gf.Vec3f(-radius, -h2, -radius), Gf.Vec3f(radius, h2, radius)),
           "X": (Gf.Vec3f(-h2, -radius, -radius), Gf.Vec3f(h2, radius, radius))}[axis]
    cyl.CreateExtentAttr(list(ext))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


def _add_sphere(stage, path: str, *, center, radius, color, collide: Callable,
                density: float | None = None):
    """One sphere child."""
    from pxr import Gf, UsdGeom, UsdPhysics

    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(float(radius))
    sph.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    xf = UsdGeom.Xformable(sph.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(sph.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(sph.GetPrim()).CreateDensityAttr(float(density))
    return sph.GetPrim()


def _qy_t(angle: float) -> tuple:
    """wxyz quat for a rotation of `angle` rad about +y."""
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def _qz_t(angle: float) -> tuple:
    """wxyz quat for a rotation of `angle` rad about +z."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The launcher base: heavy DYNAMIC plinth + pedestal column. Root MassAPI 40 kg
    with CoM at the body origin (ground level) so it stands like furniture. The
    turret's revolute joint anchors here (dynamic body0: survives reset teleports)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.base_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/plinth", center=(0.0, 0.0, c.plinth_h / 2),
             size=(c.plinth_xy, c.plinth_xy, c.plinth_h), color=c.base_color,
             collide=collide)
    _add_cyl(stage, f"{prim_path}/pedestal",
             center=(0.0, 0.0, (c.plinth_h + c.pedestal_top) / 2),
             radius=c.pedestal_r, height=c.pedestal_top - c.plinth_h,
             color=c.base_color, collide=collide)
    return root


def _spawn_turret(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The launcher turret: DYNAMIC, body origin at ground level ON the vertical
    axle. Local +x = launch direction. Children: support pylon, inclined chute
    (floor + rails + backstop), flat launch run + rails (lip at chan_lip_x), rear
    handle arm + post + knob. Spawn-authored REVOLUTE joint to the sibling Base
    (axis Z through the origin, limits +/-yaw_lim_deg; joint pair filtered).
    Masses via per-child density so the yaw inertia is real."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.turret_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    den = c.turret_density
    col = c.turret_color

    x0, z0 = c.chan_x0, c.chan_z0          # incline top (floor-top point)
    x1, z1 = c.chan_x1, c.chan_z1          # incline bottom = flat-run start
    alpha = math.atan2(z0 - z1, x1 - x0)   # incline angle below horizontal
    ln = math.hypot(x1 - x0, z0 - z1) + 0.004
    sa, ca = math.sin(alpha), math.cos(alpha)
    t = c.chan_floor_t
    w_out = 2 * (c.chan_half_w + c.rail_t)  # floor width spans under both rails
    mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
    q_inc = _qy_t(alpha)                   # +y rotation tips +x down by alpha
    # inclined chute floor (top surface passes through (x0,z0)-(x1,z1))
    _add_box(stage, f"{prim_path}/inc_floor",
             center=(mx - (t / 2) * sa, 0.0, mz - (t / 2) * ca),
             size=(ln, w_out, t), color=col, collide=collide, orient=q_inc,
             density=den)
    # inclined rails (inner faces at +/-chan_half_w, rail_h above the floor top)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/inc_rail_{'p' if sgn > 0 else 'n'}",
                 center=(mx + (c.rail_h / 2) * sa,
                         sgn * (c.chan_half_w + c.rail_t / 2),
                         mz + (c.rail_h / 2) * ca),
                 size=(ln, c.rail_t, c.rail_h), color=col, collide=collide,
                 orient=q_inc, density=den)
    # flat launch run: floor top at z1 out to the lip
    fx = (x1 + c.chan_lip_x) / 2
    fl = c.chan_lip_x - x1 + 0.002
    _add_box(stage, f"{prim_path}/flat_floor", center=(fx, 0.0, z1 - t / 2),
             size=(fl, w_out, t), color=col, collide=collide, density=den)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/flat_rail_{'p' if sgn > 0 else 'n'}",
                 center=(fx, sgn * (c.chan_half_w + c.rail_t / 2), z1 + c.rail_h / 2),
                 size=(fl, c.rail_t, c.rail_h), color=col, collide=collide,
                 density=den)
    # backstop sealing the chute's high end
    _add_box(stage, f"{prim_path}/backstop", center=(x0 - 0.009, 0.0, z0 + 0.020),
             size=(0.010, w_out, 0.062), color=col, collide=collide, density=den)
    # support pylon down toward the axle (clear of the ground and the base pedestal)
    _add_cyl(stage, f"{prim_path}/pylon", center=(0.0, 0.0, 0.155),
             radius=0.030, height=0.100, color=col, collide=collide, density=den)
    # rear handle: arm + post + red knob (the aiming grip, lever radius knob_x)
    _add_box(stage, f"{prim_path}/harm", center=(-0.170, 0.0, 0.315),
             size=(0.075, 0.024, 0.020), color=col, collide=collide, density=den)
    _add_cyl(stage, f"{prim_path}/hpost", center=(c.knob_x, 0.0, 0.345),
             radius=0.011, height=0.045, color=col, collide=collide, density=den)
    _add_sphere(stage, f"{prim_path}/knob", center=(c.knob_x, 0.0, c.knob_z),
                radius=c.knob_r, color=c.knob_color, collide=collide, density=den)

    # revolute joint to the sibling base, axis Z through the shared origin
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/axle")
    j.CreateBody0Rel().SetTargets([f"{base}/Base"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.yaw_lim_deg))
    j.CreateUpperLimitAttr(float(c.yaw_lim_deg))
    return root


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The delivery vault: KINEMATIC sealed box. Local frame: origin at the
    interior floor center at GROUND level, +x = entry direction (mouth on the -x
    side facing the launcher). Interior floor/walls/roof, snout tunnel with a
    RISING floor (the one-way ridge), flare wings and a low apron at the mouth.
    Solid plinths under tunnel and apron seal every under-floor cavity. No joints
    attach here, so kinematic is safe."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(c.contact_offset)
    col, colw = c.vault_color, c.vault_trim_color
    wt = c.wall_t
    ix, iy = c.int_half_x, c.int_half_y        # interior inner half extents
    wz = c.wall_z                              # wall top
    # interior floor (solid to the ground)
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.int_floor_z / 2),
             size=(2 * ix + 2 * wt, 2 * iy + 2 * wt, c.int_floor_z), color=col,
             collide=collide)
    # back wall, side walls
    _add_box(stage, f"{prim_path}/wall_back", center=(ix + wt / 2, 0.0, wz / 2),
             size=(wt, 2 * iy + 2 * wt, wz), color=col, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_{'yp' if sgn > 0 else 'yn'}",
                 center=(0.0, sgn * (iy + wt / 2), wz / 2),
                 size=(2 * ix + 2 * wt, wt, wz), color=col, collide=collide)
    # front wall: side segments beside the tunnel opening + header above it
    seg = iy + wt - c.tun_half_w
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/wall_f{'p' if sgn > 0 else 'n'}",
                 center=(-ix - wt / 2, sgn * (c.tun_half_w + seg / 2), wz / 2),
                 size=(wt, seg, wz), color=col, collide=collide)
    _add_box(stage, f"{prim_path}/wall_fhead",
             center=(-ix - wt / 2, 0.0, (c.tun_ceil_z + wz) / 2),
             size=(wt, 2 * c.tun_half_w, wz - c.tun_ceil_z), color=col,
             collide=collide)
    # full roof
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, wz + 0.006),
             size=(2 * ix + 2 * wt, 2 * iy + 2 * wt, 0.012), color=colw,
             collide=collide)
    # ---- snout tunnel: x from mouth_x to ridge_x -----------------------------------------------
    tx0, tx1 = c.mouth_x, c.ridge_x
    tln = tx1 - tx0
    tmx = (tx0 + tx1) / 2
    # solid plinth under the tunnel floor
    _add_box(stage, f"{prim_path}/tun_plinth",
             center=(tmx, 0.0, (c.tun_floor_z0 - c.chan_floor_t) / 2),
             size=(tln, 2 * c.tun_half_w + 2 * wt, c.tun_floor_z0 - c.chan_floor_t),
             color=col, collide=collide)
    # rising tunnel floor (top from tun_floor_z0 at the mouth to tun_floor_z1 at the ridge)
    beta = math.atan2(c.tun_floor_z1 - c.tun_floor_z0, tln)
    sb, cb = math.sin(beta), math.cos(beta)
    t = c.chan_floor_t
    _add_box(stage, f"{prim_path}/tun_floor",
             center=(tmx + (t / 2) * sb, 0.0,
                     (c.tun_floor_z0 + c.tun_floor_z1) / 2 - (t / 2) * cb),
             size=(math.hypot(tln, c.tun_floor_z1 - c.tun_floor_z0) + 0.004,
                   2 * c.tun_half_w + 2 * wt, t),
             color=colw, collide=collide, orient=_qy_t(-beta))
    # tunnel side walls, down to the ground (seal the under-snout flanks)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/tun_{'yp' if sgn > 0 else 'yn'}",
                 center=(tmx, sgn * (c.tun_half_w + wt / 2), c.tun_ceil_z / 2),
                 size=(tln, wt, c.tun_ceil_z), color=col, collide=collide)
    # tunnel ceiling
    _add_box(stage, f"{prim_path}/tun_ceil", center=(tmx, 0.0, c.tun_ceil_z + 0.006),
             size=(tln, 2 * c.tun_half_w + 2 * wt, 0.012), color=col,
             collide=collide)
    # ---- mouth furniture: flare wings + apron --------------------------------------------------
    fl = c.flare_len
    fang = math.radians(c.flare_deg)
    wh = c.tun_ceil_z - c.tun_floor_z0 + 0.012
    for sgn in (1.0, -1.0):
        cxy = (tx0 - (fl / 2) * math.cos(fang),
               sgn * (c.tun_half_w + wt / 2 + (fl / 2) * math.sin(fang)))
        _add_box(stage, f"{prim_path}/flare_{'p' if sgn > 0 else 'n'}",
                 center=(cxy[0], cxy[1], (c.tun_floor_z0 + c.tun_ceil_z) / 2),
                 size=(fl, wt, wh), color=col, collide=collide,
                 orient=_qz_t(-sgn * fang))
    ax0, az0 = c.apron_x0, c.apron_z0          # apron front (low) edge, floor top
    gamma = math.atan2(c.tun_floor_z0 - az0, tx0 - ax0)
    sg, cg = math.sin(gamma), math.cos(gamma)
    amx = (ax0 + tx0) / 2
    _add_box(stage, f"{prim_path}/apron",
             center=(amx + 0.005 * sg, 0.0, (az0 + c.tun_floor_z0) / 2 - 0.005 * cg),
             size=(math.hypot(tx0 - ax0, c.tun_floor_z0 - az0) + 0.004,
                   c.apron_w, 0.010),
             color=colw, collide=collide, orient=_qy_t(-gamma))
    _add_box(stage, f"{prim_path}/apron_plinth",
             center=(amx, 0.0, (az0 - 0.004) / 2),
             size=(tx0 - ax0, c.apron_w, az0 - 0.004), color=col, collide=collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The ball's start tray: KINEMATIC shallow dish (base plate + four low walls)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(c.contact_offset)
    col = c.tray_color
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.006),
             size=(0.12, 0.12, 0.012), color=col, collide=collide)
    for name, ctr, size in (
        ("w_xp", (0.055, 0.0, 0.022), (0.010, 0.12, 0.020)),
        ("w_xn", (-0.055, 0.0, 0.022), (0.010, 0.12, 0.020)),
        ("w_yp", (0.0, 0.055, 0.022), (0.12, 0.010, 0.020)),
        ("w_yn", (0.0, -0.055, 0.022), (0.12, 0.010, 0.020)),
    ):
        _add_box(stage, f"{prim_path}/{name}", center=ctr, size=size, color=col,
                 collide=collide)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cargo ball: DYNAMIC 32 mm sphere, density-massed (~46 g), velocity
    iterations 4 (GPU sphere-creep fix), near-zero damping (it must fly)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.01)
    pxrb.CreateAngularDampingAttr(0.01)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    _add_sphere(stage, f"{prim_path}/ball", center=(0.0, 0.0, 0.0),
                radius=c.ball_r, color=c.ball_color, collide=collide,
                density=c.ball_density)
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
            base_mass: float = 40.0
            plinth_xy: float = 0.34
            plinth_h: float = 0.06
            pedestal_r: float = 0.05
            pedestal_top: float = 0.10
            base_color: tuple = (0.24, 0.24, 0.27)
            contact_offset: float = 0.0015

        @configclass
        class TurretSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_turret)
            chan_x0: float = -0.13
            chan_z0: float = 0.280
            chan_x1: float = 0.15
            chan_z1: float = 0.120
            chan_lip_x: float = 0.211
            chan_half_w: float = 0.020
            chan_floor_t: float = 0.012
            rail_h: float = 0.035
            rail_t: float = 0.008
            knob_x: float = -0.185
            knob_z: float = 0.385
            knob_r: float = 0.020
            yaw_lim_deg: float = 87.0
            turret_ang_damping: float = 3.0
            turret_density: float = 2000.0
            turret_color: tuple = (0.72, 0.55, 0.20)
            knob_color: tuple = (0.85, 0.12, 0.12)
            contact_offset: float = 0.0015

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            int_half_x: float = 0.070
            int_half_y: float = 0.080
            int_floor_z: float = 0.010
            wall_z: float = 0.20
            wall_t: float = 0.008
            mouth_x: float = -0.190
            ridge_x: float = -0.066
            tun_half_w: float = 0.045
            tun_floor_z0: float = 0.070
            tun_floor_z1: float = 0.082
            tun_ceil_z: float = 0.175
            chan_floor_t: float = 0.012
            flare_len: float = 0.034
            flare_deg: float = 25.0
            apron_x0: float = -0.245
            apron_z0: float = 0.052
            apron_w: float = 0.130
            vault_color: tuple = (0.30, 0.38, 0.55)
            vault_trim_color: tuple = (0.55, 0.62, 0.78)
            contact_offset: float = 0.0015

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_color: tuple = (0.55, 0.45, 0.30)
            contact_offset: float = 0.0015

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            ball_r: float = 0.016
            ball_density: float = 2700.0
            ball_color: tuple = (0.85, 0.85, 0.90)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["base"] = BaseSpawnerCfg
        _SPAWNER_CACHE["turret"] = TurretSpawnerCfg
        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
        _SPAWNER_CACHE["ball"] = BallSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class VaultLauncherSceneCfg(BaseCfg):
    """Config for `VaultLauncherScene`. The launch ballistics, the one-way ridge,
    the hand exclusion and every clearance are asserted numerically in
    __post_init__ (import-time geometry audit: the gravity launch must arrive
    fast enough, low enough, and with real margins BEFORE any sim runs)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    aim_tol_deg: float = tunable(4.0)        # |turret yaw - vault bearing| to latch aimed
    aim_settle_avel: float = tunable(0.3)    # max turret |ang vel| when latching aimed (rad/s)
    load_z_tol: float = tunable(0.020)       # ball height band above the chute floor (m)
    load_y_tol: float = tunable(0.012)       # ball |y| in the chute when latching loaded (m)
    settle_speed: float = tunable(0.10)      # max ball speed when judging success (m/s)
    succ_half_x: float = tunable(0.066)      # success box (vault local): |x| below this
    succ_half_y: float = tunable(0.072)      # success box: |y| below this
    succ_max_z: float = tunable(0.055)       # success box: ball center below this (< ridge top)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    vault_bear_deg: float = tunable(50.0)    # vault bearing fan about the base +x (+/- deg)
    aim_off_range: tuple = tunable((18.0, 55.0))  # initial turret misaim off the bearing (deg)
    base_jitter: float = tunable(0.04)       # base xy jitter (+/- m)
    tray_bear_deg: float = tunable(40.0)     # tray bearing about base -x (+/- deg)
    tray_r_range: tuple = tunable((0.35, 0.50))   # tray distance from the base (m)
    ball_jit: float = tunable(0.020)         # ball xy jitter inside the tray (+/- m)

    # --- info: launcher geometry (turret local; origin on the axle at the ground) ---------------
    base_pos: tuple = info((0.0, 0.0))
    base_mass: float = info(40.0)
    plinth_xy: float = info(0.34)
    plinth_h: float = info(0.06)
    pedestal_r: float = info(0.05)
    pedestal_top: float = info(0.10)
    chan_x0: float = info(-0.13)             # chute floor-top: high end (x, z)
    chan_z0: float = info(0.280)
    chan_x1: float = info(0.15)              # chute floor-top: low end = flat-run start
    chan_z1: float = info(0.120)
    chan_lip_x: float = info(0.211)          # launch lip (end of the flat run)
    chan_half_w: float = info(0.020)         # chute inner half width (40 mm channel)
    chan_floor_t: float = info(0.012)
    rail_h: float = info(0.035)
    rail_t: float = info(0.008)
    knob_x: float = info(-0.185)             # handle knob lever radius (turret local -x)
    knob_z: float = info(0.385)
    knob_r: float = info(0.020)
    yaw_lim_deg: float = info(87.0)          # joint limits +/- (174 deg travel < 180 wrap)
    turret_ang_damping: float = info(3.0)
    turret_density: float = info(2000.0)
    drop_x: float = info(-0.10)              # solve's load point along the chute
    drop_hover: float = info(0.040)          # hover height above the resting ball center
    # --- info: ball -----------------------------------------------------------------------------
    ball_r: float = info(0.016)
    ball_density: float = info(2700.0)
    # --- info: vault geometry (vault local; origin = interior floor center at the ground) -------
    vault_dist: float = info(0.51)           # pivot -> vault origin distance
    int_half_x: float = info(0.070)
    int_half_y: float = info(0.080)
    int_floor_z: float = info(0.010)
    wall_z: float = info(0.20)
    wall_t: float = info(0.008)
    mouth_x: float = info(-0.190)            # tunnel mouth plane (vault local)
    ridge_x: float = info(-0.066)            # tunnel end / ridge crest (vault local)
    tun_half_w: float = info(0.045)          # tunnel inner half width (90 mm window)
    tun_floor_z0: float = info(0.070)        # tunnel floor top at the mouth
    tun_floor_z1: float = info(0.082)        # tunnel floor top at the ridge (12 mm rise)
    tun_ceil_z: float = info(0.175)          # tunnel ceiling underside
    flare_len: float = info(0.034)
    flare_deg: float = info(25.0)
    apron_x0: float = info(-0.245)           # apron front edge (vault local)
    apron_z0: float = info(0.052)            # apron floor-top at its front edge
    apron_w: float = info(0.130)
    # --- info: hand-exclusion model (Franka) ----------------------------------------------------
    finger_len: float = info(0.055)          # Franka finger reach past the hand body
    hand_w: float = info(0.100)              # gripper body width (cannot enter the tunnel)
    # --- info: rubric weights (0.20 + 0.20 + 0.30 = 0.70 = the non-success cap) -----------------
    w_aim: float = info(0.20)
    w_load: float = info(0.20)
    w_enter: float = info(0.30)
    contact_offset: float = info(0.0015)

    def __post_init__(self) -> None:
        """Audit the ballistics and exclusion geometry (metres, degrees)."""
        g = 9.81
        # --- chute: ball fits the channel, rails retain it, backstop above the floor
        assert 2 * self.ball_r <= 2 * self.chan_half_w - 0.006
        assert self.rail_h > self.ball_r + 0.010
        assert self.chan_z0 > self.chan_z1 and self.chan_x1 > self.chan_x0
        # --- launch speed: rolling-sphere drop from the load point to the lip
        slope = (self.chan_z0 - self.chan_z1) / (self.chan_x1 - self.chan_x0)
        drop_floor_z = self.chan_z0 - slope * (self.drop_x - self.chan_x0)
        dh = drop_floor_z - self.chan_z1
        v_exit = math.sqrt(2 * g * dh / 1.4)          # rolling sphere: E = 0.7 m v^2
        # --- gap flight: exit center height, mouth-face gap, arrival window
        z_exit = self.chan_z1 + self.ball_r
        lip_r = math.hypot(self.chan_lip_x, self.chan_half_w + self.rail_t)
        gap = (self.vault_dist + self.mouth_x) - self.chan_lip_x
        assert gap > 0.06
        t_fly = gap / v_exit
        z_arr = z_exit - 0.5 * g * t_fly * t_fly
        z_lo = self.tun_floor_z0 + self.ball_r        # lowest clean arrival center
        z_hi = self.tun_ceil_z - self.ball_r          # highest possible arrival center
        assert z_lo + 0.012 <= z_arr <= z_hi - 0.030, (z_arr, z_lo, z_hi)
        # speed floor for a clean (above-lip) arrival, with >= 25% speed margin
        v_min = gap * math.sqrt(g / (2 * (z_exit - z_lo)))
        assert v_exit >= 1.25 * v_min, (v_exit, v_min)
        # --- one-way ridge: arrival energy >= 3x the climb, and the interior drop
        # is too deep to bounce back out
        rise = self.tun_floor_z1 - self.tun_floor_z0
        assert rise > 0.008
        v_gate = math.sqrt(2 * g * rise / 0.7)        # rolling climb threshold
        assert v_exit * v_exit >= 3.0 * v_gate * v_gate, (v_exit, v_gate)
        assert self.tun_floor_z1 - self.int_floor_z >= 0.065
        # --- aiming: latch tolerance inside the clean-capture cone; a 15 deg misaim
        # misses even the flared mouth
        r_mouth = self.vault_dist + self.mouth_x
        clean = self.tun_half_w - self.ball_r
        assert r_mouth * math.sin(math.radians(self.aim_tol_deg)) <= clean - 0.004
        flared = self.tun_half_w + self.flare_len * math.sin(math.radians(self.flare_deg)) \
            + self.ball_r
        assert r_mouth * math.sin(math.radians(15.0)) >= flared + 0.005
        # --- swing clearance: the turret's farthest point clears the apron and the tray
        sweep = max(lip_r, abs(self.knob_x) + self.knob_r,
                    math.hypot(0.2075, 0.012))        # handle arm far corner
        assert (self.vault_dist + self.apron_x0) - sweep >= 0.03
        assert self.tray_r_range[0] - 0.085 - sweep >= 0.03
        # --- bearings: the vault fan and the initial misaim stay inside the joint travel
        assert self.vault_bear_deg + self.aim_off_range[1] - self.vault_bear_deg \
            <= self.yaw_lim_deg - 5.0                 # |theta0| <= off_max
        assert self.vault_bear_deg <= self.yaw_lim_deg - 5.0
        assert self.aim_off_range[0] >= self.aim_tol_deg + 10.0   # null policy scores no aim
        assert 2 * self.yaw_lim_deg <= 175.0          # PhysX 180-wrap margin
        # --- hand exclusion: fingers + held ball fall short of the ridge, the hand
        # body cannot enter the tunnel, and anything released inside rolls back out
        tun_len = self.ridge_x - self.mouth_x
        assert self.finger_len + 2 * self.ball_r + 0.010 < tun_len
        assert self.hand_w > 2 * self.tun_half_w + 0.005
        # (rising floor: a released ball always rolls toward the mouth — rise > 0 above)
        # --- ball spawn clears the tray walls (reset band vs static walls: >= 5 mm)
        assert self.ball_jit + self.ball_r <= 0.050 - 0.005
        # --- success box is reachable at rest and unreachable from outside
        assert self.succ_max_z > self.int_floor_z + self.ball_r + 0.005
        assert self.succ_max_z < self.tun_floor_z1 - 0.010
        assert self.succ_half_y >= self.int_half_y - self.ball_r
        assert self.succ_half_x >= self.int_half_x - self.ball_r - 0.002
        # --- rubric weights: the latched partial credit exactly fills the cap
        assert abs(self.w_aim + self.w_load + self.w_enter - 0.70) < 1e-9


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


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("vault_launcher")
class VaultLauncherScene(BaseScene):
    cfg: VaultLauncherSceneCfg

    def __init__(self, cfg: VaultLauncherSceneCfg | None = None) -> None:
        super().__init__(cfg or VaultLauncherSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        base_spawn = cls["base"](
            base_mass=c.base_mass, plinth_xy=c.plinth_xy, plinth_h=c.plinth_h,
            pedestal_r=c.pedestal_r, pedestal_top=c.pedestal_top,
            contact_offset=c.contact_offset)
        turret_spawn = cls["turret"](
            chan_x0=c.chan_x0, chan_z0=c.chan_z0, chan_x1=c.chan_x1,
            chan_z1=c.chan_z1, chan_lip_x=c.chan_lip_x, chan_half_w=c.chan_half_w,
            chan_floor_t=c.chan_floor_t, rail_h=c.rail_h, rail_t=c.rail_t,
            knob_x=c.knob_x, knob_z=c.knob_z, knob_r=c.knob_r,
            yaw_lim_deg=c.yaw_lim_deg, turret_ang_damping=c.turret_ang_damping,
            turret_density=c.turret_density, contact_offset=c.contact_offset)
        vault_spawn = cls["vault"](
            int_half_x=c.int_half_x, int_half_y=c.int_half_y,
            int_floor_z=c.int_floor_z, wall_z=c.wall_z, wall_t=c.wall_t,
            mouth_x=c.mouth_x, ridge_x=c.ridge_x, tun_half_w=c.tun_half_w,
            tun_floor_z0=c.tun_floor_z0, tun_floor_z1=c.tun_floor_z1,
            tun_ceil_z=c.tun_ceil_z, chan_floor_t=c.chan_floor_t,
            flare_len=c.flare_len, flare_deg=c.flare_deg, apron_x0=c.apron_x0,
            apron_z0=c.apron_z0, apron_w=c.apron_w,
            contact_offset=c.contact_offset)
        tray_spawn = cls["tray"](contact_offset=c.contact_offset)
        ball_spawn = cls["ball"](
            ball_r=c.ball_r, ball_density=c.ball_density,
            contact_offset=c.contact_offset)

        # template poses: the turret MUST spawn consistent with its authored joint
        # frames (base and turret share the origin, yaw 0)
        px, py = c.base_pos
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=base_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "turret": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Turret",
                spawn=turret_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.vault_dist, 0.0, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.0, 0.0)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.0, 0.030)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive bodies via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.base: RigidObject = env.iscene["base"]
        self.turret: RigidObject = env.iscene["turret"]
        self.vault: RigidObject = env.iscene["vault"]
        self.tray: RigidObject = env.iscene["tray"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.vault_bear = torch.zeros(n, device=dev)   # vault bearing (rad, base frame)
        self.init_yaw = torch.zeros(n, device=dev)     # turret initial yaw (rad)
        # latches (partial credit survives transients; success is judged live)
        self._aimed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._aim_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._load_cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the base (xy jitter, yaw 0) and the turret ON ITS
        AXLE at the misaimed initial yaw in the same write (the whole jointed
        linkage moves together), stand the vault at a random bearing on the far
        side with its snout facing the pivot, set the tray behind the launcher,
        and rest the ball in the tray. Clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, device=dev)   # burn the degenerate first post-seed draw

        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.base_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        pp[:, 1] = c.base_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3] = 1.0
        self.base.write_root_state_to_sim(st, env_ids)

        # vault bearing and initial turret misaim (always offset toward the center,
        # so |init yaw| stays well inside the joint travel)
        bear = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.vault_bear_deg)
        o0, o1 = (math.radians(v) for v in c.aim_off_range)
        off = o0 + torch.rand(m, device=dev) * (o1 - o0)
        sgn = torch.where(bear >= 0, -torch.ones(m, device=dev), torch.ones(m, device=dev))
        yaw0 = bear + sgn * off
        self.vault_bear[env_ids] = bear
        self.init_yaw[env_ids] = yaw0

        # turret: on the axle, at the misaimed yaw (pure pose write — the turret
        # body origin sits ON the axle)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = _qz(yaw0)
        self.turret.write_root_state_to_sim(st, env_ids)

        # vault: at the bearing, snout facing the pivot (vault +x = outward)
        vp = torch.zeros(m, 3, device=dev)
        vp[:, 0] = pp[:, 0] + c.vault_dist * torch.cos(bear)
        vp[:, 1] = pp[:, 1] + c.vault_dist * torch.sin(bear)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = vp + origin
        st[:, 3:7] = _qz(bear)
        self.vault.write_root_state_to_sim(st, env_ids)

        # tray: behind the launcher (bearing 180 +/- tray_bear_deg), random yaw
        tb = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_bear_deg)
        r0, r1 = c.tray_r_range
        tr = r0 + torch.rand(m, device=dev) * (r1 - r0)
        tp = torch.zeros(m, 3, device=dev)
        tp[:, 0] = pp[:, 0] + tr * torch.cos(tb)
        tp[:, 1] = pp[:, 1] + tr * torch.sin(tb)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = tp + origin
        st[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
        self.tray.write_root_state_to_sim(st, env_ids)

        # ball: resting in the tray with xy jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tp[:, 0] + (torch.rand(m, device=dev) * 2 - 1) * c.ball_jit
        st[:, 1] = tp[:, 1] + (torch.rand(m, device=dev) * 2 - 1) * c.ball_jit
        st[:, 2] = 0.012 + c.ball_r + 0.002
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        self._aimed[env_ids] = False
        self._aim_cnt[env_ids] = 0
        self._loaded[env_ids] = False
        self._load_cnt[env_ids] = 0
        self._entered[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "turret": self.turret.data.root_state_w[env_ids].clone(),
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "vault_bear": self.vault_bear[env_ids].clone(),
            "init_yaw": self.init_yaw[env_ids].clone(),
            "aimed": self._aimed[env_ids].clone(),
            "aim_cnt": self._aim_cnt[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "load_cnt": self._load_cnt[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.turret.write_root_state_to_sim(state["turret"], env_ids)
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.vault_bear[env_ids] = state["vault_bear"]
        self.init_yaw[env_ids] = state["init_yaw"]
        self._aimed[env_ids] = state["aimed"]
        self._aim_cnt[env_ids] = state["aim_cnt"]
        self._loaded[env_ids] = state["loaded"]
        self._load_cnt[env_ids] = state["load_cnt"]
        self._entered[env_ids] = state["entered"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A gravity BALL LAUNCHER stands on a heavy dark pedestal: a bronze "
            "TURRET that swivels on a vertical axle, carrying an inclined CHUTE "
            "— a 40 mm channel with side rails that runs from a high loading end "
            "(280 mm up, behind a backstop) down a steep ramp to a short flat "
            "run and an open launch LIP. Behind the chute, a post carries a RED "
            "KNOB: the aiming handle. Half a metre away on the far side stands "
            "the blue DELIVERY VAULT: a fully sealed box (floor, walls, roof) "
            "whose only opening is a low SNOUT tunnel facing the launcher — a "
            "90 mm wide mouth with flared cheeks and a small apron, its floor "
            "raised 70 mm off the ground and RISING 12 mm toward the interior, "
            "then dropping 72 mm into the vault. A pale steel BALL (32 mm) "
            "rests in a shallow wooden TRAY behind the launcher. The vault's "
            "bearing across a wide fan, the turret's initial heading (always "
            "well off the vault), the tray's place, and the ball's spot in the "
            "tray all vary per episode.\n"
            "Goal: get the ball INSIDE the vault. The vault is sealed and its "
            "tunnel is too deep, too narrow, and too uphill for a hand-carried "
            "delivery — anything released inside the snout rolls back out. "
            "Instead, aim the launcher: swivel the turret by its red knob until "
            "the chute points squarely at the snout mouth (about 5 degrees of "
            "tolerance), then take the ball from the tray and drop it into the "
            "chute near the high end. Gravity does the rest: the ball rolls "
            "down, launches off the lip at about 1.4 m/s, flies the gap, "
            "threads the mouth, carries the little uphill ridge, and falls "
            "into the vault for good. A misaimed or too-gentle launch leaves "
            "the ball on the open floor, where it can be fetched and tried "
            "again. Success: the ball at rest on the vault's interior floor."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Deliver the steel ball into the sealed blue vault. First swivel "
            "the launcher turret by its red knob until the chute points at the "
            "vault's snout mouth, then take the ball from the tray and drop it "
            "into the chute's high end so it rolls, launches, and flies through "
            "the mouth into the vault. The tunnel cannot be loaded by hand — a "
            "ball placed gently in the snout rolls back out."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def _turret_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.turret.data.root_quat_w,
                                  pos_w - self.turret.data.root_pos_w)

    def turret_yaw_deg(self) -> torch.Tensor:
        """(N,) float: turret yaw about the axle in degrees (base-relative). No
        joint-state API exists on a plain spawn-authored USD joint; this is the
        axle readout from the base-relative quaternion."""
        qb = self.base.data.root_quat_w
        qt = self.turret.data.root_quat_w
        qb_inv = qb * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qb.device)
        rel = _qmul(qb_inv, qt)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 3], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return ang

    def aim_err_deg(self) -> torch.Tensor:
        """(N,) float: signed turret-yaw error off the vault bearing (degrees)."""
        err = self.turret_yaw_deg() - torch.rad2deg(self.vault_bear)
        err = torch.where(err > 180.0, err - 360.0, err)
        err = torch.where(err < -180.0, err + 360.0, err)
        return err

    def _chute_floor_top(self, x: torch.Tensor) -> torch.Tensor:
        """(N,) float: chute floor-top height at turret-local x (incline + flat run)."""
        c = self.cfg
        slope = (c.chan_z0 - c.chan_z1) / (c.chan_x1 - c.chan_x0)
        z = c.chan_z0 - slope * (x - c.chan_x0)
        return torch.where(x > c.chan_x1, torch.full_like(z, c.chan_z1), z)

    def ball_loaded(self) -> torch.Tensor:
        """(N,) bool: the ball is riding the chute — turret-local x inside the
        channel span, |y| between the rails, center within load_z_tol of resting
        on the floor (a hover write above the chute reads False)."""
        c = self.cfg
        p = self._turret_local(self.ball.data.root_pos_w)
        rest = self._chute_floor_top(p[:, 0]) + c.ball_r
        return (p[:, 0] > c.chan_x0 - 0.002) & (p[:, 0] < c.chan_lip_x - 0.004) \
            & (p[:, 1].abs() < c.load_y_tol) \
            & ((p[:, 2] - rest).abs() < c.load_z_tol)

    def ball_entered(self) -> torch.Tensor:
        """(N,) bool: the ball is inside the vault's tunnel airspace or interior
        airspace (vault local). Both regions are physically enclosed except at
        the mouth; a ball hugging the OUTSIDE of the tunnel wall (|y| ~ 0.069 on
        the ground) matches neither branch."""
        c = self.cfg
        p = self._vault_local(self.ball.data.root_pos_w)
        in_tun = (p[:, 0] > c.mouth_x + 0.005) & (p[:, 0] < -c.int_half_x + 0.010) \
            & (p[:, 1].abs() < c.tun_half_w - 0.005) \
            & (p[:, 2] > c.tun_floor_z0) & (p[:, 2] < c.tun_ceil_z)
        in_int = (p[:, 0].abs() < c.int_half_x - 0.004) \
            & (p[:, 1].abs() < c.int_half_y - 0.008) \
            & (p[:, 2] > 0.015) & (p[:, 2] < c.wall_z - 0.010)
        return in_tun | in_int

    def ball_in_vault(self) -> torch.Tensor:
        """(N,) bool: the ball is inside the vault INTERIOR, below the ridge top
        (vault local) — i.e. past the one-way drop, where it cannot leave."""
        c = self.cfg
        p = self._vault_local(self.ball.data.root_pos_w)
        return (p[:, 0].abs() < c.succ_half_x) \
            & (p[:, 1].abs() < c.succ_half_y) \
            & (p[:, 2] < c.succ_max_z)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.base, self.turret, self.vault, self.ball)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        c = self.cfg
        fin = self._finite()
        aim = (self.aim_err_deg().abs() < c.aim_tol_deg) \
            & (self.turret.data.root_ang_vel_w[:, 2].abs() < c.aim_settle_avel) & fin
        self._aim_cnt = torch.where(aim, self._aim_cnt + 1,
                                    torch.zeros_like(self._aim_cnt))
        self._aimed |= self._aim_cnt >= 3
        load = self.ball_loaded() & fin
        self._load_cnt = torch.where(load, self._load_cnt + 1,
                                     torch.zeros_like(self._load_cnt))
        self._loaded |= self._load_cnt >= 3
        self._entered |= self.ball_entered() & fin

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball rests inside the vault interior — past the one-way
        ridge, below the ridge height, settled and finite. The turret is
        deliberately NOT gated: once the ball is in the sealed vault the delivery
        is done, whatever the launcher does afterwards."""
        c = self.cfg
        return self.ball_in_vault() \
            & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*aimed + 0.20*loaded + 0.30*entered (all
        latched; ~0 for the null policy — the ball rests in its tray and the
        turret starts >= 18 deg off aim), capped at 0.70 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_aim * self._aimed.float()
                + c.w_load * self._loaded.float()
                + c.w_enter * self._entered.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="vault_launcher", robot="null"))
