"""IdlerGearboxScene — restore a broken powertrain by dropping the missing IDLER GEAR
onto its bearing peg, then CRANK the completed gear train so its rack pushes the
untouchable payload cube out of a roofed tunnel into the walled catch pocket
(sim_gen task `approach_grasp_spoon_i396`).

Seed (rlbench/approach_grasp_spoon): approach and GRASP a spoon lying among clutter
and CARRY it toward a basket — a prehensile pick-carry-release of the payload itself.
Here that whole strategy is void by construction:

- The payload (an orange 40 mm cube) sits inside a fully ROOFED tunnel and can never
  be touched: its rear port (24 x 24 mm) is smaller than the cube, its roof is closed,
  and its exit hangs over the catch pocket. The ONLY thing that can move it is the
  machine's own rack nose sliding through the port.
- The only object the robot ever grasps is a MACHINE PART: the idler gear (a toothed
  brass wheel with a raised collar hub, pinch-graspable). Installing it — lowering its
  bore over the bearing peg so it drops and seats on the boss — is step one.
- Even installed, nothing happens by itself: the robot must then DRIVE the
  transmission, turning the red crank (counter-clockwise from above) through a
  sustained ~250 degrees so crank gear -> idler -> rack converts rotation into ~13 cm
  of linear rack travel that ejects the cube off the tunnel lip into the pocket.
- A same-looking smooth BLANK wheel (identical collar and bore, no teeth, rim below
  every mesh circle) also seats perfectly on the peg — and transmits nothing: cranking
  with the blank installed spins the crank freely forever (smoke-proved).

So versus the seed the payload is never grasped, carried, or even contacted; the
grasped object is infrastructure; and the delivery mechanism is an actively driven
rotary-to-linear gear train, not a carry. Versus sibling i77 (bridge seat + passive
gravity ride) the machine here must be continuously POWERED after the install —
seating the idler alone delivers nothing. Versus sibling i12 (nonprehensile tipping)
this is a prehensile install plus mechanism drive.

success(): cube settled inside the pocket AND the transmission latch earned — the
rack must have accumulated >= `trans_min` of NET forward travel in signed physical
increments (forward capped at `adv_step_max`, backward subtracted in full, floored
at 0) WHILE the idler was seated — teleport jumps credit one capped step and their
depenetration oscillation cannot ratchet, so teleports and rack-wrench cheats
cannot fake it; the geometry already hides the rack behind walls, a roof strip,
and the joint-filtered west wall.

score(), latched: 0.25 idler ever seated on the peg + 0.20 geared rack advance
>= trans_min + 0.20 cube ejected past the lip AFTER transmission; capped 0.65;
exactly 1.0 iff success(). Null policy earns exactly 0.

Assets are fully procedural (compound spawners; child colliders never self-collide):
  - frame (DYNAMIC 60 kg — dynamic so reset teleports carry the spawn-authored joint
    anchors; heavy + damped so drive reactions never budge it): base plate, gear-well
    walls, bearing boss + stepped peg (r12 -> r11 -> r8.5 capture taper), rack guide
    (outer wall + roof strips that clear the idler's drop disc by >3 mm), rear tunnel
    wall with the 24 x 24 port, roofed tunnel, and the walled catch pocket.
  - crank (dynamic): 8-tooth gear (pitch r 30.56 mm) + shaft + overhead arm + red
    knob post, on a spawn-authored FREE revolute joint (axis Z, no limits — it must
    turn multiple revolutions) to the frame.
  - rack (dynamic): toothed bar (8 teeth, 24 mm pitch, pitch line 38.2 mm from the
    peg axis) + pusher nose, on a spawn-authored prismatic joint (axis X, stops
    [0, stroke]) to the frame.
  - idler (dynamic, free): 10-tooth gear (pitch r 38.2 mm), octagonal bore
    (inradius 15 mm) over the 12 mm peg, collar hub (44 mm across) for the pinch.
  - blank (dynamic, free): identical bore + collar, smooth 29 mm rim — seats but
    can reach neither the crank teeth nor the rack teeth (asserted).
  - cube (dynamic): the 40 mm payload, spawned inside the tunnel.
Mesh arithmetic (centre distance = sum of pitch radii, tip/root clearances >= 0.4 mm,
backlash > twice the bore slop, port < cube, roof strips outside the idler disc) is
asserted in `__post_init__` so a bad edit fails at import, not on the forge.

Per-episode randomization (readback-verified in smoke): whole-machine yaw +-180 deg
and xy jitter (whole linkage re-written consistently); crank start angle +-180 deg;
Bernoulli idler/blank ground-slot swap + per-wheel xy jitter + free yaw; cube start
depth in the tunnel.

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

FRANKA_JAW_SPAN = 0.080  # Franka parallel-jaw max opening (m)


# ----- torch quaternion helpers (module-level, app-free) ---------------------------------------
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
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qapply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _qapply(_qinv(q), v)


def _rz(a: float) -> tuple:
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _friction_material(stage, path: str, static: float, dynamic: float):
    """Explicit physics material (custom spawners bind NO cfg material — the
    default ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None) -> None:
    """One box collider child (translate -> orient -> scale, authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(seg.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        _bind(seg.GetPrim(), material)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """One z-axis cylinder collider child."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        _bind(cyl.GetPrim(), material)


def _root_dynamic(prim_path: str, translation, orientation, *, mass, com, inertia,
                  lin_damp, ang_damp, kinematic=False):
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
    ma = UsdPhysics.MassAPI.Apply(root)
    ma.CreateMassAttr(float(mass))
    ma.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _gear_teeth(stage, prim_path: str, n: int, pitch_r: float, tooth_rad: float,
                tooth_circ: float, tooth_h: float, zc: float, color, co: float,
                material) -> None:
    """n radial box teeth on a pitch circle, each rotated to its spoke angle."""
    for k in range(n):
        a = 2 * math.pi * k / n
        _box(stage, f"{prim_path}/tooth_{k}", (tooth_rad, tooth_circ, tooth_h),
             (pitch_r * math.cos(a), pitch_r * math.sin(a), zc), color, co,
             material=material, orient=_rz(a))


def _bore_ring(stage, prim_path: str, tag: str, inradius: float, outer_r: float,
               height: float, zc: float, color, co: float, material) -> None:
    """8-segment octagonal annulus: bore inradius `inradius`, rim `outer_r`."""
    rad_t = outer_r - inradius
    mid_r = (outer_r + inradius) / 2
    tang = 2 * outer_r * math.tan(math.pi / 8) * 1.0
    for k in range(8):
        a = 2 * math.pi * k / 8
        _box(stage, f"{prim_path}/{tag}_{k}", (rad_t, tang, height),
             (mid_r * math.cos(a), mid_r * math.sin(a), zc), color, co,
             material=material, orient=_rz(a))


def _spawn_frame(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The machine frame: ONE DYNAMIC heavy compound. Origin: peg axis at (0,0),
    z = 0 at the ground. +x is the rack travel direction."""
    c = cfg
    stage, root = _root_dynamic(
        prim_path, translation, orientation, mass=c.mass, com=(0.03, 0.0, 0.03),
        inertia=(2.0, 2.0, 2.0), lin_damp=2.0, ang_damp=4.0)
    mat = _friction_material(stage, f"{prim_path}/mat", c.mu_static, c.mu_dynamic)
    slick = _friction_material(stage, f"{prim_path}/mat_slick", 0.15, 0.12)
    co = c.contact_offset
    grey = (0.45, 0.47, 0.52)
    dark = (0.28, 0.30, 0.35)
    green = (0.15, 0.45, 0.20)
    # --- base plate (gear well floor) + tunnel floor ---
    _box(stage, f"{prim_path}/base", (0.22, 0.24, 0.008), (-0.05, -0.02, 0.004),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/tunnel_floor", (0.13, 0.07, 0.008),
         (0.125, 0.0552, 0.004), grey, co, material=mat)
    # --- gear-well perimeter walls (west + south), height 50 mm ---
    _box(stage, f"{prim_path}/wall_w", (0.012, 0.264, 0.05), (-0.166, -0.02, 0.025),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/wall_s", (0.234, 0.012, 0.05), (-0.055, -0.146, 0.025),
         dark, co, material=mat)
    # --- bearing boss + stepped capture peg (slick: the idler must spin on it) ---
    _cyl(stage, f"{prim_path}/boss", c.boss_r, 0.012, (0.0, 0.0, 0.014),
         dark, co, material=slick)
    _cyl(stage, f"{prim_path}/peg", c.peg_r, 0.040, (0.0, 0.0, 0.040),
         dark, co, material=slick)
    _cyl(stage, f"{prim_path}/peg_s1", 0.011, 0.008, (0.0, 0.0, 0.064),
         dark, co, material=slick)
    _cyl(stage, f"{prim_path}/peg_s2", 0.0085, 0.008, (0.0, 0.0, 0.072),
         dark, co, material=slick)
    # --- rack guide: outer wall + roof strips (clear of the idler's drop disc) ---
    _box(stage, f"{prim_path}/rk_wall", (0.22, 0.012, 0.044), (-0.05, 0.074, 0.030),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/rk_roof", (0.22, 0.019, 0.008), (-0.05, 0.0585, 0.048),
         dark, co, material=mat)
    _box(stage, f"{prim_path}/rk_roof_w", (0.118, 0.023, 0.008),
         (-0.101, 0.0375, 0.048), dark, co, material=mat)
    _box(stage, f"{prim_path}/rk_roof_e", (0.018, 0.023, 0.008),
         (0.051, 0.0375, 0.048), dark, co, material=mat)
    # --- rear tunnel wall with the 24 x 24 port (frame around the opening) ---
    _box(stage, f"{prim_path}/port_l", (0.012, 0.023, 0.054), (0.066, 0.0317, 0.035),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/port_r", (0.012, 0.023, 0.054), (0.066, 0.0787, 0.035),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/port_b", (0.012, 0.024, 0.010), (0.066, 0.0552, 0.013),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/port_t", (0.012, 0.024, 0.020), (0.066, 0.0552, 0.052),
         grey, co, material=mat)
    # --- roofed tunnel (walls + roof; the payload is untouchable inside) ---
    _box(stage, f"{prim_path}/tun_l", (0.13, 0.012, 0.054), (0.125, 0.0242, 0.035),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/tun_r", (0.13, 0.012, 0.054), (0.125, 0.0862, 0.035),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/tun_roof", (0.118, 0.074, 0.008),
         (0.131, 0.0552, 0.058), grey, co, material=mat)
    # --- catch pocket (walls on the ground; floor is the ground plane) ---
    _box(stage, f"{prim_path}/pk_l", (0.106, 0.012, 0.08), (0.249, 0.0042, 0.04),
         green, co, material=mat)
    _box(stage, f"{prim_path}/pk_r", (0.106, 0.012, 0.08), (0.249, 0.1062, 0.04),
         green, co, material=mat)
    _box(stage, f"{prim_path}/pk_b", (0.012, 0.114, 0.08), (0.296, 0.0552, 0.04),
         green, co, material=mat)
    return root


def _spawn_crank(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The crank: 8-tooth gear + shaft + overhead arm + red knob post, ONE dynamic
    body on a spawn-authored FREE revolute joint (axis Z, no limits) to the sibling
    Frame. Origin ON the axis, z = 0 at frame-floor level."""
    from pxr import Gf, UsdPhysics

    c = cfg
    stage, root = _root_dynamic(
        prim_path, translation, orientation, mass=c.mass, com=(0.012, 0.0, 0.06),
        inertia=(0.004, 0.004, 0.0015), lin_damp=0.05, ang_damp=c.ang_damp)
    mat = _friction_material(stage, f"{prim_path}/mat", c.mu, c.mu)
    co = c.contact_offset
    steel = (0.55, 0.57, 0.62)
    red = (0.85, 0.13, 0.13)
    _cyl(stage, f"{prim_path}/hub", c.root_r, 0.020, (0.0, 0.0, 0.030),
         steel, co, material=mat)
    _gear_teeth(stage, prim_path, c.n_teeth, c.pitch_r, c.tooth_rad, c.tooth_circ,
                0.020, 0.030, steel, co, mat)
    _cyl(stage, f"{prim_path}/shaft", 0.008, 0.090, (0.0, 0.0, 0.085),
         steel, co, material=mat)
    _box(stage, f"{prim_path}/arm", (0.075, 0.018, 0.012), (0.030, 0.0, 0.136),
         red, co, material=mat)
    _cyl(stage, f"{prim_path}/knob", 0.008, 0.032, (0.060, 0.0, 0.158),
         red, co, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.axis_x), float(c.axis_y), 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # NO limit attrs: the crank must turn continuously (multi-revolution drive).
    return root


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The rack: toothed bar + pusher nose, ONE dynamic body on a spawn-authored
    prismatic joint (axis X, stops [0, stroke]) to the sibling Frame. Origin at the
    bar centre-line (z = 0 at bar mid-height)."""
    from pxr import Gf, UsdPhysics

    c = cfg
    stage, root = _root_dynamic(
        prim_path, translation, orientation, mass=c.mass, com=(0.0, 0.0, 0.0),
        inertia=(2e-4, 1.5e-3, 1.5e-3), lin_damp=c.lin_damp, ang_damp=1.0)
    mat = _friction_material(stage, f"{prim_path}/mat", c.mu, c.mu)
    co = c.contact_offset
    steel = (0.62, 0.60, 0.55)
    _box(stage, f"{prim_path}/bar", (0.227, 0.020, 0.020), (0.0085, 0.0, 0.0),
         steel, co, material=mat)
    for k in range(8):
        xk = -0.093 + 0.024 * k
        _box(stage, f"{prim_path}/tooth_{k}", (0.008, 0.0135, 0.020),
             (xk, -0.01675, 0.0), steel, co, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Frame"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.x0), 0.0552, 0.030))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(c.stroke))
    return root


def _spawn_wheel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A free wheel: octagonal bore ring + collar hub; the IDLER adds 10 teeth,
    the BLANK a smooth (sub-mesh) rim. Origin at ring mid-height. The collar bore
    is WIDER than the ring bore so only the 20 mm ring ever guides on the peg
    (short engagement = jam-free gravity insertion)."""
    c = cfg
    stage, root = _root_dynamic(
        prim_path, translation, orientation, mass=c.mass, com=(0.0, 0.0, 0.004),
        inertia=(2.2e-4, 2.2e-4, 3.5e-4), lin_damp=0.5, ang_damp=0.1)
    mat = _friction_material(stage, f"{prim_path}/mat", c.mu, c.mu)
    slick = _friction_material(stage, f"{prim_path}/mat_bore", 0.15, 0.12)
    co = c.contact_offset
    brass = (0.76, 0.62, 0.22)
    _bore_ring(stage, prim_path, "ring", c.bore_in, c.ring_out, 0.020, 0.0,
               brass, co, slick)
    _bore_ring(stage, prim_path, "collar", c.collar_in, c.collar_out, 0.018, 0.019,
               brass, co, mat)
    if c.toothed:
        _gear_teeth(stage, prim_path, c.n_teeth, c.pitch_r, c.tooth_rad,
                    c.tooth_circ, 0.020, 0.0, brass, co, mat)
    return root


def _spawn_cube(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The payload cube."""
    c = cfg
    stage, root = _root_dynamic(
        prim_path, translation, orientation, mass=c.mass, com=(0.0, 0.0, 0.0),
        inertia=(3.5e-5, 3.5e-5, 3.5e-5), lin_damp=0.2, ang_damp=0.2)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.4, 0.35)
    _box(stage, f"{prim_path}/body", (c.edge, c.edge, c.edge), (0.0, 0.0, 0.0),
         (0.92, 0.48, 0.10), c.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "frame" not in _SPAWNER_CACHE:

        @configclass
        class FrameSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_frame)
            mass: float = 60.0
            boss_r: float = 0.020
            peg_r: float = 0.012
            mu_static: float = 0.7
            mu_dynamic: float = 0.6
            contact_offset: float = 0.0015

        @configclass
        class CrankSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crank)
            axis_x: float = -0.04863
            axis_y: float = -0.04863
            n_teeth: int = 8
            pitch_r: float = 0.03056
            root_r: float = 0.0235
            tooth_rad: float = 0.013
            tooth_circ: float = 0.007
            mass: float = 0.6
            ang_damp: float = 0.3
            mu: float = 0.2
            contact_offset: float = 0.0015

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            x0: float = -0.048
            stroke: float = 0.130
            mass: float = 0.35
            lin_damp: float = 0.5
            mu: float = 0.2
            contact_offset: float = 0.0015

        @configclass
        class WheelSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wheel)
            toothed: bool = True
            n_teeth: int = 10
            pitch_r: float = 0.0382
            ring_out: float = 0.0312
            bore_in: float = 0.015
            collar_in: float = 0.018
            collar_out: float = 0.022
            tooth_rad: float = 0.013
            tooth_circ: float = 0.007
            mass: float = 0.25
            mu: float = 0.2
            contact_offset: float = 0.0015

        @configclass
        class CubeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cube)
            edge: float = 0.040
            mass: float = 0.12
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(frame=FrameSpawnerCfg, crank=CrankSpawnerCfg,
                              rack=RackSpawnerCfg, wheel=WheelSpawnerCfg,
                              cube=CubeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class IdlerGearboxSceneCfg(BaseCfg):
    """Config for `IdlerGearboxScene`. `__post_init__` asserts the gear-train
    arithmetic and the honesty gates: pitch circles sum to the centre distance;
    tip/root clearances are positive; backlash exceeds twice the bore slop; the
    blank's rim reaches NO mesh circle; the port is smaller than the cube; the roof
    strips clear the idler's vertical drop disc; the seat band excludes a wheel
    resting on the well floor; the full stroke ejects the cube from any spawn."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_xy_tol: float = tunable(0.008)  # idler centre offset from the peg axis (m)
    seat_z_lo: float = tunable(0.024)  # idler origin height band when seated (m)...
    seat_z_hi: float = tunable(0.036)  # ...seated = 0.030; floor-rest = 0.018 (out)
    seat_upright: float = tunable(0.90)  # min cos(axis tilt) for "seated"
    trans_min: float = tunable(0.060)  # NET geared rack advance for the transmission latch
    adv_step_max: float = tunable(0.002)  # per-step advance credit clamp (anti-teleport)
    eject_x: float = tunable(0.196)  # cube frame-x past the lip = "ejected"
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.8)  # max |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must persist this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_deg: float = tunable(180.0)  # whole-machine uniform +-yaw
    xy_jitter: float = tunable(0.05)  # whole-machine +-xy (m)
    crank0_deg: float = tunable(180.0)  # crank start angle uniform +- this
    swap_slots: bool = tunable(True)  # Bernoulli idler/blank ground-slot swap
    slot_jitter: float = tunable(0.02)  # per-wheel ground xy jitter (+-m)
    slot_yaw_deg: float = tunable(180.0)  # per-wheel free yaw
    cube_x_lo: float = tunable(0.100)  # cube spawn depth band (frame x, m)
    cube_x_hi: float = tunable(0.140)

    # --- info: gear train arithmetic ---------------------------------------------------------
    crank_axis: tuple = info((-0.04863, -0.04863))  # crank axis in the frame (m)
    crank_teeth: int = info(8)
    crank_pitch_r: float = info(0.03056)  # 8 * 0.024 / 2pi
    crank_root_r: float = info(0.0235)
    idler_teeth: int = info(10)
    idler_pitch_r: float = info(0.0382)  # 10 * 0.024 / 2pi
    idler_ring_out: float = info(0.0312)
    tooth_pitch: float = info(0.024)  # circumferential pitch, all members
    tooth_circ: float = info(0.007)  # tooth width (17 mm gaps: 10 mm backlash)
    tooth_rad: float = info(0.013)  # radial tooth depth
    rack_line_y: float = info(0.0552)  # rack bar centre-line y (frame)
    rack_x0: float = info(-0.048)  # rack origin frame-x at q = 0
    rack_stroke: float = info(0.130)  # prismatic stops [0, stroke]
    rack_nose_x: float = info(0.122)  # pusher nose, rack-local x

    # --- info: bearing + wheels --------------------------------------------------------------
    peg_r: float = info(0.012)
    boss_r: float = info(0.020)
    bore_in: float = info(0.015)  # octagonal bore inradius (ring only guides)
    collar_in: float = info(0.018)  # collar bore, WIDER: no peg engagement
    collar_out: float = info(0.022)  # pinch-hub octagon radius (44 mm across)
    blank_rim: float = info(0.029)  # smooth blank rim (reaches NO mesh circle)
    seat_origin_z: float = info(0.030)  # idler origin height when seated on the boss
    wheel_mass: float = info(0.25)

    # --- info: tunnel / port / pocket --------------------------------------------------------
    cube_edge: float = info(0.040)
    port_wh: float = info(0.024)  # port opening width = height
    tunnel_x1: float = info(0.190)  # tunnel lip (frame x)
    pocket_x0: float = info(0.202)  # cube-centre success band in the pocket
    pocket_x1: float = info(0.284)
    pocket_hw: float = info(0.040)  # |y - rack_line_y| band
    pocket_z_hi: float = info(0.036)  # cube on the ground: centre ~0.020
    slot_a: tuple = info((0.10, -0.22))  # wheel ground slot A (frame local)
    slot_b: tuple = info((-0.20, -0.22))  # wheel ground slot B

    frame_mass: float = info(60.0)
    crank_mass: float = info(0.6)
    rack_mass: float = info(0.35)
    cube_mass: float = info(0.12)
    contact_offset: float = info(0.0015)
    w_seat: float = info(0.25)
    w_trans: float = info(0.20)
    w_eject: float = info(0.20)

    # Derived (filled in __post_init__).
    centre_dist: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        ax, ay = self.crank_axis
        self.centre_dist = math.hypot(ax, ay)
        # --- pitch circles sum to the centre distance (gear law) ---
        assert abs(self.centre_dist - (self.crank_pitch_r + self.idler_pitch_r)) < 5e-4, \
            "crank-idler centre distance must equal the sum of pitch radii"
        assert abs(self.rack_line_y - 0.017 - self.idler_pitch_r) < 5e-4, \
            "idler-rack pitch line must sit one pitch radius from the peg axis"
        # --- same circumferential pitch on every member ---
        for n, r in ((self.crank_teeth, self.crank_pitch_r),
                     (self.idler_teeth, self.idler_pitch_r)):
            assert abs(2 * math.pi * r / n - self.tooth_pitch) < 2e-4, \
                "tooth pitch mismatch: gears would clash"
        # --- tip/root clearances positive on both meshes ---
        c_tip = self.crank_pitch_r + self.tooth_rad / 2  # crank tooth tip radius
        i_tip = self.idler_pitch_r + self.tooth_rad / 2
        i_root = self.idler_pitch_r - self.tooth_rad / 2
        assert self.centre_dist - c_tip - self.idler_ring_out >= 4e-4, \
            "crank tips must clear the idler root face"
        assert self.centre_dist - i_tip - self.crank_root_r >= 4e-4, \
            "idler tips must clear the crank root cylinder"
        rack_tip_y = self.rack_line_y - 0.010 - self.tooth_rad  # rack tooth tip y
        assert rack_tip_y - i_root >= 4e-4, "idler root must clear the rack tooth tips"
        assert (self.rack_line_y - 0.010) - i_tip >= 4e-4, \
            "idler tips must clear the rack bar face"
        # --- backlash swallows the bore slop (worst-case mesh still turns) ---
        bore_circum = self.bore_in / math.cos(math.pi / 8)
        slop = bore_circum - self.peg_r  # max idler centre wander on the peg
        backlash = self.tooth_pitch - 2 * self.tooth_circ
        assert backlash >= 2 * slop + 0.001, \
            f"backlash {backlash:.4f} must exceed twice the bore slop {slop:.4f}"
        # --- the blank seats but transmits NOTHING (rim below every mesh circle) ---
        assert self.blank_rim <= self.centre_dist - c_tip - 0.0015, \
            "blank rim must never touch the crank teeth"
        assert self.blank_rim <= rack_tip_y - 0.0015, \
            "blank rim must never touch the rack teeth"
        # --- the bore drops over the stepped peg; the collar fits the jaw ---
        assert self.bore_in - self.peg_r >= 0.0025, "bore must clear the peg"
        assert self.collar_in >= self.bore_in + 0.002, \
            "collar bore must be wider than the ring bore (short peg engagement)"
        assert self.collar_in <= self.collar_out - 0.003, \
            "collar wall must keep real thickness"
        assert self.boss_r > bore_circum, "the boss must be a full seat under the bore"
        assert 2 * self.collar_out <= FRANKA_JAW_SPAN - 0.020, \
            "collar hub must pinch inside the Franka jaw with margin"
        # --- the payload is untouchable; only the rack nose reaches it ---
        assert self.port_wh <= self.cube_edge - 0.012, \
            "port must be much smaller than the cube (no pull-back, no direct push)"
        assert self.cube_x_lo - self.cube_edge / 2 >= 0.072 + 0.006, \
            "cube must spawn clear of the rear wall and the parked nose"
        nose_end = self.rack_nose_x + self.rack_x0 + self.rack_stroke
        assert nose_end + self.cube_edge / 2 >= self.pocket_x0 + 0.018, \
            "full stroke must push the cube centre well into the pocket band"
        assert nose_end <= self.pocket_x0 + self.cube_edge, \
            "the nose itself must stop near the pocket entry (rack stays captive)"
        # --- roof strips clear the idler's vertical drop disc ---
        assert 0.049 - i_tip >= 0.003, "main roof strip must clear the idler disc"
        # y-span of the idler disc at the side-roof strips' inner edge x = +-0.042
        assert 0.042 - math.sqrt(max(i_tip**2 - 0.026**2, 0.0)) >= 0.003, \
            "side roof strips must clear the idler disc"
        # --- the seat band excludes a wheel resting on the well floor ---
        assert self.seat_z_lo <= self.seat_origin_z <= self.seat_z_hi
        assert 0.018 < self.seat_z_lo - 0.004, \
            "a wheel on the well floor (origin z=0.018) must sit outside the seat band"
        # resting on tooth tops (origin ~0.050) is also outside the band
        assert 0.050 > self.seat_z_hi + 0.004
        # --- transmission needs sustained physical motion, never one teleport ---
        assert self.trans_min >= 10 * self.adv_step_max, \
            "transmission latch must need >= 10 clamped steps of geared advance"
        # 2x above the ~31 mm peak a full-stroke teleport barrage can creep out of
        # depenetration, and 2x below the ~120 mm an honest full-stroke drive earns
        assert self.trans_min <= self.rack_stroke / 2, \
            "transmission latch must be earnable well within one stroke"
        # --- pocket bands sit inside the walls; the lip x splits tunnel from pocket ---
        assert self.tunnel_x1 < self.eject_x < self.pocket_x0
        assert self.pocket_x1 <= 0.290 - self.cube_edge / 2 + 0.014
        # rack advance per crank turn (for the solve's cranking budget)
        self.rack_per_rev: float = 2 * math.pi * self.crank_pitch_r


# ----- scene -----------------------------------------------------------------------------------
class IdlerGearboxScene(BaseScene):
    cfg: IdlerGearboxSceneCfg

    def __init__(self, cfg: IdlerGearboxSceneCfg | None = None) -> None:
        super().__init__(cfg or IdlerGearboxSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        mp = sim_utils.MassPropertiesCfg
        rp = sim_utils.RigidBodyPropertiesCfg

        def wheel_spawn(toothed: bool):
            return sp["wheel"](
                mass_props=mp(mass=c.wheel_mass), rigid_props=rp(),
                toothed=toothed, n_teeth=c.idler_teeth, pitch_r=c.idler_pitch_r,
                ring_out=(c.idler_ring_out if toothed else c.blank_rim),
                bore_in=c.bore_in, collar_in=c.collar_in, collar_out=c.collar_out,
                tooth_rad=c.tooth_rad, tooth_circ=c.tooth_circ,
                mass=c.wheel_mass, contact_offset=c.contact_offset)

        ax, ay = c.crank_axis
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
            "frame": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frame",
                spawn=sp["frame"](mass_props=mp(mass=c.frame_mass), rigid_props=rp(),
                                  mass=c.frame_mass, boss_r=c.boss_r, peg_r=c.peg_r,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002)),
            ),
            "crank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crank",
                spawn=sp["crank"](mass_props=mp(mass=c.crank_mass), rigid_props=rp(),
                                  axis_x=ax, axis_y=ay, n_teeth=c.crank_teeth,
                                  pitch_r=c.crank_pitch_r, root_r=c.crank_root_r,
                                  tooth_rad=c.tooth_rad, tooth_circ=c.tooth_circ,
                                  mass=c.crank_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.002)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=sp["rack"](mass_props=mp(mass=c.rack_mass), rigid_props=rp(),
                                 x0=c.rack_x0, stroke=c.rack_stroke, mass=c.rack_mass,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x0, c.rack_line_y, 0.032)),
            ),
            "idler": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Idler",
                spawn=wheel_spawn(True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], 0.012)),
            ),
            "blank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Blank",
                spawn=wheel_spawn(False),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], 0.012)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sp["cube"](mass_props=mp(mass=c.cube_mass), rigid_props=rp(),
                                 edge=c.cube_edge, mass=c.cube_mass,
                                 contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.120, c.rack_line_y, 0.032)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.frame: RigidObject = env.iscene["frame"]
        self.crank: RigidObject = env.iscene["crank"]
        self.rack: RigidObject = env.iscene["rack"]
        self.idler: RigidObject = env.iscene["idler"]
        self.blank: RigidObject = env.iscene["blank"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.seated_ever = torch.zeros(n, device=dev)
        self.geared_adv = torch.zeros(n, device=dev)  # clamped geared rack advance
        self.eject_ever = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, device=dev)
        self._prev_rack_q = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the WHOLE LINKAGE (frame + crank + rack) re-written
        consistently from one sampled machine pose (yaw +-180 deg, xy jitter) with a
        random crank angle and the rack at its q=0 stop; idler/blank Bernoulli
        slot-swapped on the ground (+ jitter + free yaw); cube depth sampled in the
        tunnel; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn (degenerate-first-draw trap)

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        th0 = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.crank0_deg)
        cube_x = c.cube_x_lo + torch.rand(m, device=dev) * (c.cube_x_hi - c.cube_x_lo)

        half = yaw / 2
        qf = torch.stack([torch.cos(half), torch.zeros_like(half),
                          torch.zeros_like(half), torch.sin(half)], dim=-1)
        pf = torch.zeros(m, 3, device=dev)
        pf[:, 0:2] = jit
        pf[:, 2] = 0.002
        pf += origin

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pf
        st[:, 3:7] = qf
        self.frame.write_root_state_to_sim(st, env_ids)

        # crank: on its axis, angle th0
        ax, ay = c.crank_axis
        off = torch.tensor([ax, ay, 0.0], device=dev).expand(m, 3)
        hh = th0 / 2
        qc = torch.stack([torch.cos(hh), torch.zeros_like(hh),
                          torch.zeros_like(hh), torch.sin(hh)], dim=-1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pf + _qapply(qf, off)
        st[:, 3:7] = _qmul(qf, qc)
        self.crank.write_root_state_to_sim(st, env_ids)

        # rack: at the q=0 stop
        off = torch.tensor([c.rack_x0, c.rack_line_y, 0.030], device=dev).expand(m, 3)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pf + _qapply(qf, off)
        st[:, 3:7] = qf
        self.rack.write_root_state_to_sim(st, env_ids)

        # cube: sampled depth in the tunnel
        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = cube_x
        off[:, 1] = c.rack_line_y
        off[:, 2] = 0.030
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pf + _qapply(qf, off)
        st[:, 3:7] = qf
        self.cube.write_root_state_to_sim(st, env_ids)

        # wheels: Bernoulli slot swap + jitter + free yaw, flat on the ground
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        for body, xy in ((self.idler, torch.where(swap.unsqueeze(1), slot_b, slot_a)),
                         (self.blank, torch.where(swap.unsqueeze(1), slot_a, slot_b))):
            off = torch.zeros(m, 3, device=dev)
            off[:, 0:2] = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            off[:, 2] = 0.012
            wyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.slot_yaw_deg)
            hw = wyaw / 2
            qw = torch.stack([torch.cos(hw), torch.zeros_like(hw),
                              torch.zeros_like(hw), torch.sin(hw)], dim=-1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pf + _qapply(qf, off)
            st[:, 3:7] = _qmul(qf, qw)
            body.write_root_state_to_sim(st, env_ids)

        for t in (self.seated_ever, self.geared_adv, self.eject_ever,
                  self.still_count):
            t[env_ids] = 0.0
        self._prev_rack_q[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "frame": self.frame.data.root_state_w[env_ids].clone(),
            "crank": self.crank.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "idler": self.idler.data.root_state_w[env_ids].clone(),
            "blank": self.blank.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([
                self.seated_ever[env_ids], self.geared_adv[env_ids],
                self.eject_ever[env_ids], self.still_count[env_ids],
                self._prev_rack_q[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("frame", "crank", "rack", "idler", "blank", "cube"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        lat = state["latches"]
        (self.seated_ever[env_ids], self.geared_adv[env_ids],
         self.eject_ever[env_ids], self.still_count[env_ids],
         self._prev_rack_q[env_ids]) = (lat[:, 0], lat[:, 1], lat[:, 2],
                                        lat[:, 3], lat[:, 4])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey MACHINE FRAME stands on the floor (its position and heading "
            f"change every episode). On its deck is an open GEAR WELL: a steel "
            f"crank gear on a vertical shaft topped by a RED CRANK ARM with a knob, "
            f"and next to the gear an empty BEARING PEG (a {2 * c.peg_r * 1000:.0f} mm "
            f"stepped post on a round boss) where a gear is obviously missing. "
            f"Behind a guide wall a toothed RACK bar runs on rails toward a roofed "
            f"TUNNEL; inside the tunnel sits an ORANGE cube "
            f"({c.cube_edge * 1000:.0f} mm). The tunnel is sealed: its rear port "
            f"({c.port_wh * 1000:.0f} x {c.port_wh * 1000:.0f} mm) is far smaller "
            f"than the cube and its roof is closed, so NOTHING can reach the cube "
            f"except the machine's own rack nose. The tunnel's far end opens over a "
            f"walled GREEN catch pocket on the floor.\n"
            f"On the ground nearby lie TWO brass wheels with identical collar hubs "
            f"(pinch the collar, {2 * c.collar_out * 1000:.0f} mm across) and "
            f"identical centre bores: one has TEETH all around (the missing IDLER "
            f"GEAR), the other is a smooth-rimmed BLANK that fits the peg but "
            f"reaches neither the crank gear nor the rack. Which wheel lies where "
            f"changes between episodes.\n"
            f"Goal: the orange cube must end up resting inside the green pocket. "
            f"The only way: FIRST pick up the TOOTHED wheel by its collar and lower "
            f"its bore over the bearing peg so it drops and seats on the boss, "
            f"completing the gear train; THEN turn the red crank COUNTER-CLOCKWISE "
            f"(seen from above) through a sustained near-full turn — crank gear "
            f"drives idler, idler drives rack — so the advancing rack pushes the "
            f"cube along the tunnel and off the lip into the pocket. Cranking "
            f"without the idler (or with the smooth blank installed) spins freely "
            f"and moves nothing; the wrong direction just parks the rack on its "
            f"stop. Installing the idler but never cranking delivers nothing. A "
            f"cube resting anywhere but inside the pocket is failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the toothed brass wheel by its collar and drop it onto the "
            "empty bearing peg to complete the gear train, then turn the red crank "
            "counter-clockwise until the rack pushes the orange cube out of the "
            "tunnel into the green pocket. The smooth wheel is a blank; cranking "
            "without the toothed idler moves nothing."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body: RigidObject) -> torch.Tensor:
        rel = body.data.root_pos_w - self.frame.data.root_pos_w
        return _qapply_inv(self.frame.data.root_quat_w, rel)

    def rack_q(self) -> torch.Tensor:
        """(N,) rack joint position (0 = retracted stop) from pose readback."""
        return self._local(self.rack)[:, 0] - self.cfg.rack_x0

    def crank_angle(self) -> torch.Tensor:
        """(N,) crank angle about +z relative to the frame, wrapped to (-pi, pi]."""
        rel = _qmul(_qinv(self.frame.data.root_quat_w), self.crank.data.root_quat_w)
        return 2.0 * torch.atan2(rel[:, 3], rel[:, 0])

    def idler_seated(self) -> torch.Tensor:
        """(N,) bool: the IDLER (the toothed wheel, specifically) down on the boss:
        centred on the peg, origin in the seat height band, upright."""
        c = self.cfg
        loc = self._local(self.idler)
        up_w = _qapply(self.idler.data.root_quat_w,
                       torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(
                           loc.shape[0], 3))
        fup_w = _qapply(self.frame.data.root_quat_w,
                        torch.tensor([0.0, 0.0, 1.0], device=loc.device).expand(
                            loc.shape[0], 3))
        return ((loc[:, 0:2].norm(dim=-1) < c.seat_xy_tol)
                & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)
                & ((up_w * fup_w).sum(dim=-1) > c.seat_upright))

    def cube_in_pocket(self) -> torch.Tensor:
        c = self.cfg
        loc = self._local(self.cube)
        return ((loc[:, 0] > c.pocket_x0) & (loc[:, 0] < c.pocket_x1)
                & ((loc[:, 1] - c.rack_line_y).abs() < c.pocket_hw)
                & (loc[:, 2] > 0.006) & (loc[:, 2] < c.pocket_z_hi))

    def _finite(self) -> torch.Tensor:
        ok = torch.ones_like(self.still_count, dtype=torch.bool)
        for body in (self.frame, self.crank, self.rack, self.idler, self.blank,
                     self.cube):
            ok &= torch.isfinite(body.data.root_state_w).all(dim=-1)
        return ok

    def _still_now(self) -> torch.Tensor:
        """Stillness of the PAYLOAD and the machine base only. The gear train is
        deliberately excluded: a seated ring-on-peg idler meshed against sharp
        tooth edges sustains a layout-dependent PhysX micro-swing limit cycle
        (velocity readings in the 0.05 m/s band with frozen poses), which would
        block success forever on some episodes. Delivery anti-cheat lives in the
        transmission latch, not in mechanism stillness."""
        c = self.cfg
        return ((self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.cube.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.frame.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    def settled(self) -> torch.Tensor:
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + latches. The transmission credit accumulates SIGNED
        per-step rack increments taken WHILE the idler is seated: forward steps are
        capped at `adv_step_max`, backward steps subtract IN FULL, and the total is
        floored at 0. A teleport jump credits at most one capped step, and the
        depenetration oscillation it excites cannot ratchet (its forward creep is
        wiped by the full-value backward strokes) — so reaching `trans_min` needs
        sustained NET forward geared motion, which only real cranking produces
        (honest crank-driven steps are ~0.5 mm, far under the cap)."""
        c = self.cfg
        fin = self._finite()
        self.still_count = (self.still_count + 1.0) * (self._still_now() & fin).float()
        seated = self.idler_seated() & fin
        self.seated_ever = torch.maximum(self.seated_ever, seated.float())
        q = self.rack_q()
        dq = torch.minimum(q - self._prev_rack_q, q.new_tensor(c.adv_step_max))
        self.geared_adv = (self.geared_adv + dq * seated.float()).clamp(min=0.0)
        self._prev_rack_q = q
        trans = self.geared_adv >= c.trans_min
        loc_cx = self._local(self.cube)[:, 0]
        self.eject_ever = torch.maximum(
            self.eject_ever, ((loc_cx > c.eject_x) & trans & fin).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: cube settled inside the pocket, everything still and finite,
        AND the transmission latch earned (the cube got there because the geared
        rack pushed it — the tunnel geometry allows no other pusher, and the latch
        refuses teleported or wrenched-rack deliveries)."""
        return (self.cube_in_pocket() & (self.geared_adv >= self.cfg.trans_min)
                & self.settled() & self._finite())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 seated + 0.20 transmission + 0.20 ejected —
        all latched; capped 0.65; exactly 1.0 iff success(). Null policy earns 0."""
        c = self.cfg
        trans = (self.geared_adv >= c.trans_min).float()
        base = (c.w_seat * self.seated_ever + c.w_trans * trans
                + c.w_eject * self.eject_ever).clamp(0.0, 0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# Idempotent registration (forge discovery may import this module twice under
# different names). The scene name is corpus-unique.
if "idler_gearbox_i396" not in SCENES.list():
    SCENES.register("idler_gearbox_i396", IdlerGearboxScene)
from robobench.core.registries import ENVS as _ENVS  # noqa: E402

if "simgen.idler_gearbox_i396" not in _ENVS.list():
    register_env("simgen",
                 lambda: EnvCfg(scene="idler_gearbox_i396", robot="null",
                                env_spacing=3.0))
