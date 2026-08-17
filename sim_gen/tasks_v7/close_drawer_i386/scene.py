"""BayonetDrawerScene — press the self-opening drawer of a nose-down-tilted cabinet
fully shut, then twist its T-handle a quarter turn so the lug bar behind the slotted
bezel BAYONET-LOCKS it closed (sim_gen task `close_drawer_i386`).

Derived from rlbench/close_drawer, but STRATEGICALLY different: the seed's drawer is
a plain level prismatic drawer — one guided push on the drawer front closes it, and
it STAYS closed because nothing reopens it. Here the whole cabinet is pitched
nose-down on its plinth, so gravity is the ADVERSARY: the drawer joint is
near-frictionless and any un-locked drawer glides fully open on its own within a
fraction of a second. Closing is therefore only a MEANS; the goal is a bistable
LATCHED state that survives hands-off. The drawer carries a rotating bayonet rotor
on its front face: a lug bar behind a slotted bezel plate. The bar passes the
horizontal slot only when near-horizontal (within ~13 deg); pressed home it sits
BEHIND the plate, and a ~90 deg twist of the red T-handle turns it vertical so the
released drawer is caught by the plate after ~7 mm. The seed's whole strategy
(push shut, let go) demonstrably scores ~0: the drawer glides right back open.
The rotor also creates a PRE-CONDITION and a TRAP: spawned with a random
misalignment (up to 22 deg), the bar may not pass the slot at all until squared,
and a pre-twisted rotor ARRESTS the press ~25 mm short of closed (both proven in
smoke with real force-limited presses).

Distinct from the sibling variants examined: `close_drawer_i239` re-seats a fallen
FREE-BODY drawer into an open bay (6-DoF retrieval + aperture insertion, no joints,
no locking); `close_drawer_i58` is a never-touch-the-tray interlock-removal task
where gravity CLOSES a self-closing tray. Here gravity OPENS the jointed drawer,
the drawer is manipulated directly, and the crux is operating a two-DoF
press-AND-twist locking mechanism in the forced order square -> press -> twist ->
release.

success(): drawer joint within `closed_tol` of the closed stop AND rotor twisted to
at least `lock_min_deg` AND everything persistently still and finite. The state is
self-persistent: released, the drawer creeps ~7 mm until the vertical lug bar bears
on the bezel back face and rests there (q_rest < closed_tol by construction,
asserted). An under-twisted (< lock_min) or wrong-direction twist may also retain
mechanically but is REFUSED; a pre-twisted press arrests at q_trap >> closed_tol
and can never earn the locked latch (band separation asserted).

score(), latched: 0.15 * max closure fraction inside the last `q_credit` of travel
(null policy earns exactly 0 — spawns start beyond it and glide AWAY) + 0.15 once
ever pressed to `closed_tol` + 0.30 once ever twisted past lock_min WHILE pressed;
capped 0.60; exactly 1.0 iff success().

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - station (DYNAMIC, 40 kg — dynamic so reset teleports keep the spawn-authored
    joint anchors with the body; heavy + grippy so nothing budges it): cabinet
    shell pitched `pitch_deg` nose-down inside its own frame's tilt, standing on a
    world-flat plinth (child box counter-rotated Ry(-pitch), so the resting pose IS
    the pitch, under any yaw); front bezel: two side arms + a 4-box plate with a
    horizontal slot (slot_w x slot_h) centred on the rotor axis.
  - drawer (dynamic): amber box riding a spawn-authored prismatic joint (axis X,
    stops [0, stroke]); station<->drawer collisions are joint-filtered (the joint
    is the runner — nothing rubs).
  - rotor (dynamic): lug bar (bar_span x bar_t) + visual-only round shaft + red
    T-handle crossbar, on a spawn-authored revolute joint to the drawer (axis X,
    limits [-25 deg, +100 deg]). Rotor<->station DOES collide: every load-bearing
    lock/arrest contact is bar-vs-bezel.
Masses/CoM/inertia and friction materials are AUTHORED in the spawners (custom
spawn funcs apply no cfg schemas).

Per-episode randomization (readback-verified in smoke): station yaw +-180 deg and
xy jitter; drawer initial opening q0 in [q0_lo, q0_hi]*stroke (read back before it
glides away); rotor misalignment theta0, random sign, magnitude in
[rot0_min_deg, rot0_max_deg] — straddling the ~13 deg pass angle, so some episodes
require squaring first and some merely reward it.

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


# ----- torch quaternion helpers (module-level, app-free) ---------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(...,4) x (...,4) -> (...,4), wxyz Hamilton product."""
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
    """Rotate vectors v (...,3) by quats q (...,4)."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qapply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _qapply(_qinv(q), v)


def _ry(p: float) -> tuple:
    """wxyz quat for a rotation of p rad about +Y."""
    return (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)


def _ry_apply(p: float, v: tuple) -> tuple:
    """Rotate a 3-vector by Ry(p)."""
    c, s = math.cos(p), math.sin(p)
    return (c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


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


def _friction_material(stage, path: str, static: float, dynamic: float):
    """One USD physics material (explicit binding — the default-material ~0.5 trap)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None, collide: bool = True) -> None:
    """Author one box child prim (translate -> orient -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC heavy cabinet. The body frame is the PITCHED cabinet frame (+x = out
    the front, tilted `pitch` nose-down when the body rests on its plinth): all
    shell geometry is axis-aligned in it; the plinth is counter-rotated Ry(-pitch)
    so its bottom face is world-flat at rest under any yaw. Dynamic (not
    kinematic) so per-episode teleports keep joint anchors with the body."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    com = _ry_apply(-cfg.pitch, (-0.10, 0.0, 0.02 - cfg.base_lift))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.9, 0.9, 0.9))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    az = cfg.axis_z  # rotor axis height in the cabinet frame
    grey = (0.42, 0.46, 0.52)
    dark = (0.22, 0.24, 0.28)
    # --- shell around the drawer cavity (cavity x [-0.21,0], y +-0.083, z [0.10,0.173]) ---
    _box(stage, f"{prim_path}/floor", (0.222, 0.190, 0.012), (-0.111, 0.0, 0.094),
         grey, co, material=mat)
    _box(stage, f"{prim_path}/roof", (0.222, 0.190, 0.012), (-0.111, 0.0, 0.179),
         grey, co, material=mat)
    for s, nm in ((-1.0, "wall_n"), (1.0, "wall_p")):
        _box(stage, f"{prim_path}/{nm}", (0.222, 0.012, 0.097), (-0.111, s * 0.089, 0.1365),
             grey, co, material=mat)
    _box(stage, f"{prim_path}/back", (0.012, 0.190, 0.097), (-0.216, 0.0, 0.1365),
         grey, co, material=mat)
    # --- bezel: two side arms + 4-box slotted plate (slot centred on the rotor axis) ---
    bx = cfg.bezel_back_x + cfg.bezel_t / 2  # plate centre x
    hs_w, hs_h = cfg.slot_w / 2, cfg.slot_h / 2
    for s, nm in ((-1.0, "arm_n"), (1.0, "arm_p")):
        _box(stage, f"{prim_path}/{nm}", (cfg.bezel_back_x + cfg.bezel_t, 0.012, 0.160),
             ((cfg.bezel_back_x + cfg.bezel_t) / 2, s * 0.089, az), dark, co, material=mat)
    _box(stage, f"{prim_path}/bezel_top", (cfg.bezel_t, 0.190, 0.215 - (az + hs_h)),
         (bx, 0.0, (0.215 + az + hs_h) / 2), dark, co, material=mat)
    _box(stage, f"{prim_path}/bezel_bot", (cfg.bezel_t, 0.190, (az - hs_h) - 0.055),
         (bx, 0.0, (az - hs_h + 0.055) / 2), dark, co, material=mat)
    for s, nm in ((-1.0, "bezel_n"), (1.0, "bezel_p")):
        _box(stage, f"{prim_path}/{nm}", (cfg.bezel_t, 0.095 - hs_w, cfg.slot_h),
             (bx, s * (0.095 + hs_w) / 2, az), dark, co, material=mat)
    # --- plinth: counter-rotated so its bottom is world-flat at rest ---
    ph = cfg.base_lift + 0.1165  # top reaches the shell's back underside
    d = (-0.12, 0.0, ph / 2 - cfg.base_lift)  # centre offset in the LEVEL (yaw) frame
    c = _ry_apply(-cfg.pitch, d)
    _box(stage, f"{prim_path}/plinth", (0.24, 0.20, ph), c, (0.30, 0.30, 0.32), co,
         material=mat, orient=_ry(-cfg.pitch))
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The drawer: one amber box (+ front veneer), ONE dynamic rigid body on a
    spawn-authored prismatic joint (axis X, stops [0, stroke]) against the sibling
    Station. The joint is the runner (station<->drawer collision filtered); the
    pitched gravity component is the return spring that GLIDES IT OPEN. Linear
    damping sets the glide terminal speed. Mass/CoM/inertia AUTHORED."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.004, 0.006, 0.008))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(cfg.lin_damping))
    pxrb.CreateAngularDampingAttr(2.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (0.196, 0.160, 0.070), (-0.002, 0.0, 0.0),
         (0.72, 0.50, 0.22), co, material=mat)
    _box(stage, f"{prim_path}/veneer", (0.004, 0.160, 0.070), (0.098, 0.0, 0.0),
         (0.80, 0.56, 0.24), co, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/runner")
    j.CreateBody0Rel().SetTargets([f"{base}/Station"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(cfg.closed_x), 0.0, float(cfg.axis_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(cfg.stroke))
    return root


def _spawn_rotor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The bayonet rotor: steel lug bar + visual-only round shaft + red T-handle
    crossbar, ONE dynamic body on a spawn-authored revolute joint (axis X, limits
    [-back_deg, +fore_deg]) against the sibling Drawer. Body origin ON the axis at
    the drawer front face. The BAR is the only lock-bearing collider (the shaft
    never collides, so nothing square ever turns inside the slot). Rotor<->drawer
    collision is joint-filtered; rotor-vs-station (bar-vs-bezel) is live — that
    contact IS the mechanism. Mass/CoM/inertia AUTHORED."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.025, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0e-4, 2.5e-4, 2.5e-4))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    # lug bar: x [0.001, 0.001+bar_t], long axis y — the ONLY lock collider
    _box(stage, f"{prim_path}/bar", (cfg.bar_t, cfg.bar_span, cfg.bar_t),
         (0.001 + cfg.bar_t / 2, 0.0, 0.0), (0.75, 0.76, 0.80), co, material=mat)
    # round shaft: VISUAL ONLY (rotationally symmetric — and it never collides)
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/shaft")
    cyl.CreateAxisAttr("X")
    cyl.CreateRadiusAttr(float(cfg.shaft_r))
    cyl.CreateHeightAttr(float(cfg.knob_back_x))
    cxf = UsdGeom.Xformable(cyl.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(float(cfg.knob_back_x) / 2, 0.0, 0.0))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.56, 0.60)])
    # red T-handle crossbar: the grasp/press feature
    _box(stage, f"{prim_path}/knob", (cfg.knob_t, cfg.knob_span, cfg.knob_t),
         (cfg.knob_back_x + cfg.knob_t / 2, 0.0, 0.0), (0.85, 0.13, 0.13), co, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Drawer"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.10, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.back_deg))
    j.CreateUpperLimitAttr(float(cfg.fore_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            pitch: float = 0.1309
            base_lift: float = 0.02
            axis_z: float = 0.135
            slot_w: float = 0.112
            slot_h: float = 0.034
            bezel_back_x: float = 0.020
            bezel_t: float = 0.006
            mass: float = 40.0
            mu_static: float = 0.8
            mu_dynamic: float = 0.7
            contact_offset: float = 0.002

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            closed_x: float = -0.10
            axis_z: float = 0.135
            stroke: float = 0.13
            mass: float = 1.5
            lin_damping: float = 6.0
            mu_static: float = 0.4
            mu_dynamic: float = 0.35
            contact_offset: float = 0.002

        @configclass
        class RotorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rotor)
            bar_span: float = 0.100
            bar_t: float = 0.012
            shaft_r: float = 0.014
            knob_back_x: float = 0.050
            knob_t: float = 0.016
            knob_span: float = 0.090
            back_deg: float = 25.0
            fore_deg: float = 100.0
            mass: float = 0.12
            mu_static: float = 0.5
            mu_dynamic: float = 0.4
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(station=StationSpawnerCfg, drawer=DrawerSpawnerCfg,
                              rotor=RotorSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BayonetDrawerSceneCfg(BaseCfg):
    """Config for `BayonetDrawerScene`. `__post_init__` asserts the strategic
    honesty invariants with pre-computed lock geometry: the bar passes the slot
    squared and NEVER passes twisted; the locked rest pose lies INSIDE the closed
    band; the pre-twisted arrest lies far OUTSIDE it (so the trap can never earn
    the locked latch); the spawn misalignment straddles the pass angle; every
    grasp fits a parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    closed_tol: float = tunable(0.012)  # drawer q counted CLOSED (m); locked rest 0.007
    lock_min_deg: float = tunable(70.0)  # rotor twist counted LOCKED (deg); full lock 90
    lock_q_slack: float = tunable(0.002)  # locked latch gate: q <= closed_tol + this
    q_credit: float = tunable(0.05)  # closure credit ramps over the last 5 cm only
    settle_lin: float = tunable(0.05)  # max drawer/station |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.6)  # max drawer/rotor |ang vel| when judging (rad/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_deg: float = tunable(180.0)  # station uniform +- yaw per episode
    xy_jitter: float = tunable(0.05)  # station uniform +- xy per episode (m)
    rot0_min_deg: float = tunable(8.0)  # rotor spawn misalignment magnitude ...
    rot0_max_deg: float = tunable(22.0)  # ... straddles the pass angle (~12.8 deg)
    q0_lo: float = tunable(0.55)  # drawer spawn opening, fraction of stroke ...
    q0_hi: float = tunable(0.95)  # ... (glides to the open stop on its own)

    # --- info: structure ---------------------------------------------------------------------
    pitch_deg: float = info(7.5)  # cabinet nose-down pitch (gravity opens the drawer)
    base_lift: float = info(0.02)  # station origin height above the plinth bottom
    axis_z: float = info(0.135)  # rotor axis height (cabinet frame)
    closed_x: float = info(-0.10)  # drawer centre x at q=0 (cabinet frame)
    stroke: float = info(0.13)  # prismatic hard stops [0, stroke]
    drawer_mass: float = info(1.5)
    lin_damping: float = info(6.0)  # drawer glide terminal speed ~ g sin(pitch)/this
    bar_span: float = info(0.100)  # lug bar long axis (y at 0 twist)
    bar_t: float = info(0.012)  # lug bar thickness; bar x [0.001, 0.013] (rotor frame)
    bar_x0: float = info(0.001)  # bar back face offset in front of the drawer face
    slot_w: float = info(0.112)  # bezel slot width (y)
    slot_h: float = info(0.034)  # bezel slot height (z)
    bezel_back_x: float = info(0.020)  # bezel plate back face x (cabinet frame)
    bezel_t: float = info(0.006)  # bezel plate thickness
    shaft_r: float = info(0.014)  # round shaft radius (visual-only prim)
    knob_back_x: float = info(0.050)  # T-handle crossbar back face (rotor frame)
    knob_t: float = info(0.016)
    knob_span: float = info(0.090)
    back_deg: float = info(25.0)  # rotor limit, wrong direction
    fore_deg: float = info(100.0)  # rotor limit, lock direction (90 = full lock)
    rotor_mass: float = info(0.12)
    station_mass: float = info(40.0)
    mu_static: float = info(0.8)
    mu_dynamic: float = info(0.7)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    pitch: float = field(default=None, init=False)  # rad
    q_rest: float = field(default=None, init=False)  # locked drawer rest q
    q_trap: float = field(default=None, init=False)  # pre-twisted press arrest q
    pass_deg: float = field(default=None, init=False)  # max twist that passes the slot

    def __post_init__(self) -> None:
        self.pitch = math.radians(self.pitch_deg)
        self.q_rest = self.bezel_back_x - (self.bar_x0 + self.bar_t)  # bar front hits plate back
        self.q_trap = (self.bezel_back_x + self.bezel_t) - self.bar_x0  # bar back hits plate front
        # pass angle: bar z-projection span*sin(t) + bar_t*cos(t) fills the slot
        lo, hi = 0.0, math.pi / 2
        for _ in range(60):
            mid = (lo + hi) / 2
            proj = self.bar_span * math.sin(mid) + self.bar_t * math.cos(mid)
            lo, hi = (lo, mid) if proj > self.slot_h else (mid, hi)
        self.pass_deg = math.degrees(lo)

        # -- the bar passes the slot squared, with real margin --
        assert self.bar_span <= self.slot_w - 0.008, "squared bar must pass the slot in y"
        assert self.bar_t <= self.slot_h - 0.016, "squared bar must pass the slot in z"
        # -- and NEVER passes at lock_min: retention overlap is deep --
        lock = math.radians(self.lock_min_deg)
        proj = self.bar_span * math.sin(lock) + self.bar_t * math.cos(lock)
        assert proj >= self.slot_h + 0.010, f"locked bar (proj {proj:.3f}) must catch the plate"
        # -- locked rest INSIDE the closed band; trap arrest far OUTSIDE it --
        assert 0.0 < self.q_rest <= self.closed_tol - 0.004, \
            f"locked rest q={self.q_rest:.4f} must sit inside closed_tol with margin"
        assert self.q_trap >= self.closed_tol + self.lock_q_slack + 0.008, \
            f"pre-twisted arrest q={self.q_trap:.4f} must never reach the locked-latch gate"
        # -- spawn misalignment straddles the pass angle (squaring sometimes REQUIRED) --
        assert self.rot0_max_deg >= self.pass_deg + 3.0, \
            f"max spawn twist must jam the slot (pass angle {self.pass_deg:.1f} deg)"
        assert self.rot0_min_deg <= self.pass_deg - 3.0, \
            "min spawn twist must still pass (variety, not uniform difficulty)"
        assert self.back_deg + 3.0 < self.lock_min_deg, "wrong-direction twist can never lock"
        assert self.lock_min_deg <= 90.0 - 5.0 and self.fore_deg >= 90.0 + 5.0, \
            "the full 90-deg lock must be comfortably reachable inside the stops"
        assert self.fore_deg <= 175.0, "stay clear of the revolute +-180 wrap"
        # -- the round shaft turns freely inside the slot --
        assert self.shaft_r <= self.slot_h / 2 - 0.0025, "shaft must clear the slot in z"
        # -- swing clearance: bar diagonal inside the arms; knob proud of the bezel --
        diag = math.hypot(self.bar_span / 2, self.bar_t / 2)
        assert diag + 0.004 <= 0.083, "bar swing circle must clear the bezel arms"
        finger = self.knob_back_x - (self.bezel_back_x + self.bezel_t)
        assert finger >= 0.020, "T-handle must stand proud of the bezel for a jaw grasp"
        # -- gravity really re-opens: closure credit band is unreachable at spawn --
        assert self.q0_lo * self.stroke >= self.q_credit + 0.01, \
            "spawn openings must start beyond the closure-credit ramp (null earns 0)"
        assert self.pitch_deg >= 5.0, "the pitch must actually drive the glide"
        assert self.stroke >= 8 * self.closed_tol, "closed band must be a small stroke fraction"
        # -- jaw fits (parallel jaw ~80 mm) --
        assert self.knob_t <= 0.075, "T-handle bar must fit the jaw"


# ----- scene -----------------------------------------------------------------------------------
class BayonetDrawerScene(BaseScene):
    cfg: BayonetDrawerSceneCfg

    def __init__(self, cfg: BayonetDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetDrawerSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        # spawn-time chain: station level pose at the origin, drawer at q=0.09, rotor 0 twist
        st_pos = (0.0, 0.0, c.base_lift + 0.002)
        st_quat = _ry(c.pitch)
        dr_off = _ry_apply(c.pitch, (c.closed_x + 0.09, 0.0, c.axis_z))
        ro_off = _ry_apply(c.pitch, (0.09, 0.0, c.axis_z))
        dr_pos = tuple(a + b for a, b in zip(st_pos, dr_off))
        ro_pos = tuple(a + b for a, b in zip(st_pos, ro_off))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_static, dynamic_friction=c.mu_dynamic,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=spawners["station"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.station_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    pitch=c.pitch, base_lift=c.base_lift, axis_z=c.axis_z,
                    slot_w=c.slot_w, slot_h=c.slot_h,
                    bezel_back_x=c.bezel_back_x, bezel_t=c.bezel_t,
                    mass=c.station_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=st_pos, rot=st_quat),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=spawners["drawer"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    closed_x=c.closed_x, axis_z=c.axis_z, stroke=c.stroke,
                    mass=c.drawer_mass, lin_damping=c.lin_damping,
                    mu_static=0.4, mu_dynamic=0.35, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=dr_pos, rot=st_quat),
            ),
            "rotor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rotor",
                spawn=spawners["rotor"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rotor_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    bar_span=c.bar_span, bar_t=c.bar_t, shaft_r=c.shaft_r,
                    knob_back_x=c.knob_back_x, knob_t=c.knob_t, knob_span=c.knob_span,
                    back_deg=c.back_deg, fore_deg=c.fore_deg, mass=c.rotor_mass,
                    mu_static=0.5, mu_dynamic=0.4, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=ro_pos, rot=st_quat),
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
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.station: RigidObject = env.iscene["station"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.rotor: RigidObject = env.iscene["rotor"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.closure_latch = torch.zeros(n, device=dev)  # max closure fraction (last 5 cm)
        self.pressed_latch = torch.zeros(n, device=dev)  # ever pressed to closed_tol
        self.locked_latch = torch.zeros(n, device=dev)  # ever locked while pressed
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the WHOLE LINKAGE (station + drawer + rotor) written
        consistently from one sampled station pose — yaw +-180 deg, xy jitter,
        drawer opening q0, rotor misalignment theta0 (random sign) — then the
        drawer glides open on its own (the dead-man). Latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn (the degenerate-first-draw trap)

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        mag = math.radians(c.rot0_min_deg) + torch.rand(m, device=dev) * (
            math.radians(c.rot0_max_deg) - math.radians(c.rot0_min_deg))
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        th0 = (sign * mag).clamp(math.radians(-c.back_deg + 2.0),
                                 math.radians(c.fore_deg - 2.0))
        q0 = (c.q0_lo + torch.rand(m, device=dev) * (c.q0_hi - c.q0_lo)) * c.stroke

        half = yaw / 2
        qz = torch.stack([torch.cos(half), torch.zeros_like(half),
                          torch.zeros_like(half), torch.sin(half)], dim=-1)
        qy = torch.tensor(_ry(c.pitch), device=dev).expand(m, 4)
        q_st = _qmul(qz, qy)  # station orientation Rz(yaw) * Ry(pitch)
        p_st = torch.zeros(m, 3, device=dev)
        p_st[:, 0:2] = jit
        p_st[:, 2] = c.base_lift + 0.002
        p_st += origin

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = p_st
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = c.closed_x + q0
        off[:, 2] = c.axis_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = p_st + _qapply(q_st, off)
        st[:, 3:7] = q_st
        self.drawer.write_root_state_to_sim(st, env_ids)

        off[:, 0] = q0
        hh = th0 / 2
        qx = torch.stack([torch.cos(hh), torch.sin(hh),
                          torch.zeros_like(hh), torch.zeros_like(hh)], dim=-1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = p_st + _qapply(q_st, off)
        st[:, 3:7] = _qmul(q_st, qx)
        self.rotor.write_root_state_to_sim(st, env_ids)

        for t in (self.closure_latch, self.pressed_latch, self.locked_latch,
                  self.still_count):
            t[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "rotor": self.rotor.data.root_state_w[env_ids].clone(),
            "latches": torch.stack([
                self.closure_latch[env_ids], self.pressed_latch[env_ids],
                self.locked_latch[env_ids], self.still_count[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.rotor.write_root_state_to_sim(state["rotor"], env_ids)
        lat = state["latches"]
        (self.closure_latch[env_ids], self.pressed_latch[env_ids],
         self.locked_latch[env_ids], self.still_count[env_ids]) = (
            lat[:, 0], lat[:, 1], lat[:, 2], lat[:, 3])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A small cabinet stands on a plinth, pitched {c.pitch_deg:.0f} degrees "
            f"NOSE-DOWN (its open front faces you and points slightly at the floor). Its "
            f"single amber drawer rides near-frictionless rails, so gravity slides it "
            f"fully OPEN on its own: push it shut and let go, and it glides back out "
            f"within a second. Its position and heading change every episode.\n"
            f"On the drawer's front face is a BAYONET LOCK: a red T-handle on a round "
            f"shaft, which turns a steel lug bar ({c.bar_span * 100:.0f} cm wide) mounted "
            f"just in front of the drawer face. Fixed to the cabinet, standing proud of "
            f"the drawer, is a dark BEZEL PLATE with a horizontal slot "
            f"({c.slot_w * 1000:.0f} x {c.slot_h * 1000:.0f} mm). The bar fits through the "
            f"slot only when roughly horizontal (within about {c.pass_deg:.0f} degrees); "
            f"the handle spawns misaligned by up to {c.rot0_max_deg:.0f} degrees, so you "
            f"may need to SQUARE it first or the press jams on the plate about "
            f"{c.q_trap * 1000:.0f} mm short of closed.\n"
            f"To lock: square the handle, press the drawer fully shut by the T-handle "
            f"(the bar passes through the slot and sits behind the plate), then twist "
            f"the handle a quarter turn COUNTER-CLOCKWISE as seen from the front (at "
            f"least {c.lock_min_deg:.0f} degrees, a full 90 is natural) so the bar turns "
            f"vertical behind the plate, and let go. The released drawer creeps a few "
            f"millimetres until the bar catches the plate's back — and stays shut. "
            f"Twisting the wrong way or short of {c.lock_min_deg:.0f} degrees does not "
            f"count, even if it happens to snag. Success is judged with everything at "
            f"rest and hands off: drawer within {c.closed_tol * 1000:.0f} mm of fully "
            f"closed, handle twisted at least {c.lock_min_deg:.0f} degrees."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The tilted cabinet's drawer slides itself open. Square the red T-handle, "
            "press the drawer fully shut with it, then twist the handle a quarter turn "
            "counter-clockwise so the bar behind the slotted plate locks the drawer, "
            "and let go. The drawer must stay shut on its own."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def drawer_q(self) -> torch.Tensor:
        """(N,) drawer joint position (0 = closed stop) from pose readback."""
        c = self.cfg
        rel = _qapply_inv(self.station.data.root_quat_w,
                          self.drawer.data.root_pos_w - self.station.data.root_pos_w)
        return rel[:, 0] - c.closed_x

    def rotor_theta(self) -> torch.Tensor:
        """(N,) rotor twist (rad, + = lock direction) about the joint x axis."""
        rel = _qmul(_qinv(self.drawer.data.root_quat_w), self.rotor.data.root_quat_w)
        return 2.0 * torch.atan2(rel[:, 1], rel[:, 0])

    def _finite(self) -> torch.Tensor:
        ok = torch.ones_like(self.still_count, dtype=torch.bool)
        for body in (self.station, self.drawer, self.rotor):
            ok &= torch.isfinite(body.data.root_state_w).all(dim=-1)
        return ok

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        return ((self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.rotor.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.station.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + progress latches: max closure fraction over the last
        `q_credit` of travel (spawns start beyond it and glide away — null earns
        exactly 0); ever pressed to `closed_tol`; ever twisted past lock_min WHILE
        pressed (the pre-twisted trap arrests at q_trap, outside this gate by
        construction)."""
        c = self.cfg
        fin = self._finite()
        self.still_count = (self.still_count + 1.0) * (self._still_now() & fin).float()
        q = self.drawer_q()
        th = self.rotor_theta()
        frac = ((c.q_credit - q) / c.q_credit).clamp(0.0, 1.0) * fin.float()
        self.closure_latch = torch.maximum(self.closure_latch, frac)
        self.pressed_latch = torch.maximum(
            self.pressed_latch, ((q <= c.closed_tol) & fin).float())
        locked_now = ((th >= math.radians(c.lock_min_deg))
                      & (q <= c.closed_tol + c.lock_q_slack) & fin)
        self.locked_latch = torch.maximum(self.locked_latch, locked_now.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: drawer within closed_tol of the closed stop, rotor twisted to
        at least lock_min, everything persistently still and finite. Self-persistent
        hands-off ONLY via the bayonet: un-locked, the drawer glides open and the
        stillness gate never holds with q inside the band."""
        c = self.cfg
        return ((self.drawer_q() <= c.closed_tol)
                & (self.rotor_theta() >= math.radians(c.lock_min_deg))
                & self.settled() & self._finite())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * latched closure fraction + 0.15 pressed +
        0.30 locked-while-pressed — all latched, credit never evaporates; capped
        0.60; exactly 1.0 iff success(). Null policy earns exactly 0."""
        base = (0.15 * self.closure_latch + 0.15 * self.pressed_latch
                + 0.30 * self.locked_latch).clamp(0.0, 0.60)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# Idempotent registration: on the forge, discovery may import this module under a
# different module name before the solve/smoke package-relative import re-executes it.
# NOTE: the name must be corpus-unique ("bayonet_drawer" is taken by
# libero_kitchen_scene1_open_bottom_drawer_i88, whose cfg has different fields).
if "twistlock_drawer_i386" not in SCENES.list():
    SCENES.register("twistlock_drawer_i386", BayonetDrawerScene)
from robobench.core.registries import ENVS as _ENVS  # noqa: E402

if "simgen.twistlock_drawer_i386" not in _ENVS.list():
    register_env("simgen",
                 lambda: EnvCfg(scene="twistlock_drawer_i386", robot="null",
                                env_spacing=3.0))
