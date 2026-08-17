"""PawlLadderScene — meter a gravity-driven rack cart down a pitched rail, one
notch at a time, by lifting and releasing a spring-of-gravity pawl; park it at the
GREEN post without ever coasting past it (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i431`).

Derived from libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet,
but STRATEGICALLY different: the seed grasps a drawer handle and pulls a prismatic
joint monotonically past a threshold — one guided pull, any speed, any stopping
point past -0.1 rad. Here the prismatic DOF is self-propelled (gravity on an 8-deg
ramp) and the manipulated handle is NOT on the sliding body at all: the only thing
the hand ever operates is a vertical PAWL above the rail. The cart's rack carries
six notches; the dropped pawl blade is a TWO-WAY detent (vertical walls both
sides), so the cart can neither advance nor be dragged back while seated — pulling
or pushing the cart directly, the seed's whole strategy, moves it a few millimetres
and scores 0 (proven in smoke with a real shove). Progress is quantized: lift the
pawl -> the cart accelerates away -> release EARLY so the blade rides the next land
and drops into the next notch, arresting the cart one station further. The episode
commands a target station via a visible GREEN post (2..4 notches away, resampled
per episode); a RED post marks the next notch beyond it. Coasting more than
`overshoot` past the target rest fires a PERMANENT foul latch (score 0): with one
arm there is no way back uphill, because moving the seated cart uphill would
require lifting the pawl WHILE pushing the cart — two hands. A loose ball rides in
a shallow tray on the cart and must still be aboard at judging, so slam-and-crash
metering is refused.

Distinct from every examined ratchet/escapement task in the corpus:
`open_the_middle_drawer_i34` (gumball_meter) counts DISPENSED FREE OBJECTS through
a shuttle airlock (pull+push strokes, credit = ball count); `open_window_i283`
props a dead-man sash with a ONE-WAY ratchet (a single retention event);
`butter_i307` is a vertical gravity-press with a one-way pawl. Here nothing is
dispensed and nothing is propped: the pawl is a TWO-WAY detent used k times as a
release-timing regulator to position ONE carriage at a commanded spatial station,
with an irreversible overshoot foul and a fragile rider. Also distinct from
sibling i279 (sealed hopper, tilt the housing) and i74 (fold a hinged screen), and
from close_drawer_i386 (press-and-twist bayonet lock on the sliding body itself).

success(): no foul latched AND cart within `q_tol` of the target notch rest AND
pawl seated (blade dropped into the notch, pawl joint within `seat_tol` of its
lower stop — a blade resting on a land sits ~10 mm higher and is refused) AND the
ball inside the tray (cart-frame box test) AND everything persistently still and
finite. The state is self-persistent: the seated blade arrests the cart against
ramp gravity hands-off.

score(), latched: 0 forever once fouled; else 0.10 once the pawl was ever lifted
clear + 0.50 * (distinct stations 1..k ever arrived-at seated-and-slow)/k; capped
0.60; exactly 1.0 iff success(). Null policy earns exactly 0 (spawn arrest at
station 0 is not credited).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - station (DYNAMIC, 40 kg — dynamic so reset teleports keep spawn-authored joint
    anchors with the body): rail bed + side fences pitched `pitch_deg` nose-down
    (+x downhill) in the body frame, standing on a counter-rotated world-flat
    plinth; a gantry post + arm carries the pawl guide visually.
  - cart (dynamic, 0.8 kg): deck + 7 raised rack lands forming 6 notches (pitch
    `notch_pitch`, gap `notch_gap`) + shallow ball tray + white pointer fin, on a
    spawn-authored prismatic joint (axis X, stops [q_lo, q_hi]); station<->cart
    collisions joint-filtered (the joint is the runner). Linear damping sets the
    downhill terminal speed — the release-timing margin is ASSERTED in cfg.
  - pawl (dynamic, 0.15 kg): blade + stem + red T-handle crossbar on a
    spawn-authored VERTICAL prismatic joint to the station (axis Z, stops
    [0, pawl_travel]); pawl<->station joint-filtered, pawl<->cart LIVE — the
    blade-vs-rack contact IS the mechanism. Seated, the blade hangs 4 mm above the
    notch floor on its own joint stop (no rubbing load on the cart).
  - ball (free sphere) riding the tray; green/red KINEMATIC marker posts standing
    on the ground, teleported per episode to the fin-alignment line of the target
    and the first-forbidden notch.
Masses/CoM/inertia and friction materials are AUTHORED in the spawners (custom
spawn funcs apply no cfg schemas).

Per-episode randomization (readback-verified in smoke): station yaw +-180 deg and
xy jitter; target station k in {k_min..k_max} (post positions are the readback);
cart spawn q0 in [0, q0_max] uphill of the station-0 rest (it glides a few mm into
the pawl — the dead-man demo); ball xy jitter in the tray.

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

G = 9.81


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


def _rb_common(root, mass_kg: float, com, inertia, lin_damp: float, ang_damp: float) -> None:
    """Rigid-body + authored mass/CoM/inertia + PhysX solver settings on one prim."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(mass_kg))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC heavy ramp station. Body frame = PITCHED frame (+x downhill along
    the rail when resting on the counter-rotated plinth, under any yaw)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    com = _ry_apply(-cfg.pitch, (0.0, 0.0, 0.04 - cfg.base_lift))
    _rb_common(root, cfg.mass, com, (1.2, 1.2, 1.2), 0.5, 2.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    grey = (0.42, 0.46, 0.52)
    dark = (0.24, 0.26, 0.30)
    # rail bed (top 5 mm below the cart deck bottom; cart<->station is filtered anyway)
    _box(stage, f"{prim_path}/bed", (0.60, 0.16, 0.016), (0.0, 0.0, 0.112),
         grey, co, material=mat)
    for s, nm in ((-1.0, "fence_n"), (1.0, "fence_p")):
        _box(stage, f"{prim_path}/{nm}", (0.60, 0.008, 0.030), (0.0, s * 0.082, 0.135),
             grey, co, material=mat)
    # pawl gantry (visual anchor for the vertical pawl guide; pawl<->station filtered)
    _box(stage, f"{prim_path}/gantry_post", (0.030, 0.030, 0.325),
         (cfg.pawl_x, 0.115, 0.1725), dark, co, material=mat)
    _box(stage, f"{prim_path}/gantry_arm", (0.030, 0.095, 0.020),
         (cfg.pawl_x, 0.0825, 0.345), dark, co, material=mat)
    # plinth: counter-rotated so its bottom face is world-flat at rest
    ph = cfg.base_lift + 0.104
    d = (0.0, 0.0, ph / 2 - cfg.base_lift)
    c = _ry_apply(-cfg.pitch, d)
    _box(stage, f"{prim_path}/plinth", (0.50, 0.20, ph), c, (0.30, 0.30, 0.32), co,
         material=mat, orient=_ry(-cfg.pitch))
    return root


def _spawn_cart(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The rack cart: deck + 7 rack lands (6 notches) + ball tray + white pointer
    fin, ONE dynamic body on a spawn-authored prismatic joint (axis X, stops
    [q_lo, q_hi]) against the sibling Station. Deck centre at cart-local
    (deck_cx, 0, 0), deck top at local z = +0.010. Notch i is the gap centred at
    cart-local x = -i*notch_pitch."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rb_common(root, cfg.mass, (cfg.deck_cx, 0.0, 0.004), (0.002, 0.008, 0.010),
               cfg.lin_damping, 2.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    amber = (0.72, 0.50, 0.22)
    n, np_, ng = cfg.n_notch, cfg.notch_pitch, cfg.notch_gap
    land_len = np_ - ng
    _box(stage, f"{prim_path}/deck", (0.320, 0.150, 0.020), (cfg.deck_cx, 0.0, 0.0),
         amber, co, material=mat)
    # rack lands: downhill end land, 5 between-lands, uphill end land (z on deck top)
    lz = 0.010 + cfg.land_h / 2
    _box(stage, f"{prim_path}/land_end_d", (land_len, 0.030, cfg.land_h),
         (ng / 2 + land_len / 2, cfg.rack_y, lz), (0.55, 0.42, 0.20), co, material=mat)
    for i in range(n - 1):
        _box(stage, f"{prim_path}/land_{i}", (land_len, 0.030, cfg.land_h),
             (-i * np_ - np_ / 2, cfg.rack_y, lz), (0.55, 0.42, 0.20), co, material=mat)
    _box(stage, f"{prim_path}/land_end_u", (land_len, 0.030, cfg.land_h),
         (-(n - 1) * np_ - ng / 2 - land_len / 2, cfg.rack_y, lz),
         (0.55, 0.42, 0.20), co, material=mat)
    # ball tray: 4 walls on the deck (floor = the deck itself)
    ih, wt, wh = cfg.tray_inner / 2, cfg.wall_t, cfg.wall_h
    wz = 0.010 + wh / 2
    for s, nm in ((-1.0, "tray_xu"), (1.0, "tray_xd")):
        _box(stage, f"{prim_path}/{nm}", (wt, cfg.tray_inner + 2 * wt, wh),
             (cfg.tray_x + s * (ih + wt / 2), cfg.tray_y, wz), (0.20, 0.35, 0.60),
             co, material=mat)
    for s, nm in ((-1.0, "tray_yn"), (1.0, "tray_yp")):
        _box(stage, f"{prim_path}/{nm}", (cfg.tray_inner, wt, wh),
             (cfg.tray_x, cfg.tray_y + s * (ih + wt / 2), wz), (0.20, 0.35, 0.60),
             co, material=mat)
    # white pointer fin (overhangs the deck edge toward the marker posts)
    _box(stage, f"{prim_path}/fin", (0.008, 0.030, 0.030),
         (cfg.fin_x, cfg.fin_y, 0.025), (0.95, 0.95, 0.95), co, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/runner")
    j.CreateBody0Rel().SetTargets([f"{base}/Station"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(cfg.cart0_x), 0.0, float(cfg.axis_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(cfg.q_lo))
    j.CreateUpperLimitAttr(float(cfg.q_hi))
    return root


def _spawn_pawl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The pawl: blade + stem + red T-handle crossbar, ONE dynamic body on a
    spawn-authored VERTICAL prismatic joint (axis Z, stops [0, travel]) against
    the sibling Station. Body origin at the blade centre when seated. Pawl<->cart
    collision is LIVE (the detent contact); pawl<->station is joint-filtered."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rb_common(root, cfg.mass, (0.0, 0.0, 0.035), (4.0e-4, 4.0e-4, 1.0e-4), 0.2, 1.0)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    # blade: the ONLY detent collider (vertical faces both sides = two-way)
    _box(stage, f"{prim_path}/blade", (cfg.blade_t, cfg.blade_w, cfg.blade_h),
         (0.0, 0.0, 0.0), (0.75, 0.76, 0.80), co, material=mat)
    _box(stage, f"{prim_path}/stem", (0.012, 0.012, 0.070), (0.0, 0.0, 0.065),
         (0.55, 0.56, 0.60), co, material=mat)
    _box(stage, f"{prim_path}/handle", (0.016, 0.090, 0.016), (0.0, 0.0, 0.108),
         (0.85, 0.13, 0.13), co, material=mat)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/guide")
    j.CreateBody0Rel().SetTargets([f"{base}/Station"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(False)
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(cfg.pawl_x), float(cfg.pawl_y), float(cfg.pawl_z0)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(cfg.travel))
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
            pitch: float = 0.1396
            base_lift: float = 0.02
            pawl_x: float = -0.010
            mass: float = 40.0
            mu_static: float = 0.8
            mu_dynamic: float = 0.7
            contact_offset: float = 0.0015

        @configclass
        class CartSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cart)
            cart0_x: float = -0.010
            axis_z: float = 0.135
            q_lo: float = -0.004
            q_hi: float = 0.270
            deck_cx: float = -0.125
            n_notch: int = 6
            notch_pitch: float = 0.050
            notch_gap: float = 0.030
            land_h: float = 0.014
            rack_y: float = 0.045
            tray_x: float = -0.125
            tray_y: float = -0.032
            tray_inner: float = 0.060
            wall_t: float = 0.008
            wall_h: float = 0.020
            fin_x: float = -0.125
            fin_y: float = -0.090
            mass: float = 0.8
            lin_damping: float = 7.0
            mu_static: float = 0.3
            mu_dynamic: float = 0.25
            contact_offset: float = 0.0015

        @configclass
        class PawlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pawl)
            pawl_x: float = -0.010
            pawl_y: float = 0.045
            pawl_z0: float = 0.179
            travel: float = 0.030
            blade_t: float = 0.016
            blade_w: float = 0.026
            blade_h: float = 0.060
            mass: float = 0.15
            mu_static: float = 0.3
            mu_dynamic: float = 0.25
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(station=StationSpawnerCfg, cart=CartSpawnerCfg,
                              pawl=PawlSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PawlLadderSceneCfg(BaseCfg):
    """Config for `PawlLadderScene`. `__post_init__` asserts the strategic honesty
    invariants with pre-computed mechanism physics: the release-timing race (drop
    into the passing notch) has real margin at the damped terminal speed; the foul
    threshold sits strictly between the target rest and the first q where the next
    notch could capture; the seated blade hangs clear of the notch floor; the
    seated-pawl gate cannot be met by a blade resting on a land; the ball stays
    trayed through every legitimate arrest; the T-handle fits a parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    q_tol: float = tunable(0.012)  # |q - q_rest(k)| counted AT the target station (m)
    seat_tol: float = tunable(0.004)  # pawl joint z counted SEATED (in-notch) (m)
    lift_min: float = tunable(0.014)  # pawl z counted LIFTED-clear for credit (m)
    overshoot: float = tunable(0.030)  # q past target rest that fires the FOUL (m)
    settle_lin: float = tunable(0.05)  # max cart/pawl/station |lin vel| judging (m/s)
    settle_ang: float = tunable(0.6)  # max cart |ang vel| when judging (rad/s)
    ball_vmax: float = tunable(0.10)  # max ball |lin vel| when judging (m/s)
    settle_steps_min: int = tunable(30)  # stillness must PERSIST this many steps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    yaw_deg: float = tunable(180.0)  # station uniform +- yaw per episode
    xy_jitter: float = tunable(0.05)  # station uniform +- xy per episode (m)
    k_min: int = tunable(2)  # target station lower bound (notches to advance)
    k_max: int = tunable(4)  # target station upper bound
    q0_max: float = tunable(0.004)  # cart spawn q uniform in [0, this] (m)
    ball_jitter: float = tunable(0.010)  # ball spawn xy jitter inside the tray (m)

    # --- info: structure ---------------------------------------------------------------------
    pitch_deg: float = info(8.0)  # ramp pitch, +x downhill (gravity drives the cart)
    base_lift: float = info(0.02)  # station origin height above the plinth bottom
    axis_z: float = info(0.135)  # cart joint height (station frame)
    cart0_x: float = info(-0.010)  # cart origin station-x at q=0
    pawl_x: float = info(-0.010)  # pawl axis station-x (over notch 0 at q=0)
    pawl_y: float = info(0.045)  # pawl axis station-y (over the rack strip)
    pawl_z0: float = info(0.179)  # pawl body origin z when seated (station frame)
    pawl_travel: float = info(0.030)  # pawl joint stops [0, travel]
    q_lo: float = info(-0.004)  # cart joint stops ...
    q_hi: float = info(0.270)
    n_notch: int = info(6)  # notches 0..5; spawn seats notch 0
    notch_pitch: float = info(0.050)  # notch spacing along the rack (m)
    notch_gap: float = info(0.030)  # notch opening in x (m)
    land_h: float = info(0.014)  # rack land height above the deck (m)
    blade_t: float = info(0.016)  # pawl blade x thickness (m)
    blade_w: float = info(0.026)  # pawl blade y width (m)
    blade_h: float = info(0.060)  # pawl blade z height (m)
    blade_clear: float = info(0.004)  # seated blade bottom above the notch floor (m)
    deck_cx: float = info(-0.125)  # deck centre cart-local x
    deck_top: float = info(0.145)  # deck top surface z (station frame)
    tray_x: float = info(-0.125)  # tray centre cart-local x
    tray_y: float = info(-0.032)  # tray centre cart-local y
    tray_inner: float = info(0.060)  # tray inner square side (m)
    wall_t: float = info(0.008)
    wall_h: float = info(0.020)  # tray wall height above the deck (m)
    fin_x: float = info(-0.125)  # pointer fin cart-local x (the alignment line)
    fin_y: float = info(-0.090)  # pointer fin cart-local y (overhangs the deck)
    post_y: float = info(-0.125)  # marker-post line station-frame y
    post_wh: float = info(0.02)  # marker post cross-section (m)
    post_h: float = info(0.20)  # marker post height (m)
    cart_mass: float = info(0.8)
    cart_damping: float = info(7.0)  # terminal speed = g sin(pitch)/this
    pawl_mass: float = info(0.15)
    station_mass: float = info(40.0)
    ball_r: float = info(0.011)
    ball_mass: float = info(0.02)
    mu_static: float = info(0.8)
    mu_dynamic: float = info(0.7)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    pitch: float = field(default=None, init=False)  # rad
    rest_off: float = field(default=None, init=False)  # q_rest(i) = i*pitch + this
    v_term: float = field(default=None, init=False)  # cart damped terminal speed
    lift_clear: float = field(default=None, init=False)  # pawl z where blade clears lands

    def __post_init__(self) -> None:
        self.pitch = math.radians(self.pitch_deg)
        self.rest_off = (self.notch_gap - self.blade_t) / 2  # 0.007
        self.v_term = G * math.sin(self.pitch) / self.cart_damping
        # blade bottom = deck_top + blade_clear + q_pawl; clears land top when
        # q_pawl > land_h - blade_clear
        self.lift_clear = self.land_h - self.blade_clear  # 0.010

        # -- the detent geometry: blade fits the gap with drop play both sides --
        assert self.blade_t + 0.010 <= self.notch_gap, "blade must drop into the gap freely"
        land_len = self.notch_pitch - self.notch_gap
        assert land_len >= 0.016, "lands must be long enough to ride"
        # -- seated blade hangs on its own joint stop, engaged deep in the walls --
        assert 0.002 <= self.blade_clear <= 0.006, "seated blade floats just off the floor"
        engage = self.land_h - self.blade_clear
        assert engage >= 0.008, f"seated engagement {engage:.3f} must be solid"
        assert self.seat_tol <= self.lift_clear - 0.004, \
            "a blade resting ON a land (z=lift_clear) must FAIL the seated gate"
        assert self.lift_min >= self.lift_clear + 0.002, "lift credit = genuinely clear"
        assert self.pawl_travel >= self.lift_min + 0.010, "travel leaves lift headroom"
        # -- release-timing race: drop-to-catch beats the passing gap, with margin --
        window = (self.notch_gap - self.blade_t) / self.v_term  # full-footprint transit
        t_drop = math.sqrt(2 * (self.land_h - 0.005) / (G * math.cos(self.pitch)))
        assert window >= 1.5 * t_drop, \
            f"catch window {window * 1e3:.0f}ms must beat the {t_drop * 1e3:.0f}ms drop"
        # -- foul strictly between target rest and any next-notch capture --
        play = self.notch_gap - self.blade_t  # in-notch q play, 0.014
        assert self.overshoot >= self.q_tol + 0.010, "target band well clear of the foul"
        assert self.overshoot < self.notch_pitch - play - 0.004, \
            "the foul must fire BEFORE the next notch could capture the cart"
        # -- station bands separated; spawn glides INTO the station-0 rest --
        assert self.q_tol <= self.notch_pitch / 2 - 0.010, "station bands must separate"
        assert self.q_tol >= self.rest_off + 0.003, "band covers the gravity rest + jitter"
        assert self.q0_max < self.rest_off, "spawn sits uphill of the station-0 rest"
        assert self.q_lo <= -0.002 and self.q_hi >= (self.n_notch - 1) * self.notch_pitch \
            + self.rest_off + 0.008, "joint stops must bracket all stations"
        assert self.k_max + 1 <= self.n_notch - 1, "the red post must mark a real notch"
        assert self.k_min >= 2, "at least two metered releases per episode"
        # -- ball is retained through every legitimate arrest (energy bound) --
        climb = self.v_term ** 2 / (2 * G)
        assert self.wall_h >= 3 * climb + 0.008, \
            f"tray wall {self.wall_h:.3f} must beat the {climb * 1e3:.1f}mm arrest climb"
        assert self.tray_inner / 2 >= self.ball_r + self.ball_jitter + 0.004, "ball fits"
        assert self.wall_h > self.ball_r, "wall taller than the contact point radius"
        # -- static detent load far below the GPU light-body creep threshold --
        assert self.cart_mass * G * math.sin(self.pitch) <= 4.0, "detent load << ~9N creep"
        # -- marker posts clear the cart sweep (fin tip vs post inner face) --
        fin_tip = self.fin_y - 0.015  # fin y half-width 0.015
        assert abs(self.post_y) - self.post_wh / 2 >= abs(fin_tip) + 0.008, \
            "posts must stand clear of the swept fin"
        # -- the T-handle fits a parallel jaw --
        assert 0.016 <= 0.075, "handle bar must fit the jaw"
        assert self.pitch_deg >= 5.0, "the pitch must genuinely drive the cart"

    def q_rest_i(self, i: int) -> float:
        return i * self.notch_pitch + self.rest_off


# ----- scene -----------------------------------------------------------------------------------
class PawlLadderScene(BaseScene):
    cfg: PawlLadderSceneCfg

    def __init__(self, cfg: PawlLadderSceneCfg | None = None) -> None:
        super().__init__(cfg or PawlLadderSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        st_pos = (0.0, 0.0, c.base_lift + 0.002)
        st_quat = _ry(c.pitch)
        ca_off = _ry_apply(c.pitch, (c.cart0_x + 0.002, 0.0, c.axis_z))
        pw_off = _ry_apply(c.pitch, (c.pawl_x, c.pawl_y, c.pawl_z0))
        bl_off = _ry_apply(c.pitch, (c.cart0_x + 0.002 + c.tray_x, c.tray_y,
                                     c.deck_top + c.ball_r + 0.002))
        ca_pos = tuple(a + b for a, b in zip(st_pos, ca_off))
        pw_pos = tuple(a + b for a, b in zip(st_pos, pw_off))
        bl_pos = tuple(a + b for a, b in zip(st_pos, bl_off))
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5,
                                             restitution=0.0)
        post_spawn = dict(
            size=(c.post_wh, c.post_wh, c.post_h),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=mat,
        )
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
                    pitch=c.pitch, base_lift=c.base_lift, pawl_x=c.pawl_x,
                    mass=c.station_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=st_pos, rot=st_quat),
            ),
            "cart": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cart",
                spawn=spawners["cart"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cart_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    cart0_x=c.cart0_x, axis_z=c.axis_z, q_lo=c.q_lo, q_hi=c.q_hi,
                    deck_cx=c.deck_cx, n_notch=c.n_notch, notch_pitch=c.notch_pitch,
                    notch_gap=c.notch_gap, land_h=c.land_h, rack_y=c.pawl_y,
                    tray_x=c.tray_x, tray_y=c.tray_y, tray_inner=c.tray_inner,
                    wall_t=c.wall_t, wall_h=c.wall_h, fin_x=c.fin_x, fin_y=c.fin_y,
                    mass=c.cart_mass, lin_damping=c.cart_damping,
                    mu_static=0.3, mu_dynamic=0.25, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=ca_pos, rot=st_quat),
            ),
            "pawl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pawl",
                spawn=spawners["pawl"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pawl_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    pawl_x=c.pawl_x, pawl_y=c.pawl_y, pawl_z0=c.pawl_z0,
                    travel=c.pawl_travel, blade_t=c.blade_t, blade_w=c.blade_w,
                    blade_h=c.blade_h, mass=c.pawl_mass,
                    mu_static=0.3, mu_dynamic=0.25, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pw_pos, rot=st_quat),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, linear_damping=0.1,
                        angular_damping=0.1),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.95, 0.75, 0.10)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=bl_pos),
            ),
            "post_g": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PostG",
                spawn=sim_utils.CuboidCfg(
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.75, 0.20)), **post_spawn),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.5, c.post_h / 2)),
            ),
            "post_r": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PostR",
                spawn=sim_utils.CuboidCfg(
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.12, 0.12)), **post_spawn),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, -0.5, c.post_h / 2)),
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
        self.cart: RigidObject = env.iscene["cart"]
        self.pawl: RigidObject = env.iscene["pawl"]
        self.ball: RigidObject = env.iscene["ball"]
        self.post_g: RigidObject = env.iscene["post_g"]
        self.post_r: RigidObject = env.iscene["post_r"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.k = torch.full((n,), c.k_min, device=dev, dtype=torch.long)
        self.rests = torch.tensor([c.q_rest_i(i) for i in range(c.n_notch)], device=dev)
        self.lift_latch = torch.zeros(n, device=dev)
        self.arrive_latch = torch.zeros(n, c.n_notch, device=dev)
        self.foul_latch = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the WHOLE RIG (station + cart + pawl + ball) written
        consistently from one sampled station pose — yaw +-180 deg, xy jitter,
        cart spawn q0 just uphill of the station-0 rest (it glides into the seated
        pawl: the dead-man demo), ball jittered in the tray — and the two marker
        posts teleported to the fin-alignment lines of the freshly sampled target
        station k and the first-forbidden notch k+1. Latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 2, device=dev)  # burn (the degenerate-first-draw trap)

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        q0 = torch.rand(m, device=dev) * c.q0_max
        bj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.ball_jitter
        kk = torch.randint(c.k_min, c.k_max + 1, (m,), device=dev)
        self.k[env_ids] = kk

        half = yaw / 2
        qz = torch.stack([torch.cos(half), torch.zeros_like(half),
                          torch.zeros_like(half), torch.sin(half)], dim=-1)
        qy = torch.tensor(_ry(c.pitch), device=dev).expand(m, 4)
        q_st = _qmul(qz, qy)  # station orientation Rz(yaw) * Ry(pitch)
        p_st = torch.zeros(m, 3, device=dev)
        p_st[:, 0:2] = jit
        p_st[:, 2] = c.base_lift + 0.002
        p_st += origin

        def write(body, off, quat) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = p_st + _qapply(q_st, off)
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = p_st
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = c.cart0_x + q0
        off[:, 2] = c.axis_z
        write(self.cart, off, q_st)

        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = c.pawl_x
        off[:, 1] = c.pawl_y
        off[:, 2] = c.pawl_z0
        write(self.pawl, off, q_st)

        off = torch.zeros(m, 3, device=dev)
        off[:, 0] = c.cart0_x + q0 + c.tray_x + bj[:, 0]
        off[:, 1] = c.tray_y + bj[:, 1]
        off[:, 2] = c.deck_top + c.ball_r + 0.002
        iq = torch.zeros(m, 4, device=dev)
        iq[:, 0] = 1.0
        write(self.ball, off, iq)

        # marker posts: ground-standing at the fin-alignment world xy of station k / k+1
        for body, idx in ((self.post_g, kk), (self.post_r, kk + 1)):
            off = torch.zeros(m, 3, device=dev)
            off[:, 0] = c.cart0_x + self.rests[idx] + c.fin_x
            off[:, 1] = c.post_y
            off[:, 2] = 0.15
            w = p_st + _qapply(q_st, off)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = w[:, 0]
            st[:, 1] = w[:, 1]
            st[:, 2] = c.post_h / 2
            st[:, 3:7] = qz
            body.write_root_state_to_sim(st, env_ids)

        self.lift_latch[env_ids] = 0.0
        self.arrive_latch[env_ids] = 0.0
        self.foul_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "cart": self.cart.data.root_state_w[env_ids].clone(),
            "pawl": self.pawl.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "post_g": self.post_g.data.root_state_w[env_ids].clone(),
            "post_r": self.post_r.data.root_state_w[env_ids].clone(),
            "k": self.k[env_ids].clone(),
            "arrive": self.arrive_latch[env_ids].clone(),
            "latches": torch.stack([
                self.lift_latch[env_ids], self.foul_latch[env_ids],
                self.still_count[env_ids]], dim=-1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.cart.write_root_state_to_sim(state["cart"], env_ids)
        self.pawl.write_root_state_to_sim(state["pawl"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.post_g.write_root_state_to_sim(state["post_g"], env_ids)
        self.post_r.write_root_state_to_sim(state["post_r"], env_ids)
        self.k[env_ids] = state["k"]
        self.arrive_latch[env_ids] = state["arrive"]
        lat = state["latches"]
        (self.lift_latch[env_ids], self.foul_latch[env_ids],
         self.still_count[env_ids]) = (lat[:, 0], lat[:, 1], lat[:, 2])

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A rail ramp stands on a plinth, pitched {c.pitch_deg:.0f} degrees so its "
            f"rail runs gently DOWNHILL; its position and heading change every episode. "
            f"An amber CART rides the rail on near-frictionless runners: unrestrained, "
            f"gravity rolls it steadily downhill. Along the cart's top runs a RACK of "
            f"raised lands with {c.n_notch} square notches, one every "
            f"{c.notch_pitch * 100:.0f} cm. Above the rack, a gantry guides a steel "
            f"PAWL with a red T-handle that slides only VERTICALLY "
            f"({c.pawl_travel * 1000:.0f} mm of travel). Dropped, its blade sits inside "
            f"a notch between two vertical walls, so the cart can move neither forward "
            f"nor back — pushing or pulling the cart directly just rattles it a few "
            f"millimetres. Lift the handle about {c.lift_min * 1000:.0f} mm and the "
            f"blade clears the rack: the cart immediately accelerates away downhill.\n"
            f"The cart carries a small yellow BALL in a shallow walled tray, and a "
            f"white POINTER FIN on its side. Two posts stand on the floor beside the "
            f"rail: a GREEN post marking the commanded stop (2 to 4 notches downhill — "
            f"it moves every episode) and a RED post marking the very next notch, which "
            f"is FORBIDDEN. To advance exactly one notch: lift the pawl, let the cart "
            f"run about 2 cm, and RELEASE EARLY — the blade lands on the passing rack "
            f"land, rides it, and drops into the next notch, arresting the cart. Repeat "
            f"until the fin lines up with the green post and the pawl is seated in the "
            f"notch. If the cart ever coasts more than {c.overshoot * 1000:.0f} mm past "
            f"the green-post rest — toward the red post — the episode is PERMANENTLY "
            f"failed: with the pawl blocking both directions there is no way back "
            f"uphill. Success is judged hands-off, at rest: no overshoot ever, fin at "
            f"the green post (cart within {c.q_tol * 1000:.0f} mm of that notch rest), "
            f"pawl fully dropped into the notch (a blade parked ON a land does not "
            f"count), and the ball still inside the tray."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift and release the pawl's red T-handle to let the rack cart roll down "
            "the ramp one notch at a time. Park the cart with its white fin at the "
            "GREEN post, pawl dropped into the notch, and never coast past it toward "
            "the red post. Keep the yellow ball in the cart's tray."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def cart_q(self) -> torch.Tensor:
        """(N,) cart joint position (0 = notch-0 alignment) from pose readback."""
        rel = _qapply_inv(self.station.data.root_quat_w,
                          self.cart.data.root_pos_w - self.station.data.root_pos_w)
        return rel[:, 0] - self.cfg.cart0_x

    def pawl_q(self) -> torch.Tensor:
        """(N,) pawl joint position (0 = seated stop) from pose readback."""
        rel = _qapply_inv(self.station.data.root_quat_w,
                          self.pawl.data.root_pos_w - self.station.data.root_pos_w)
        return rel[:, 2] - self.cfg.pawl_z0

    def q_target(self) -> torch.Tensor:
        """(N,) the target-station rest position q_rest(k)."""
        return self.rests[self.k]

    def ball_in_tray(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the tray box, in the CART frame."""
        c = self.cfg
        rel = _qapply_inv(self.cart.data.root_quat_w,
                          self.ball.data.root_pos_w - self.cart.data.root_pos_w)
        return ((rel[:, 0] - c.tray_x).abs() <= c.tray_inner / 2 - c.ball_r + 0.006) \
            & ((rel[:, 1] - c.tray_y).abs() <= c.tray_inner / 2 - c.ball_r + 0.006) \
            & (rel[:, 2] >= 0.012) & (rel[:, 2] <= 0.060)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones_like(self.still_count, dtype=torch.bool)
        for body in (self.station, self.cart, self.pawl, self.ball):
            ok &= torch.isfinite(body.data.root_state_w).all(dim=-1)
        return ok

    def _still_now(self) -> torch.Tensor:
        c = self.cfg
        return ((self.cart.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.cart.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
                & (self.pawl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.ball_vmax)
                & (self.station.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    def settled(self) -> torch.Tensor:
        """(N,) bool: stillness has PERSISTED `settle_steps_min` consecutive steps."""
        return self.still_count >= self.cfg.settle_steps_min

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Stillness counter + latches: FOUL (cart ever past target rest +
        `overshoot` — permanent, checked FIRST so a fouling arrival earns
        nothing); pawl ever lifted clear; per-station arrival (near rest_i AND
        pawl seated AND cart slow AND not fouled)."""
        c = self.cfg
        fin = self._finite()
        self.still_count = (self.still_count + 1.0) * (self._still_now() & fin).float()
        q = self.cart_q()
        pq = self.pawl_q()
        self.foul_latch = torch.maximum(
            self.foul_latch, ((q > self.q_target() + c.overshoot) & fin).float())
        self.lift_latch = torch.maximum(
            self.lift_latch, ((pq >= c.lift_min) & fin).float())
        near = (q.unsqueeze(-1) - self.rests.unsqueeze(0)).abs() <= c.q_tol  # (N, n_notch)
        seated = (pq <= c.seat_tol)
        slow = self.cart.data.root_lin_vel_w.norm(dim=-1) < 0.05
        ok = (seated & slow & fin & (self.foul_latch < 0.5)).unsqueeze(-1)
        self.arrive_latch = torch.maximum(self.arrive_latch, (near & ok).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: never fouled, cart within q_tol of the TARGET notch rest,
        pawl seated in the notch (blade on a land reads ~lift_clear and fails),
        ball inside the tray, everything persistently still and finite. The state
        is self-persistent: the seated two-way detent holds the cart hands-off."""
        c = self.cfg
        return ((self.foul_latch < 0.5)
                & ((self.cart_q() - self.q_target()).abs() <= c.q_tol)
                & (self.pawl_q() <= c.seat_tol)
                & self.ball_in_tray()
                & self.settled() & self._finite())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0 forever once fouled; else 0.10 ever-lifted +
        0.50 * (stations 1..k arrived)/k (order physically forced downhill);
        capped 0.60; exactly 1.0 iff success(). Null policy earns exactly 0."""
        c = self.cfg
        idx = torch.arange(c.n_notch, device=self.k.device).unsqueeze(0)  # (1, n)
        mask = (idx >= 1) & (idx <= self.k.unsqueeze(-1))  # stations 1..k
        credit = (self.arrive_latch * mask.float()).sum(dim=-1) / self.k.float()
        base = (0.10 * self.lift_latch + 0.50 * credit).clamp(0.0, 0.60)
        base = torch.where(self.foul_latch > 0.5, torch.zeros_like(base), base)
        return torch.where(self.success(), base.new_tensor(1.0), base)


# Idempotent registration: on the forge, discovery may import this module under a
# different module name before the solve/smoke package-relative import re-executes it.
if "pawl_ladder_i431" not in SCENES.list():
    SCENES.register("pawl_ladder_i431", PawlLadderScene)
from robobench.core.registries import ENVS as _ENVS  # noqa: E402

if "simgen.pawl_ladder_i431" not in _ENVS.list():
    register_env("simgen",
                 lambda: EnvCfg(scene="pawl_ladder_i431", robot="null",
                                env_spacing=3.0))
