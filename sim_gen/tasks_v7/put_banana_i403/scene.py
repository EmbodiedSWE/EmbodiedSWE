"""FragilePackScene — install the sprung shock cradle into the crate FIRST, then set the
fragile orb down onto the cradle's tray; any hard landing snaps the orb's internal core
loose, permanently ruining it.

Derived from embodiedgen/put_banana ("pick the banana off the cluttered table and drop it
into the mug" — a free aerial pick-and-place into an open-topped container, judged by a
bounding-box containment; the payload is indestructible and any release height works).
Here the container (an open-topped shipping crate) keeps the seed's affordance — you CAN
carry the payload over the wall and drop it in — but the payload is FRAGILE: the white
porcelain orb carries a loose orange core welded to the shell by a brittle web (a
breakable fixed joint). A drop onto any hard surface snaps the web (the core rattles
loose — physically detectable as core/shell separation) and the damage is PERMANENT: no
later action can restore an intact orb, and a broken orb can never satisfy success. The
crate floor is hard, so the seed's plan — carry over the wall, release — produces an
irrecoverable FAILING state (smoke-checked). The only way to end with an intact orb in
the crate is to first build the soft landing: a carryable SHOCK CRADLE (teal base slab
with a tall grasp fin and a spring-suspended red tray) starts on the ground outside and
must be lowered into the crate; its sprung tray then arrests the orb's set-down
compliantly (the same release height that breaks the orb on the crate floor is harmless
on the tray — both outcomes smoke-checked). Terminal state: cradle seated in the crate
AND the intact orb settled on the cradle tray. Order cradle-into-crate BEFORE
orb-into-crate is forced by fragility: the orb cannot wait inside the crate (no hard
in-crate resting place is accepted, and getting it there intact requires the tray).

Assets are fully procedural, authored by custom compound spawners (child colliders of one
body never self-collide; joints are authored at spawn — post-play joints are dead on
this stack):
  - crate: KINEMATIC compound — 4 walls, open top, NO floor (the ground runs through).
    Interior 200 x 200 mm, walls 80 mm tall. Origin at the interior ground centre.
  - cradle: DYNAMIC compound — 120 x 120 x 10 mm base slab + a 15 mm-thick grasp fin on
    one edge rising to 120 mm (clears the crate wall by 40 mm when seated). Origin at
    the slab bottom centre.
  - tray: DYNAMIC compound — 80 x 80 x 6 mm plate with 12 mm curb walls, riding a
    Z-prismatic joint on the cradle (30 mm stroke) with a spring+damper linear drive
    (k = 130 N/m, c = 3 N.s/m): a real shock absorber, compresses under the orb.
  - orb: DYNAMIC 22 mm-radius white sphere (50 g), the fragile payload.
  - core: DYNAMIC 10 mm-radius orange sphere (20 g) concentric inside the orb, attached
    by a BREAKABLE fixed joint (break force ~1.5 N — the crash sensor). While intact the
    core tracks the shell exactly; a hard landing decelerates the shell in ~one solver
    step and the core's inertia snaps the weld, after which the core no longer tracks
    the shell (it settles ~12 mm low / spills out) — orientation-independent detection
    by core-vs-shell offset in the shell frame. The break is irreversible for the whole
    process lifetime (PhysX joint breaks do not heal on env.reset()).

Per-episode randomization (readback-verifiable): crate xy + yaw, cradle xy + yaw, orb xy.

Rubric (0..1; partial progress latched; a broken orb freezes all further credit):
  0.25 * seated       — cradle ever seated in the crate (latched bool)
  0.20 * approach     — orb progress toward the tray, gated on seated (latched max)
  0.40 * packed       — intact orb ever settled on the tray of the seated cradle
                        (latched bool)
  1.0 iff success()   — LIVE: cradle seated in the crate AND the orb settled on the
                        tray, still, with the core weld intact. Non-success capped 0.85.
All latch updates are gated on the orb being unbroken at that instant: after a break
nothing new can be earned (pre-break latched credit stays — credit never evaporates).

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root(stage, prim_path: str, translation, orientation):
    """Define the root Xform and author its (single) translate/orient ops."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _collider(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, size, center, color, collide: Callable) -> None:
    """Author one axis-aligned box child collider (Cube prim + scale op)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_sphere(stage, path: str, radius, center, color, collide: Callable) -> None:
    """Author one sphere child collider."""
    from pxr import Gf, UsdGeom

    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(float(radius))
    xf = UsdGeom.Xformable(sph.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(sph.GetPrim())


def _rigid_dynamic(root, mass: float, lin_damp: float, ang_damp: float,
                   max_depen: float) -> None:
    """Dynamic rigid body: custom spawners apply NO cfg schemas, so author MassAPI and
    the physx body attrs (sleep off — pose-held/spring-borne bodies must never sleep;
    16/4 solver iterations — velocity iters 4 kill sphere-on-box phantom creep)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(float(max_depen))
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)


def _spawn_crate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The shipping crate: KINEMATIC, 4 walls, open top, no floor (the hard ground runs
    flat through it — the naked crate floor is exactly as lethal as open ground)."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    collide = _collider(cfg.contact_offset)
    hx, hy, t, H = cfg.hx, cfg.hy, cfg.wall_t, cfg.wall_h
    _add_box(stage, f"{prim_path}/wall_xp", (t, 2 * (hy + t), H),
             (hx + t / 2, 0.0, H / 2), cfg.color, collide)
    _add_box(stage, f"{prim_path}/wall_xn", (t, 2 * (hy + t), H),
             (-hx - t / 2, 0.0, H / 2), cfg.color, collide)
    _add_box(stage, f"{prim_path}/wall_yp", (2 * hx, t, H),
             (0.0, hy + t / 2, H / 2), cfg.color, collide)
    _add_box(stage, f"{prim_path}/wall_yn", (2 * hx, t, H),
             (0.0, -hy - t / 2, H / 2), cfg.color, collide)
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The shock-cradle base: DYNAMIC slab (origin at slab bottom centre) + grasp fin on
    the +x edge rising well above the crate wall (top pinch stays reachable seated)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.1, ang_damp=0.2, max_depen=0.5)
    collide = _collider(cfg.contact_offset)
    sh, st = cfg.slab_half, cfg.slab_t
    _add_box(stage, f"{prim_path}/slab", (2 * sh, 2 * sh, st),
             (0.0, 0.0, st / 2), cfg.color, collide)
    _add_box(stage, f"{prim_path}/fin", (cfg.fin_t, cfg.fin_w, cfg.fin_h),
             (cfg.fin_x, 0.0, st + cfg.fin_h / 2), cfg.color, collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The sprung tray: DYNAMIC curbed plate (origin at plate centre) riding a
    Z-prismatic joint on the sibling cradle, with a spring+damper linear drive — a real
    shock absorber authored at spawn. Rest position = joint 0 (upper limit); the orb's
    weight and landing energy compress it downward against the spring."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.05, ang_damp=0.1, max_depen=0.5)
    collide = _collider(cfg.contact_offset)
    ph, pt = cfg.plate_half, cfg.plate_t
    _add_box(stage, f"{prim_path}/plate", (2 * ph, 2 * ph, pt),
             (0.0, 0.0, 0.0), cfg.color, collide)
    ct, chh = cfg.curb_t, cfg.curb_h
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/curb_x{'p' if sgn > 0 else 'n'}",
                 (ct, 2 * ph, chh),
                 (sgn * (ph - ct / 2), 0.0, pt / 2 + chh / 2), cfg.color, collide)
        _add_box(stage, f"{prim_path}/curb_y{'p' if sgn > 0 else 'n'}",
                 (2 * (ph - ct), ct, chh),
                 (0.0, sgn * (ph - ct / 2), pt / 2 + chh / 2), cfg.color, collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/shock")
    j.CreateBody0Rel().SetTargets([f"{base}/Cradle"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(cfg.rest_lx), 0.0, float(cfg.rest_lz)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(cfg.stroke))
    j.CreateUpperLimitAttr(0.0005)
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(cfg.spring_k))
    drv.CreateDampingAttr(float(cfg.spring_c))
    drv.CreateTargetPositionAttr(0.0)
    drv.CreateTargetVelocityAttr(0.0)
    return root


def _spawn_orb(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The fragile payload shell: DYNAMIC white sphere."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.05, ang_damp=0.3, max_depen=0.2)
    collide = _collider(cfg.contact_offset)
    _add_sphere(stage, f"{prim_path}/shell", cfg.radius, (0.0, 0.0, 0.0),
                cfg.color, collide)
    return root


def _spawn_core(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The crash-sensor core: DYNAMIC orange sphere spawned concentric inside the orb,
    welded to the sibling shell by a BREAKABLE fixed joint (collision between the pair
    stays DISABLED — they are concentric). A hard landing stops the shell in ~one solver
    step; the weld must then supply m_core*dv/dt, which exceeds break_force and snaps —
    after which the core stops tracking the shell, in any orientation."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass, lin_damp=0.05, ang_damp=0.3, max_depen=0.2)
    collide = _collider(cfg.contact_offset)
    _add_sphere(stage, f"{prim_path}/core", cfg.radius, (0.0, 0.0, 0.0),
                cfg.color, collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/weld")
    j.CreateBody0Rel().SetTargets([f"{base}/Orb"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateBreakForceAttr(float(cfg.break_force))
    j.CreateBreakTorqueAttr(float(cfg.break_torque))
    j.CreateCollisionEnabledAttr(False)
    return root


def _crate_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "crate" not in _SPAWNER_CACHE:

        @configclass
        class CrateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crate)
            hx: float = 0.10
            hy: float = 0.10
            wall_t: float = 0.012
            wall_h: float = 0.08
            color: tuple = (0.42, 0.45, 0.50)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["crate"] = CrateSpawnerCfg

    return _SPAWNER_CACHE["crate"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        hx=c.crate_hx, hy=c.crate_hy, wall_t=c.crate_wall_t, wall_h=c.crate_wall_h,
        color=c.crate_color, contact_offset=c.contact_offset,
    )


def _cradle_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cradle" not in _SPAWNER_CACHE:

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            slab_half: float = 0.06
            slab_t: float = 0.01
            fin_t: float = 0.015
            fin_w: float = 0.05
            fin_h: float = 0.11
            fin_x: float = 0.0525
            mass: float = 0.18
            color: tuple = (0.10, 0.55, 0.50)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cradle"] = CradleSpawnerCfg

    return _SPAWNER_CACHE["cradle"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.cradle_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        slab_half=c.slab_half, slab_t=c.slab_t, fin_t=c.fin_t, fin_w=c.fin_w,
        fin_h=c.fin_h, fin_x=c.fin_x, mass=c.cradle_mass, color=c.cradle_color,
        contact_offset=c.contact_offset,
    )


def _tray_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            plate_half: float = 0.04
            plate_t: float = 0.006
            curb_t: float = 0.006
            curb_h: float = 0.012
            mass: float = 0.02
            rest_lx: float = -0.005
            rest_lz: float = 0.055
            stroke: float = 0.030
            spring_k: float = 130.0
            spring_c: float = 3.0
            color: tuple = (0.80, 0.25, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        plate_half=c.plate_half, plate_t=c.plate_t, curb_t=c.curb_t, curb_h=c.curb_h,
        mass=c.tray_mass, rest_lx=c.tray_rest_lx, rest_lz=c.tray_rest_lz,
        stroke=c.tray_stroke, spring_k=c.spring_k, spring_c=c.spring_c,
        color=c.tray_color, contact_offset=c.contact_offset,
    )


def _orb_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "orb" not in _SPAWNER_CACHE:

        @configclass
        class OrbSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_orb)
            radius: float = 0.022
            mass: float = 0.05
            color: tuple = (0.93, 0.91, 0.86)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["orb"] = OrbSpawnerCfg

    return _SPAWNER_CACHE["orb"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.orb_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        radius=c.orb_r, mass=c.orb_mass, color=c.orb_color,
        contact_offset=c.contact_offset,
    )


def _core_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "core" not in _SPAWNER_CACHE:

        @configclass
        class CoreSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_core)
            radius: float = 0.010
            mass: float = 0.02
            break_force: float = 1.5
            break_torque: float = 0.05
            color: tuple = (0.95, 0.58, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["core"] = CoreSpawnerCfg

    return _SPAWNER_CACHE["core"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.core_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        radius=c.core_r, mass=c.core_mass, break_force=c.break_force,
        break_torque=c.break_torque, color=c.core_color,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FragilePackSceneCfg(BaseCfg):
    """Config for `FragilePackScene`. Fragility calibration (dt = 1/120): a hard landing
    stops the shell in ~one solver step, so the weld sees ~ m_core*v*120: a 10 cm free
    drop (v = 1.40 m/s) loads it ~3.4 N >> 1.5 N break; the spawn settle (2 mm,
    v = 0.20 m/s) loads ~0.48 N; a 15 mm release onto the sprung tray decelerates over
    the spring stroke (~0.6 N peak incl. gravity) — both safely under the threshold."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    break_tol: float = tunable(0.008)  # core-vs-shell offset (shell frame) beyond this = broken (m)
    seat_xy_tol: float = tunable(0.035)  # cradle centre within this of the crate centre, xy (m)
    seat_z_tol: float = tunable(0.008)  # cradle origin at ground rest within this (m)
    seat_tilt_max_deg: float = tunable(15.0)  # cradle up-axis within this of world-up
    orb_xy_tol: float = tunable(0.028)  # orb centre within this of the tray centre, tray frame (m)
    orb_z_tol: float = tunable(0.012)  # orb centre at plate_t/2 + orb_r in tray frame within this (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    approach_d0: float = tunable(0.50)  # approach ramp: p = 1 - d/approach_d0

    # --- tunable: fragility + suspension (the task-family knobs) ---------------------------------
    break_force: float = tunable(1.5)  # core weld break force (N) — the crash sensor
    break_torque: float = tunable(0.05)  # core weld break torque (N.m)
    spring_k: float = tunable(130.0)  # tray suspension stiffness (N/m)
    spring_c: float = tunable(3.0)  # tray suspension damping (N.s/m)

    # --- tunable: randomization ------------------------------------------------------------------
    crate_jitter: float = tunable(0.03)  # crate xy jitter (+/- m)
    crate_yaw_deg: float = tunable(180.0)  # crate yaw (+/- deg)
    cradle_jitter: float = tunable(0.03)  # cradle spawn xy jitter (+/- m)
    cradle_yaw_deg: float = tunable(180.0)  # cradle spawn yaw (+/- deg)
    orb_jitter: float = tunable(0.04)  # orb spawn xy jitter (+/- m)

    # --- info: layout (single Franka base at the origin, facing +x) ------------------------------
    crate_pos: tuple = info((0.54, -0.16))  # crate centre
    cradle_pos: tuple = info((0.40, 0.22))  # cradle spawn centre (on open ground)
    orb_pos: tuple = info((0.32, 0.00))  # orb spawn centre (on open ground)

    # --- info: structure ---------------------------------------------------------------------------
    crate_hx: float = info(0.10)  # crate interior half-extent
    crate_hy: float = info(0.10)
    crate_wall_t: float = info(0.012)
    crate_wall_h: float = info(0.08)
    crate_color: tuple = info((0.42, 0.45, 0.50))  # slate grey
    slab_half: float = info(0.06)  # cradle base slab half-extent
    slab_t: float = info(0.01)
    fin_t: float = info(0.015)  # grasp fin thickness (x)
    fin_w: float = info(0.05)  # grasp fin width (y)
    fin_h: float = info(0.11)  # fin height above the slab (top at 120 mm)
    fin_x: float = info(0.0525)  # fin centre offset (slab +x edge)
    cradle_mass: float = info(0.18)
    cradle_color: tuple = info((0.10, 0.55, 0.50))  # teal
    plate_half: float = info(0.04)  # tray plate half-extent
    plate_t: float = info(0.006)
    curb_t: float = info(0.006)
    curb_h: float = info(0.012)
    tray_mass: float = info(0.02)  # light: first-contact momentum exchange stays gentle
    tray_rest_lx: float = info(-0.005)  # tray origin in cradle frame at joint rest
    tray_rest_lz: float = info(0.055)
    tray_stroke: float = info(0.030)  # suspension travel (m)
    tray_color: tuple = info((0.80, 0.25, 0.20))  # red
    orb_r: float = info(0.022)
    orb_mass: float = info(0.05)
    orb_color: tuple = info((0.93, 0.91, 0.86))  # porcelain white
    core_r: float = info(0.010)
    core_mass: float = info(0.02)
    core_color: tuple = info((0.95, 0.58, 0.10))  # orange
    contact_offset: float = info(0.002)
    # rubric weights (0.25 + 0.20 + 0.40 = 0.85 = the non-success cap)
    w_seat: float = info(0.25)
    w_app: float = info(0.20)
    w_pack: float = info(0.40)

    # Derived (filled in __post_init__).
    orb_rest_z: float = field(default=None, init=False)  # orb centre height on the ground
    orb_tray_lz: float = field(default=None, init=False)  # orb centre in TRAY frame when riding

    def __post_init__(self) -> None:
        self.orb_rest_z = round(self.orb_r, 4)
        self.orb_tray_lz = round(self.plate_t / 2 + self.orb_r, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("fragile_pack")
class FragilePackScene(BaseScene):
    cfg: FragilePackSceneCfg

    def __init__(self, cfg: FragilePackSceneCfg | None = None) -> None:
        super().__init__(cfg or FragilePackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        # NOTE: dict order = spawn order. Cradle before Tray and Orb before Core — each
        # joint spawner targets its already-authored sibling.
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
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=_crate_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crate_pos[0], c.crate_pos[1], 0.0)),
            ),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=_cradle_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cradle_pos[0], c.cradle_pos[1], 0.002)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cradle_pos[0] + c.tray_rest_lx, c.cradle_pos[1],
                         0.002 + c.tray_rest_lz)),
            ),
            "orb": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Orb",
                spawn=_orb_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.orb_pos[0], c.orb_pos[1], c.orb_rest_z + 0.002)),
            ),
            "core": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Core",
                spawn=_core_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.orb_pos[0], c.orb_pos[1], c.orb_rest_z + 0.002)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.crate: RigidObject = env.iscene["crate"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.tray: RigidObject = env.iscene["tray"]
        self.orb: RigidObject = env.iscene["orb"]
        self.core: RigidObject = env.iscene["core"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._broken = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._seated = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._app_max = torch.zeros(n, device=env.device)
        self._packed = torch.zeros(n, dtype=torch.bool, device=env.device)

    def _yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        return q

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: crate placed with xy jitter + yaw, the cradle (with its tray
        at joint rest — jointed pairs are always written together, consistently) on open
        ground with xy + yaw jitter, the orb (with its core concentric) on open ground;
        clear latches. NOTE: env.reset() cannot heal a snapped core weld — a break is
        permanent for the process; the smoke battery orders its probes accordingly."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- crate (kinematic fixture) ---
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.crate_yaw_deg)
        cxy = torch.tensor(c.crate_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cxy
        st[:, 3:7] = self._yaw_quat(cyaw)
        st[:, 0:3] += origin
        self.crate.write_root_state_to_sim(st, env_ids)

        # --- cradle + tray (jointed pair: written together, joint at rest) ---
        kyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cradle_yaw_deg)
        kq = self._yaw_quat(kyaw)
        kxy = torch.tensor(c.cradle_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.cradle_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = kxy
        st[:, 2] = 0.002
        st[:, 3:7] = kq
        st[:, 0:3] += origin
        self.cradle.write_root_state_to_sim(st, env_ids)
        cosy, siny = torch.cos(kyaw), torch.sin(kyaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = kxy[:, 0] + cosy * c.tray_rest_lx
        st[:, 1] = kxy[:, 1] + siny * c.tray_rest_lx
        st[:, 2] = 0.002 + c.tray_rest_lz
        st[:, 3:7] = kq
        st[:, 0:3] += origin
        self.tray.write_root_state_to_sim(st, env_ids)

        # --- orb + core (concentric pair: written together) ---
        oxy = torch.tensor(c.orb_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.orb_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = oxy
        st[:, 2] = c.orb_rest_z + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.orb.write_root_state_to_sim(st.clone(), env_ids)
        self.core.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._broken[env_ids] = False
        self._seated[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._packed[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "cradle": self.cradle.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "orb": self.orb.data.root_state_w[env_ids].clone(),
            "core": self.core.data.root_state_w[env_ids].clone(),
            "broken": self._broken[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "packed": self._packed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.cradle.write_root_state_to_sim(state["cradle"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.orb.write_root_state_to_sim(state["orb"], env_ids)
        self.core.write_root_state_to_sim(state["core"], env_ids)
        self._broken[env_ids] = state["broken"]
        self._seated[env_ids] = state["seated"]
        self._app_max[env_ids] = state["app_max"]
        self._packed[env_ids] = state["packed"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the ground stands an open-topped grey shipping CRATE (interior "
            f"{2 * c.crate_hx * 100:.0f} x {2 * c.crate_hy * 100:.0f} cm, walls "
            f"{c.crate_wall_h * 100:.0f} cm tall, bare hard floor). Nearby on open "
            f"ground sit two loose items: a teal SHOCK CRADLE — a "
            f"{2 * c.slab_half * 100:.0f} cm square base slab with a tall grasp fin on "
            f"one edge (top at {(c.slab_t + c.fin_h) * 100:.0f} cm, well above the "
            f"crate wall) carrying a RED TRAY on a real spring suspension "
            f"({c.tray_stroke * 100:.0f} cm of travel — it visibly compresses under "
            f"load) — and a fragile white porcelain ORB ({2 * c.orb_r * 100:.1f} cm "
            f"across). Inside the orb an orange CORE is held by a brittle web: any hard "
            f"landing — dropping the orb onto the ground, the crate floor, or any other "
            f"hard surface — SNAPS the web (the loose core visibly rattles out of "
            f"place) and the orb is PERMANENTLY ruined; nothing can repair it. A "
            f"gentle set-down onto the sprung red tray is harmless — the suspension "
            f"soaks up the landing. Positions and headings of the crate, the cradle "
            f"and the orb change between episodes — locate them visually.\n"
            f"Goal — pack the fragile orb into the crate, which is only survivable in "
            f"this order: (1) lower the shock cradle into the crate by its fin and "
            f"seat it centred on the crate floor; (2) set the orb down gently onto the "
            f"cradle's red tray so the suspension catches it. Success requires the "
            f"cradle seated inside the crate (centred within "
            f"{c.seat_xy_tol * 100:.1f} cm, upright) AND the orb at rest ON the tray, "
            f"with the core weld still intact, everything settled. A broken orb can "
            f"never succeed, wherever it ends up; an orb parked on the bare crate "
            f"floor, on open ground, or anywhere off the tray fails; a cradle left "
            f"outside the crate fails."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lower the teal shock cradle into the grey crate by its fin and seat it on "
            "the crate floor, then set the fragile white orb down gently onto the "
            "cradle's sprung red tray — never drop the orb onto a hard surface, or its "
            "core snaps loose and the task is failed for good."
        )

    # ----- frames / predicates ---------------------------------------------------------------------
    def _local(self, body, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3): `pos_w` in `body`'s frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body.data.root_quat_w, pos_w - body.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        """(N,) the body's local +z axis, world z component."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    def core_offset(self) -> torch.Tensor:
        """(N,) |core - shell| in the SHELL frame — 0 while the weld is intact,
        >= ~12 mm once snapped (the core settles low / spills out), any orientation."""
        return self._local(self.orb, self.core.data.root_pos_w).norm(dim=-1)

    def intact_now(self) -> torch.Tensor:
        """(N,) bool: the core still tracks the shell AND no break was ever latched."""
        return (self.core_offset() < self.cfg.break_tol) & ~self._broken

    def cradle_seated_now(self) -> torch.Tensor:
        """(N,) bool: cradle centred on the crate floor, at ground rest, upright, still."""
        c = self.cfg
        loc = self._local(self.crate, self.cradle.data.root_pos_w)
        in_xy = (loc[:, 0].abs() < c.seat_xy_tol) & (loc[:, 1].abs() < c.seat_xy_tol)
        on_ground = (loc[:, 2] - 0.002).abs() < c.seat_z_tol
        upright = self._up_z(self.cradle) >= math.cos(math.radians(c.seat_tilt_max_deg))
        still = self.cradle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return in_xy & on_ground & upright & still

    def orb_on_tray_now(self) -> torch.Tensor:
        """(N,) bool: orb resting ON the tray plate (tray frame: inside the curbs, at
        plate_t/2 + orb_r within tolerance — the spring sag moves tray and orb TOGETHER,
        so the tray-frame height is suspension-invariant), still."""
        c = self.cfg
        loc = self._local(self.tray, self.orb.data.root_pos_w)
        in_xy = (loc[:, 0].abs() < c.orb_xy_tol) & (loc[:, 1].abs() < c.orb_xy_tol)
        at_z = (loc[:, 2] - c.orb_tray_lz).abs() < c.orb_z_tol
        still = self.orb.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return in_xy & at_z & still

    def _update_latches(self) -> None:
        c = self.cfg
        self._broken |= self.core_offset() >= c.break_tol
        ok = ~self._broken  # a broken orb freezes ALL further credit
        self._seated |= self.cradle_seated_now() & ok
        d = (self.orb.data.root_pos_w[:, :2] - self.tray.data.root_pos_w[:, :2]).norm(dim=-1)
        app = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * (self._seated & ok).float()
        self._app_max = torch.maximum(self._app_max, app)
        self._packed |= self.cradle_seated_now() & self.orb_on_tray_now() & ok

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool, LIVE physical outcome: cradle seated in the crate AND the orb
        settled on its tray AND the core weld intact (never broken), everything still."""
        self._update_latches()
        return self.cradle_seated_now() & self.orb_on_tray_now() & self.intact_now()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*seated + 0.20*approach (gated on seated) +
        0.40*packed — all latched, all latch updates gated on an unbroken orb (a break
        freezes credit; ~0 for the null policy and for the seed's drop-it-in plan),
        capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seat * self._seated.float() + c.w_app * self._app_max
                + c.w_pack * self._packed.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="fragile_pack", robot="null"))
