"""ButterCellarScene — press the loaded dumbwaiter platform down past the one-way
ratchet so the butter is latched deep inside the silo.

Derived from libero_90/kitchen_scene10 "put the butter at the back in the top drawer of
the cabinet and close it", but the CONTAINMENT MECHANISM is replaced wholesale: the
seed's receptacle is a drawer that must be OPENED (grasp the handle, pull the prismatic
drawer out), have the butter placed inside it, and then be CLOSED again — an
open/insert/close cycle on a reversible sliding volume. Here NOTHING opens and nothing
closes. The receptacle is a BUTTER CELLAR: an open-topped silo whose interior is
blocked by a spring-raised dumbwaiter PLATFORM riding a vertical prismatic slide. At
rest the platform's tray sits just below the rim, held up by a constant-force spring —
the deep interior is unreachable. The solver must put the butter ON the exposed tray
and then DRIVE THE WHOLE RECEPTACLE DOWN: press the platform 17 cm down its slide,
past a brass one-way ratchet PAWL, and let go. The spring throws the platform back up,
the tray's edge jams against the pawl's underside, and the platform is LATCHED deep in
the silo with the butter riding it — a one-way, irreversible press cycle in which the
transported object never enters a static cavity at all; the loaded surface ITSELF
descends into the goal region. The plunger post's socket cap invites the intended
press: rest the heavy iron INGOT in the socket and its weight overwhelms the spring
(gravity does the pressing); the ingot must afterwards be lifted out and returned to
the floor clear of the silo, or the goal does not count. A white LARD block of
identical shape must stay out of the silo. The seed's plan (open a receptacle, place
inside, close it) has no purchase here: there is no handle, no door, no drawer, and the
end state a seed-strategy solver would produce — butter resting in the receptacle with
the receptacle "shut" (i.e. the platform still up at its rest stop) — is an explicit
smoke-tested failure.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - cellar: KINEMATIC compound silo — floor slab, four walls around a 140 x 140 mm
    vertical bore, rim 300 mm up, with a shallow full-height channel recessed into the
    +x wall housing the pawl. Origin at the ground centre of the silo.
  - platform: DYNAMIC compound — 124 x 124 x 16 mm tray (origin at tray centre), a
    235 mm plunger post rising from the tray's -x/-y corner to a 60 x 60 mm socket cap
    with a 12 mm lip (socket stays above the rim over the full stroke, so no finger
    ever enters the bore). Rides a bind-time vertical PrismaticJoint (cellar ->
    platform, joint-pair collision disabled; the slide owns alignment). A constant
    upward spring force + viscous damping (post_step-owned) holds it at the top stop.
  - pawl: DYNAMIC brass plate (50 x 48 x 8 mm) on a bind-time RevoluteJoint (axis +y)
    hinged inside the wall channel, origin ON the hinge line. Limits [-85, 0] deg:
    it can fold DOWN into the channel (a descending tray cams it aside by contact) but
    cannot rise above horizontal — an ascending tray jams against it at the 0 deg
    stop. Gravity is disabled; a weak post_step spring torque returns it to
    horizontal. Joint-pair collision (cellar<->pawl) disabled; pawl<->platform contact
    is the ratchet and stays ON.
  - butter (yellow) / lard (white): 58 x 32 x 26 mm blocks — the 32 mm faces are the
    parallel-jaw grasp feature. ingot (dark iron): 45 x 45 x 75 mm, 0.55 kg — heavy
    enough that resting it in the socket presses the platform down (spring surplus is
    ~1.1 N with the butter aboard), light enough for a single-arm carry.

Per-episode randomization (readback-verifiable): a full random permutation assigning
{butter, lard, ingot} to the three floor spawn slots, plus per-object xy jitter and
free yaw. The mechanism (cellar + platform + pawl) is authored at its final pose and
never moved — the jointed pairs are re-posed follower-only at reset.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * approach   — butter approach to the tray load point, measured against the
                      per-episode spawn distance (p = 1 - d/d_init, running max;
                      exactly 0 for doing nothing)
  0.20 * loaded     — butter ever settled on the tray (latched bool)
  0.25 * depth      — press progress: running MIN of platform height mapped over the
                      rest->latch stroke (rising-only credit)
  0.20 * latched    — platform ever at latch depth, settled, pawl horizontal
                      (latched bool; the spring makes this band unreachable as a rest
                      state except under the pawl or under an external load)
  0.10 * cleared    — latched AND the ingot clear of the silo on the floor
                      (latched bool)
  1.0 iff success() — butter settled ON the tray, platform in the latch band with the
                      pawl seated, ingot returned to the floor clear of the silo, lard
                      outside, everything at rest. Non-success cap 0.85.

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
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _root_xform(prim_path: str, translation, orientation):
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


def _dynamic_body(root, mass: float, *, lin_damp: float, ang_damp: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(lin_damp)
    pxrb.CreateAngularDampingAttr(ang_damp)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_cellar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the silo: KINEMATIC open-topped shaft around a square vertical bore.
    Origin at the ground centre. Interior bore: |x|,|y| <= bore/2, z in [floor_t, rim_h];
    the +x wall carries a full-height recess channel (|y| <= chan_w/2) for the pawl."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    b = c.bore / 2  # bore half-width
    t = c.wall_t
    oh = b + t  # outer half-width
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(c.bore, c.bore, c.floor_t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_xn", center=(-(b + t / 2), 0.0, c.rim_h / 2),
             size=(t, 2 * oh, c.rim_h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_yn")):
        _add_box(stage, f"{prim_path}/{nm}", center=(0.0, sgn * (b + t / 2), c.rim_h / 2),
                 size=(c.bore, t, c.rim_h), color=c.color, collide=collide)
    # +x wall: two strips flanking the pawl channel + a back plate sealing the recess
    strip_w = oh - c.chan_w / 2
    for sgn, nm in ((1.0, "wall_xp_l"), (-1.0, "wall_xp_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(b + t / 2, sgn * (c.chan_w / 2 + strip_w / 2), c.rim_h / 2),
                 size=(t, strip_w, c.rim_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/chan_back",
             center=(b + c.chan_d + t / 2, 0.0, c.rim_h / 2),
             size=(t, c.chan_w + 0.004, c.rim_h), color=c.chan_color, collide=collide)
    return root


def _spawn_platform(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dumbwaiter platform: DYNAMIC compound, origin at the TRAY CENTRE.
    Children: tray plate, plunger post rising from the -x/-y corner, socket cap with a
    four-strip lip (socket floor = local z 0.008 + post_h + cap_t)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=0.0, ang_damp=0.05)
    collide = _make_collide(cfg)
    c = cfg
    ht = c.tray_t / 2
    _add_box(stage, f"{prim_path}/tray", center=(0.0, 0.0, 0.0),
             size=(c.tray_w, c.tray_w, c.tray_t), color=c.tray_color, collide=collide)
    px, py = c.post_xy
    _add_box(stage, f"{prim_path}/post", center=(px, py, ht + c.post_h / 2),
             size=(c.post_w, c.post_w, c.post_h), color=c.steel_color, collide=collide)
    cap_z = ht + c.post_h + c.cap_t / 2
    _add_box(stage, f"{prim_path}/cap", center=(px, py, cap_z),
             size=(c.cap_w, c.cap_w, c.cap_t), color=c.steel_color, collide=collide)
    lip_z = ht + c.post_h + c.cap_t + c.lip_h / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/lip_y{'p' if sgn > 0 else 'n'}",
                 center=(px, py + sgn * (c.cap_w / 2 - c.lip_t / 2), lip_z),
                 size=(c.cap_w, c.lip_t, c.lip_h), color=c.lip_color, collide=collide)
        _add_box(stage, f"{prim_path}/lip_x{'p' if sgn > 0 else 'n'}",
                 center=(px + sgn * (c.cap_w / 2 - c.lip_t / 2), py, lip_z),
                 size=(c.lip_t, c.cap_w - 2 * c.lip_t, c.lip_h), color=c.lip_color,
                 collide=collide)
    return root


def _spawn_pawl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the ratchet pawl: DYNAMIC brass plate, origin ON the hinge line; the
    plate spans local x in [-pawl_l, 0] (protruding toward the bore centre at rest),
    y in +/- pawl_w/2, z in +/- pawl_t/2. Gravity disabled (a post_step spring torque
    returns it to horizontal); sleep zeroed — it must respond instantly."""
    from pxr import PhysxSchema

    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=0.01, ang_damp=0.01)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateDisableGravityAttr(True)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/plate", center=(-cfg.pawl_l / 2, 0.0, 0.0),
             size=(cfg.pawl_l, cfg.pawl_w, cfg.pawl_t), color=cfg.color, collide=collide)
    return root


def _spawn_block(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a free block (butter / lard / ingot): DYNAMIC box, origin at its centre."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, cfg.mass_props.mass, lin_damp=0.05, ang_damp=0.10)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(cfg.bx, cfg.by, cfg.bz), color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cellar" not in _SPAWNER_CACHE:

        @configclass
        class CellarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cellar)
            bore: float = 0.14
            wall_t: float = 0.02
            floor_t: float = 0.02
            rim_h: float = 0.30
            chan_w: float = 0.052
            chan_d: float = 0.016
            color: tuple = (0.35, 0.38, 0.45)
            chan_color: tuple = (0.28, 0.30, 0.36)
            contact_offset: float = 0.002

        @configclass
        class PlatformSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platform)
            tray_w: float = 0.124
            tray_t: float = 0.016
            post_xy: tuple = (-0.035, -0.035)
            post_w: float = 0.024
            post_h: float = 0.235
            cap_w: float = 0.060
            cap_t: float = 0.008
            lip_t: float = 0.005
            lip_h: float = 0.012
            tray_color: tuple = (0.75, 0.62, 0.40)
            steel_color: tuple = (0.55, 0.56, 0.60)
            lip_color: tuple = (0.40, 0.41, 0.45)
            contact_offset: float = 0.002

        @configclass
        class PawlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pawl)
            pawl_l: float = 0.05
            pawl_w: float = 0.048
            pawl_t: float = 0.008
            color: tuple = (0.80, 0.62, 0.22)
            contact_offset: float = 0.002

        @configclass
        class BlockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_block)
            bx: float = 0.058
            by: float = 0.032
            bz: float = 0.026
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cellar=CellarSpawnerCfg, platform=PlatformSpawnerCfg,
                              pawl=PawlSpawnerCfg, block=BlockSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ButterCellarSceneCfg(BaseCfg):
    """Config for `ButterCellarScene`. The one-way interlock is metric: the spring's
    constant 3.6 N always beats the platform (0.18 kg) plus butter (0.08 kg) weights
    (~2.55 N), so the latch band below the pawl is unreachable as a rest state except
    jammed under the pawl (or pinned by an external load, which the ingot-clear clause
    then rejects); the 0.55 kg ingot (5.4 N) tips the balance and presses the platform
    down; the pawl's 0-deg joint stop blocks the return stroke."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    latch_z_lo: float = tunable(0.082)  # latch band: platform tray-centre height (m) ...
    latch_z_hi: float = tunable(0.150)  # ... must sit in (lo, hi) — only the pawl holds this
    pawl_closed_deg: float = tunable(10.0)  # pawl counts as seated within this of horizontal
    on_tray_xy: float = tunable(0.060)  # butter centre within this of the tray centre (m)
    on_tray_dz_lo: float = tunable(0.004)  # butter CoM height above the tray top plane ...
    on_tray_dz_hi: float = tunable(0.050)  # ... must sit in (lo, hi)
    ingot_clear_xy: float = tunable(0.13)  # ingot centre at least this far from the silo axis
    ingot_clear_z: float = tunable(0.10)  # ... and at most this high (on the floor)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    spawn_jitter: float = tunable(0.025)  # per-object spawn xy jitter (+/- m)
    yaw_free: bool = tunable(True)  # free spawn yaw (demo sets False)
    permute_slots: bool = tunable(True)  # random object->slot permutation (demo sets False)

    # --- tunable: mechanism plant ----------------------------------------------------------------
    spring_f: float = tunable(3.6)  # constant upward spring force on the platform (N)
    # 3.6 N: beats platform (1.77 N) and platform+butter (2.55 N) weights, loses to
    # +ingot (7.95 N); the ~1 N loaded surplus keeps per-step chatter against the
    # pawl far below settle_speed (a larger surplus made velocity readback flicker)
    spring_c: float = tunable(12.0)  # viscous damping on the platform slide (N*s/m)
    pawl_k: float = tunable(0.010)  # pawl return-spring stiffness (N*m/rad)
    pawl_c: float = tunable(0.0010)  # pawl hinge damping (N*m*s/rad)

    # --- info: layout (single Franka base at the origin) -----------------------------------------
    cellar_pos: tuple = info((0.56, 0.0))  # silo axis xy
    spawn_slots: tuple = info(((0.36, 0.20), (0.33, 0.00), (0.36, -0.20)))
    load_xy: tuple = info((-0.028, 0.032))  # butter load point on the tray (platform frame)

    # --- info: cellar structure ------------------------------------------------------------------
    bore: float = info(0.14)
    wall_t: float = info(0.02)
    floor_t: float = info(0.02)
    rim_h: float = info(0.30)
    chan_w: float = info(0.052)
    chan_d: float = info(0.016)
    cellar_color: tuple = info((0.35, 0.38, 0.45))  # slate blue-gray
    chan_color: tuple = info((0.28, 0.30, 0.36))

    # --- info: platform --------------------------------------------------------------------------
    tray_w: float = info(0.124)
    tray_t: float = info(0.016)
    post_xy: tuple = info((-0.035, -0.035))  # plunger post at the tray's -x/-y corner
    post_w: float = info(0.024)
    post_h: float = info(0.235)
    cap_w: float = info(0.060)
    cap_t: float = info(0.008)
    lip_t: float = info(0.005)
    lip_h: float = info(0.012)
    platform_mass: float = info(0.18)
    z_top: float = info(0.262)  # tray-centre height at the top (rest) stop
    travel: float = info(0.175)  # prismatic stroke (tray centre 0.262 -> 0.087)
    z_latch_eq: float = info(0.138)  # tray-centre height when jammed under the pawl

    # --- info: pawl ------------------------------------------------------------------------------
    pawl_l: float = info(0.05)
    pawl_w: float = info(0.048)
    pawl_t: float = info(0.008)
    pawl_mass: float = info(0.02)
    pawl_fold_deg: float = info(85.0)  # downward fold stop (into the wall channel)
    hinge_local: tuple = info((0.080, 0.0, 0.150))  # hinge in cellar frame (axis = +y)
    pawl_color: tuple = info((0.80, 0.62, 0.22))  # brass

    # --- info: free blocks -----------------------------------------------------------------------
    butter_size: tuple = info((0.058, 0.032, 0.026))
    butter_mass: float = info(0.08)
    butter_color: tuple = info((0.93, 0.80, 0.22))  # butter yellow — the payload
    lard_color: tuple = info((0.93, 0.93, 0.90))  # white — identical shape, must stay out
    ingot_size: tuple = info((0.045, 0.045, 0.075))
    ingot_mass: float = info(0.55)  # 5.4 N — beats the spring surplus (~3.5 N)
    ingot_color: tuple = info((0.25, 0.26, 0.30))  # dark iron — the press weight

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.20 + 0.25 + 0.20 + 0.10 = 0.85 = the non-success cap)
    w_app: float = info(0.10)
    w_load: float = info(0.20)
    w_depth: float = info(0.25)
    w_latch: float = info(0.20)
    w_clear: float = info(0.10)

    # Derived (filled in __post_init__).
    socket_floor_dz: float = field(default=None, init=False)  # tray centre -> socket floor
    tray_top_dz: float = field(default=None, init=False)  # tray centre -> tray top plane

    def __post_init__(self) -> None:
        self.tray_top_dz = self.tray_t / 2
        self.socket_floor_dz = self.tray_t / 2 + self.post_h + self.cap_t


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("butter_cellar")
class ButterCellarScene(BaseScene):
    cfg: ButterCellarSceneCfg

    def __init__(self, cfg: ButterCellarSceneCfg | None = None) -> None:
        super().__init__(cfg or ButterCellarSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        cellar_spawn = spawners["cellar"](
            mass_props=sim_utils.MassPropertiesCfg(mass=12.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bore=c.bore, wall_t=c.wall_t, floor_t=c.floor_t, rim_h=c.rim_h,
            chan_w=c.chan_w, chan_d=c.chan_d, color=c.cellar_color,
            chan_color=c.chan_color, contact_offset=c.contact_offset,
        )
        platform_spawn = spawners["platform"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.platform_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            tray_w=c.tray_w, tray_t=c.tray_t, post_xy=c.post_xy, post_w=c.post_w,
            post_h=c.post_h, cap_w=c.cap_w, cap_t=c.cap_t, lip_t=c.lip_t, lip_h=c.lip_h,
            contact_offset=c.contact_offset,
        )
        pawl_spawn = spawners["pawl"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.pawl_mass),
            # disable_gravity set HERE (not only as a raw USD attr in the spawner):
            # isaaclab applies rigid_props after the spawn func, so this is the
            # authoritative setting — the post_step spring is the pawl's only return
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            pawl_l=c.pawl_l, pawl_w=c.pawl_w, pawl_t=c.pawl_t, color=c.pawl_color,
            contact_offset=c.contact_offset,
        )

        def block_spawn(size, mass, color):
            return spawners["block"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                bx=size[0], by=size[1], bz=size[2], color=color,
                contact_offset=c.contact_offset,
            )

        cx, cy = c.cellar_pos
        hinge_w = (cx + c.hinge_local[0], cy + c.hinge_local[1], c.hinge_local[2])
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
            "cellar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cellar",
                spawn=cellar_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.0)),
            ),
            "platform": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platform",
                spawn=platform_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, c.z_top)),
            ),
            "pawl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pawl",
                spawn=pawl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=hinge_w),
            ),
            "butter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Butter",
                spawn=block_spawn(c.butter_size, c.butter_mass, c.butter_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_slots[0][0], c.spawn_slots[0][1],
                         c.butter_size[2] / 2 + 0.002)),
            ),
            "lard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lard",
                spawn=block_spawn(c.butter_size, c.butter_mass, c.lard_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_slots[1][0], c.spawn_slots[1][1],
                         c.butter_size[2] / 2 + 0.002)),
            ),
            "ingot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ingot",
                spawn=block_spawn(c.ingot_size, c.ingot_mass, c.ingot_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.spawn_slots[2][0], c.spawn_slots[2][1],
                         c.ingot_size[2] / 2 + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                # NOTE: enable_external_forces_every_iteration must stay OFF — it
                # breaks the set_external_force_and_torque-based plant (the pawl's
                # hinge spring acquires a phantom equilibrium at -25 deg). The spring
                # surplus is instead sized (see spring_f) so per-step contact chatter
                # stays well under the settle_speed threshold.
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
        self.cellar: RigidObject = env.iscene["cellar"]
        self.platform: RigidObject = env.iscene["platform"]
        self.pawl: RigidObject = env.iscene["pawl"]
        self.butter: RigidObject = env.iscene["butter"]
        self.lard: RigidObject = env.iscene["lard"]
        self.ingot: RigidObject = env.iscene["ingot"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._app_max = torch.zeros(n, device=dev)  # tray-load approach, running max
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)  # butter ever on tray
        self._min_z = torch.full((n,), self.cfg.z_top, device=dev)  # deepest platform height
        self._latched = torch.zeros(n, dtype=torch.bool, device=dev)  # ever held in latch band
        self._cleared = torch.zeros(n, dtype=torch.bool, device=dev)  # latched + ingot clear
        self._d_init = torch.full((n,), 0.3, device=dev)  # spawn->load distance (reset-set)

    def _author_joints(self) -> None:
        """Per env, two bind-time joints anchored on the kinematic cellar (the mechanism
        is authored at its final pose and never moved):
          - cellar->platform vertical PrismaticJoint on the bore axis, limits
            [-travel, 0] about the top stop; joint-pair collision disabled (the slide
            owns alignment; the walls still block the free blocks everywhere);
          - cellar->pawl RevoluteJoint (axis +y) on the channel hinge line, limits
            [-fold, 0] deg: it folds DOWN under a descending tray and jams an ascending
            tray at the 0-deg stop (the one-way ratchet); joint-pair collision
            disabled (the pawl sweeps through the wall recess), pawl<->platform
            contact stays ON — it IS the latch."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/platform_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Cellar"])
            j.CreateBody1Rel().SetTargets([f"{base}/Platform"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.z_top)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.travel))
            j.CreateUpperLimitAttr(0.0)

            r = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/pawl_hinge")
            r.CreateBody0Rel().SetTargets([f"{base}/Cellar"])
            r.CreateBody1Rel().SetTargets([f"{base}/Pawl"])
            r.CreateCollisionEnabledAttr(False)
            r.CreateAxisAttr("Y")
            r.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in c.hinge_local]))
            r.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            r.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLowerLimitAttr(-float(c.pawl_fold_deg))
            r.CreateUpperLimitAttr(0.0)

    # ----- frames --------------------------------------------------------------------------------
    def _cellar_xy(self) -> torch.Tensor:
        return self.cellar.data.root_pos_w[:, 0:2]

    def _hinge_world(self) -> torch.Tensor:
        c = self.cfg
        h = torch.tensor([c.cellar_pos[0] + c.hinge_local[0],
                          c.cellar_pos[1] + c.hinge_local[1], c.hinge_local[2]],
                         device=self.env.device)
        return h.expand(self.env.num_envs, 3)

    def plat_z(self) -> torch.Tensor:
        """(N,) platform tray-centre height above its env origin."""
        return self.platform.data.root_pos_w[:, 2] - self.env_origins[:, 2]

    def pawl_angle(self) -> torch.Tensor:
        """(N,) pawl hinge angle in rad (0 = horizontal, negative = folded down). The
        pawl only ever rotates about the hinge +y axis, so the root quat is
        (cos t/2, 0, sin t/2, 0)."""
        q = self.pawl.data.root_quat_w
        return (2.0 * torch.atan2(q[:, 2], q[:, 0])).clamp(min=-math.pi, max=math.pi)

    def _load_point_world(self) -> torch.Tensor:
        """(N, 3) world position of the tray load point at the CURRENT platform pose."""
        c = self.cfg
        off = torch.tensor([c.load_xy[0], c.load_xy[1],
                            c.tray_top_dz + c.butter_size[2] / 2],
                           device=self.env.device)
        return self.platform.data.root_pos_w + off.expand(self.env.num_envs, 3)

    # ----- predicates ----------------------------------------------------------------------------
    def on_tray(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre over the tray footprint, resting just above the tray
        top plane — judged RELATIVE to the live platform pose, so it holds at any
        platform height."""
        c = self.cfg
        rel = body.data.root_pos_w - self.platform.data.root_pos_w
        dz = rel[:, 2] - c.tray_top_dz
        return ((rel[:, 0].abs() <= c.on_tray_xy) & (rel[:, 1].abs() <= c.on_tray_xy)
                & (dz > c.on_tray_dz_lo) & (dz < c.on_tray_dz_hi))

    def in_latch_band(self) -> torch.Tensor:
        """(N,) bool: platform tray centre inside the latch band below the pawl."""
        c = self.cfg
        z = self.plat_z()
        return (z > c.latch_z_lo) & (z < c.latch_z_hi)

    def pawl_seated(self) -> torch.Tensor:
        c = self.cfg
        return self.pawl_angle().abs() <= math.radians(c.pawl_closed_deg)

    def ingot_clear(self) -> torch.Tensor:
        """(N,) bool: ingot back on the floor, clear of the silo footprint."""
        c = self.cfg
        rel = self.ingot.data.root_pos_w[:, 0:2] - self._cellar_xy()
        far = rel.abs().max(dim=-1).values > c.ingot_clear_xy
        low = (self.ingot.data.root_pos_w[:, 2] - self.env_origins[:, 2]) < c.ingot_clear_z
        return far & low

    def in_bore(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre inside the silo bore column, below the rim."""
        c = self.cfg
        rel = body.data.root_pos_w[:, 0:2] - self._cellar_xy()
        z = body.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (rel.abs() < c.bore / 2).all(dim=-1) & (z < c.rim_h + 0.02)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cellar re-asserted at its fixed pose, platform re-posed to the
        top stop and pawl to horizontal (pure follower-only writes about the unchanged
        joint frames — the proven safe articulated re-pose), free blocks randomly
        PERMUTED over the three floor spawn slots (+ xy jitter + free yaw), latches
        cleared, per-episode approach baseline recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- cellar (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = c.cellar_pos
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.cellar.write_root_state_to_sim(st, env_ids)

        # --- platform: top stop, identity orientation (follower-only) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.cellar_pos[0], c.cellar_pos[1], c.z_top
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.platform.write_root_state_to_sim(st, env_ids)

        # --- pawl: horizontal at the 0-deg stop (follower-only) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self._hinge_world()[env_ids] + origin
        st[:, 3] = 1.0
        self.pawl.write_root_state_to_sim(st, env_ids)

        # --- free blocks: random permutation over the three spawn slots ---
        slots = torch.tensor(c.spawn_slots, device=dev)  # (3, 2)
        if c.permute_slots:
            perms = torch.tensor(
                [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
                device=dev)
            assign = perms[torch.randint(0, 6, (m,), device=dev)]  # (m, 3)
        else:
            assign = torch.tensor([[0, 1, 2]], device=dev).expand(m, 3)
        heights = (c.butter_size[2] / 2, c.butter_size[2] / 2, c.ingot_size[2] / 2)
        butter_xy = None
        for k, (body, hz) in enumerate(((self.butter, heights[0]),
                                        (self.lard, heights[1]),
                                        (self.ingot, heights[2]))):
            xy = slots[assign[:, k]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
            if k == 0:
                butter_xy = xy
            half = ((torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
                    if c.yaw_free else torch.zeros(m, device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = hz + 0.002
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- approach baseline: butter spawn -> tray load point (null scores exactly 0) ---
        lp = torch.tensor([c.cellar_pos[0] + c.load_xy[0], c.cellar_pos[1] + c.load_xy[1],
                           c.z_top + c.tray_top_dz + c.butter_size[2] / 2],
                          device=dev).expand(m, 3)
        spawn = torch.cat([butter_xy,
                           torch.full((m, 1), c.butter_size[2] / 2 + 0.002, device=dev)],
                          dim=1)
        self._d_init[env_ids] = (spawn - lp).norm(dim=-1).clamp(min=0.05)

        # --- clear latches ---
        self._app_max[env_ids] = 0.0
        self._loaded[env_ids] = False
        self._min_z[env_ids] = c.z_top
        self._latched[env_ids] = False
        self._cleared[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cellar": self.cellar.data.root_state_w[env_ids].clone(),
            "platform": self.platform.data.root_state_w[env_ids].clone(),
            "pawl": self.pawl.data.root_state_w[env_ids].clone(),
            "butter": self.butter.data.root_state_w[env_ids].clone(),
            "lard": self.lard.data.root_state_w[env_ids].clone(),
            "ingot": self.ingot.data.root_state_w[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "min_z": self._min_z[env_ids].clone(),
            "latched": self._latched[env_ids].clone(),
            "cleared": self._cleared[env_ids].clone(),
            "d_init": self._d_init[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cellar.write_root_state_to_sim(state["cellar"], env_ids)
        self.platform.write_root_state_to_sim(state["platform"], env_ids)
        self.pawl.write_root_state_to_sim(state["pawl"], env_ids)
        self.butter.write_root_state_to_sim(state["butter"], env_ids)
        self.lard.write_root_state_to_sim(state["lard"], env_ids)
        self.ingot.write_root_state_to_sim(state["ingot"], env_ids)
        self._app_max[env_ids] = state["app_max"]
        self._loaded[env_ids] = state["loaded"]
        self._min_z[env_ids] = state["min_z"]
        self._latched[env_ids] = state["latched"]
        self._cleared[env_ids] = state["cleared"]
        self._d_init[env_ids] = state["d_init"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-blue open-topped SILO (the butter cellar) stands on the floor: a "
            f"{(c.bore + 2 * c.wall_t) * 100:.0f} cm square shaft, {c.rim_h * 100:.0f} cm "
            f"tall, with a {c.bore * 100:.0f} cm square vertical bore. Inside the bore a "
            f"light-wood TRAY rides a vertical slide; a spring holds it up at its top "
            f"stop, {(c.rim_h - c.z_top - c.tray_t / 2) * 100:.0f} cm below the rim, where "
            f"it blocks the deep interior. From the tray's near corner a steel PLUNGER "
            f"POST rises well above the rim, ending in a shallow square SOCKET (a "
            f"{c.cap_w * 100:.0f} cm cap with a raised lip). A brass RATCHET PAWL sticks "
            f"out of a channel in the far bore wall, {c.hinge_local[2] * 100:.0f} cm above "
            f"the ground: it folds DOWNWARD out of the way when the tray is pressed past "
            f"it, but cannot swing above horizontal, so a rising tray jams underneath it "
            f"— pressing the platform below the pawl is IRREVERSIBLE. On the floor near "
            f"the silo lie three loose items whose positions shuffle between episodes — "
            f"identify by COLOR: a YELLOW butter stick, a WHITE lard block of identical "
            f"shape, and a heavy DARK-IRON INGOT ({c.ingot_size[2] * 100:.0f} cm tall).\n"
            f"Goal: stash the YELLOW butter deep in the cellar. Place the butter flat on "
            f"the raised tray (drop it on the open tray area away from the post and the "
            f"pawl side), then drive the platform down its slide until the tray passes "
            f"below the brass pawl, and let the spring seat the tray up against the "
            f"pawl's underside. The intended press: set the IRON INGOT into the plunger's "
            f"socket — its weight overpowers the spring and lowers the platform; a "
            f"steady downward push on the socket cap works too. Finally the ingot must "
            f"END on the floor, clear of the silo (lift it out of the socket and set it "
            f"down away from the walls) — a press weight left in the socket does not "
            f"count. The WHITE lard must remain OUTSIDE the silo. Success: the butter "
            f"lying settled on the tray while the platform is latched under the pawl "
            f"(butter roughly {(c.rim_h - c.z_latch_eq) * 100:.0f} cm below the rim), the "
            f"ingot on the floor away from the silo, the lard outside, everything at "
            f"rest. Dropping the butter down the bore without ever pressing the platform "
            f"past the pawl, pressing an EMPTY platform, latching the lard instead, "
            f"balancing the butter on the socket cap, or leaving the ingot in the socket "
            f"is failure. Loading the butter first and then pressing is the reliable "
            f"order (a butter dropped down the bore after latching may catch on the "
            f"pawl ledge)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the yellow butter stick on the raised tray inside the blue silo, "
            "then press the platform down past the brass ratchet pawl until it latches "
            "deep in the silo — the iron ingot dropped into the plunger's socket makes "
            "a good press weight. Finish with the ingot back on the floor clear of the "
            "silo and the white lard block left outside."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        d = (self.butter.data.root_pos_w - self._load_point_world()).norm(dim=-1)
        app = (1.0 - d / self._d_init).clamp(0.0, 1.0)
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        butter_still = self.butter.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._loaded |= self.on_tray(self.butter) & butter_still
        self._min_z = torch.minimum(self._min_z, self.plat_z())
        plat_still = self.platform.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        latch_now = self.in_latch_band() & self.pawl_seated() & plat_still
        self._latched |= latch_now
        self._cleared |= self._latched & self.ingot_clear()

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Mechanism plant, owned by the scene in every module equally: a constant
        upward spring force + viscous damping on the platform slide, and a weak
        return-spring torque + damping on the pawl hinge (its gravity is disabled).
        Then latch rubric progress. Owns both bodies' external-wrench slots."""
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs

        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 2] = c.spring_f - c.spring_c * self.platform.data.root_lin_vel_w[:, 2]
        self.platform.set_external_force_and_torque(
            f, torch.zeros(n, 1, 3, device=dev))

        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 1] = (-c.pawl_k * self.pawl_angle()
                       - c.pawl_c * self.pawl.data.root_ang_vel_w[:, 1])
        self.pawl.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tq)

        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: butter settled ON the tray, platform held in the latch band with
        the pawl seated horizontal, ingot back on the floor clear of the silo, lard not
        in the bore, platform and butter at rest. Physical outcomes only — the latch
        band is a live pose readback that the spring makes unreachable except under the
        pawl (or under an external load, which the ingot-clear clause rejects)."""
        c = self.cfg
        self._update_latches()
        butter_still = self.butter.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        plat_still = self.platform.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return (self.on_tray(self.butter) & butter_still
                & self.in_latch_band() & self.pawl_seated() & plat_still
                & self.ingot_clear() & ~self.in_bore(self.lard))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*load-point approach (vs the episode's own spawn
        distance) + 0.20*loaded + 0.25*press-depth progress + 0.20*latched +
        0.10*ingot-cleared — all latched/rising-only, exactly 0 for doing nothing,
        capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        depth = ((c.z_top - self._min_z) / (c.z_top - c.z_latch_eq)).clamp(0.0, 1.0)
        base = (c.w_app * self._app_max + c.w_load * self._loaded.float()
                + c.w_depth * depth + c.w_latch * self._latched.float()
                + c.w_clear * self._cleared.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="butter_cellar", robot="null"))
