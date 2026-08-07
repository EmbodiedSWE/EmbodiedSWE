"""PileDriverScene — drive the RED post flush with the rig deck by repeated gravity
impacts of a carried steel slug (sim_gen task `libero_pick_milk_i51`).

Derived from libero/libero_pick_milk — "pick the milk and place it in the basket": one
target among look-alike distractors, one grasp-carry-lower, one bounding-box
containment readout. STRATEGICALLY different: nothing here is delivered anywhere.
The judged interaction is IMPULSE against DRY FRICTION — a physical regime the seed
(and every corpus task read, see TASK.md) never touches. A driver rig holds two square
posts (one RED, one WHITE) sliding in vertical friction clamps: each post is squeezed
between two spring-loaded high-mu brake pads whose static grip (~200 N of tangential
force-closure per post — forge-CALIBRATED: PhysX patch friction saturates sublinearly
in normal force, so the design derates by a measured clamp_eff and buys its margin
with brake-pad-grade friction coefficients instead) is far above anything
a manipulator arm can press quasi-statically (~90 N), and 10x above the resting
weight of the slug standing on a post head. Steady pushing therefore does NOT move a
post — smoke proves a sustained 90 N push advances it < 4 mm — and gravity alone
never moves it (null policy scores ~0). The ONLY way to sink a post is to convert
carried potential energy into contact impulse: pick up the 1.2 kg slug, drop it on
the post head from a hand-height of ~10-20 cm, and let the collision's force spike
(orders of magnitude above the static grip) break the friction threshold for a few
milliseconds. Each blow buys ~4-10 mm of travel; the clamp re-latches the new depth
the instant the impulse decays (no creep, no spring-back — progress is physically
ratcheted). Around ten blows sink the red head flush with the deck. The deck itself is the depth gauge and the overdrive guard: the
slug face is wider than the deck aperture, so the final blow bottoms the slug on the
deck with the head flush, and the joint's hard stop sits at the flush band's lower
edge — the tolerance cannot be overshot.

Goal state (success()): RED post head flush with the deck (proud height within
[flush_lo, flush_hi] of the deck top), WHITE post undisturbed (within white_tol of
its episode-start proud height — driving it is physically IRREVERSIBLE, the rubric
never forgives it), the slug parked back in its yellow floor holster, and everything
settled. score(): latched, monotone along the demonstrated solution — 0.08 the slug
ever brought over the rig + 0.52 x best drive fraction of the red post (latched
maximum) + 0.15 flush-achieved latch (red flush while white still undisturbed);
1.0 iff success(). Null policy ~0 (holster spawns beyond the near radius; posts do
not move without impacts).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - rig (dynamic 40 kg fixture — NEVER kinematic, so every joint anchor follows the
    reset teleport): ground slab, two columns, a gray DECK plate at 0.26 m with two
    square apertures; the two posts rise through the apertures.
  - posts (RED target / WHITE decoy, 40 mm square, 0.25 kg): each on a vertical
    PrismaticJoint into the rig. Upper limit ~0 (a post can never be pulled UP);
    lower limit = the flush band's lower edge (hard stop against overdrive).
  - brake pads (4x dark blocks): each on a horizontal PrismaticJoint into the rig
    with a stiff linear drive pressing it into its post's face — a spring-loaded
    friction clamp; grip normal ~319 N per pad, high-friction material both sides.
    Joint pairs (pad-rig, post-rig) stay collision-FILTERED (USD default): the posts
    are guided by their joints, braked by REAL pad contact.
  - slug (steel-gray 1.2 kg block, 72 mm square face, with a 30 mm grip stud on top
    — a natural parallel-jaw pinch): the drop hammer. The only free tool body.
  - holster (yellow walled tray, heavy): the slug's home; slug must END here.

Per-episode randomization (readback-verified in smoke): rig xy jitter + FREE yaw,
per-post initial proud height (red and white independently in [proud_lo, proud_hi]),
holster on a jittered world arc (angle + radius + yaw) with the slug standing in it.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
A_POST = 0.040  # post square cross-section
L_POST = 0.220  # post length
Y_POST = 0.090  # post lateral offset in the rig frame (one at +y, one at -y)
DECK_TOP = 0.260  # deck plate TOP surface height (rig frame z) — the flush reference
DECK_T = 0.020  # deck plate thickness
HOLE_HALF = 0.023  # deck aperture half-width (46 mm — post 40 mm + visual clearance)
PROUD_SPAWN = 0.090  # spawn proud height (head top above deck); joint zero reference
TRAVEL = 0.098  # prismatic lower limit magnitude -> head can reach 8 mm below deck
BASE_T = 0.024  # rig ground slab thickness
PAD_T = 0.020  # brake pad thickness (x)
PAD_W = 0.050  # brake pad width (y)
PAD_H = 0.060  # brake pad height (z)
PAD_Z = 0.160  # brake pad center height (always inside the post's z-span)
PAD_GAP = 0.0005  # spawn gap pad-face -> post-face (crushed by the drive)
SLUG_S = 0.072  # slug square face (wider than the deck aperture: bottoms on the deck)
SLUG_H = 0.060  # slug body height
KNOB_S = 0.030  # grip stud square (parallel-jaw pinch)
KNOB_H = 0.050  # grip stud height
HOL_S = 0.120  # holster plate square
HOL_T = 0.010  # holster plate thickness
HOL_RIM_H = 0.030  # holster rim height
HOL_RIM_T = 0.008  # holster rim thickness


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- custom compound spawners ----------------------------------------------------------------
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
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                *, iters: int = 16, vel_iters: int = 1, damp: float = 0.0):
    """Author a dynamic compound-body root with zeroed sleep (a sleeping fixture
    would freeze its joint anchors)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(1.0)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    if damp:
        pxrb.CreateLinearDampingAttr(float(damp))
        pxrb.CreateAngularDampingAttr(float(damp))
    return root


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The driver rig: ground slab + two columns + deck plate with two square
    apertures. One heavy DYNAMIC compound body (anchors follow reset teleports).
    Origin = rig center on the ground; posts rise at (0, +/-Y_POST)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       damp=0.5)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    gray = (0.42, 0.45, 0.48)
    dark = (0.28, 0.30, 0.33)
    # ground slab
    _box(stage, f"{prim_path}/base", (0.20, 0.42, BASE_T), (0.0, 0.0, BASE_T / 2),
         dark, 0.002, material=body)
    # columns (support the deck at the y-ends, clear of the posts)
    for tag, sy in (("n", 1.0), ("s", -1.0)):
        _box(stage, f"{prim_path}/col_{tag}", (0.14, 0.030, 0.216),
             (0.0, sy * 0.175, BASE_T + 0.108), gray, 0.002, material=body)
    # deck plate (top = DECK_TOP) with two HOLE_HALF square apertures at y=+/-Y_POST
    zc = DECK_TOP - DECK_T / 2
    _box(stage, f"{prim_path}/deck_w", (0.047, 0.38, DECK_T), (-0.0465, 0.0, zc),
         gray, 0.004, material=body)
    _box(stage, f"{prim_path}/deck_e", (0.047, 0.38, DECK_T), (0.0465, 0.0, zc),
         gray, 0.004, material=body)
    _box(stage, f"{prim_path}/deck_mid", (0.046, 2 * (Y_POST - HOLE_HALF), DECK_T),
         (0.0, 0.0, zc), gray, 0.004, material=body)
    for tag, sy in (("n", 1.0), ("s", -1.0)):
        y0 = Y_POST + HOLE_HALF
        _box(stage, f"{prim_path}/deck_{tag}", (0.046, 0.19 - y0, DECK_T),
             (0.0, sy * (y0 + (0.19 - y0) / 2), zc), gray, 0.004, material=body)
    return root


def _spawn_post(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One square post + its vertical PrismaticJoint into the sibling rig. Origin =
    post center. Upper limit ~0 (never pulled up past spawn), lower limit -TRAVEL
    (the flush band's hard floor). NOTE: physxJoint:jointFriction is NOT used — it
    is silently ignored for joints outside articulations; the brake is pure pad
    contact friction with high-mu pad material. 4 velocity iterations (the TGS max
    before the changed >4 behavior) kill the reset-settle friction creep."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       vel_iters=4)
    mat = _friction_material(stage, f"{prim_path}/post_mat", cfg.mu_post_s, cfg.mu_post_d)
    _box(stage, f"{prim_path}/body", (A_POST, A_POST, L_POST), (0.0, 0.0, 0.0),
         cfg.color, 0.008, material=mat)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slide")
    j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    # anchor at the post's SPAWN center in the rig frame
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(cfg.y_side * Y_POST),
                                   float(DECK_TOP + PROUD_SPAWN - L_POST / 2)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-TRAVEL))
    j.CreateUpperLimitAttr(0.0005)
    return root


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One brake pad + its horizontal PrismaticJoint into the sibling rig, with a
    stiff linear drive pressing the pad face into the post face (the friction
    clamp's spring). Origin = pad center; presses along the rig x axis."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       vel_iters=4)
    mat = _friction_material(stage, f"{prim_path}/pad_mat", cfg.mu_pad_s, cfg.mu_pad_d)
    _box(stage, f"{prim_path}/body", (PAD_T, PAD_W, PAD_H), (0.0, 0.0, 0.0),
         (0.16, 0.17, 0.20), 0.002, material=mat)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/press")
    j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    x0 = cfg.x_side * (A_POST / 2 + PAD_T / 2 + PAD_GAP)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(x0), float(cfg.y_side * Y_POST), float(PAD_Z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # limits in the press direction (toward the post = -x_side); wide enough that
    # the drive target always sits INSIDE the limit range
    if cfg.x_side > 0:
        j.CreateLowerLimitAttr(float(-0.018))
        j.CreateUpperLimitAttr(0.002)
    else:
        j.CreateLowerLimitAttr(-0.002)
        j.CreateUpperLimitAttr(float(0.018))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drv.CreateTypeAttr("force")
    drv.CreateStiffnessAttr(float(cfg.clamp_k))
    drv.CreateDampingAttr(float(cfg.clamp_c))
    drv.CreateTargetPositionAttr(float(-cfg.x_side * cfg.clamp_target))
    return root


def _spawn_slug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The drop hammer: steel block + grip stud, one free dynamic body."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       damp=0.05)
    mat = _friction_material(stage, f"{prim_path}/slug_mat", cfg.mu_slug_s, cfg.mu_slug_d)
    steel = (0.52, 0.55, 0.60)
    _box(stage, f"{prim_path}/body", (SLUG_S, SLUG_S, SLUG_H), (0.0, 0.0, 0.0),
         steel, 0.008, material=mat)
    _box(stage, f"{prim_path}/knob", (KNOB_S, KNOB_S, KNOB_H),
         (0.0, 0.0, SLUG_H / 2 + KNOB_H / 2), (0.75, 0.20, 0.12), 0.002, material=mat)
    return root


def _spawn_holster(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The slug's home: a yellow walled tray, heavy dynamic. Origin = plate center
    on the ground."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _rigid_root(stage, prim_path, translation, orientation, cfg.mass_props.mass,
                       damp=0.5)
    mat = _friction_material(stage, f"{prim_path}/hol_mat", 0.5, 0.45)
    yellow = (0.90, 0.78, 0.10)
    _box(stage, f"{prim_path}/plate", (HOL_S, HOL_S, HOL_T), (0.0, 0.0, HOL_T / 2),
         yellow, 0.002, material=mat)
    for tag, sx, sy in (("e", 1.0, 0.0), ("w", -1.0, 0.0),
                        ("n", 0.0, 1.0), ("s", 0.0, -1.0)):
        # e/w rims: thin in x, long in y; n/s rims: long in x, thin in y.
        size = (HOL_RIM_T, HOL_S, HOL_RIM_H) if sx else (HOL_S, HOL_RIM_T, HOL_RIM_H)
        off = (HOL_S / 2 - HOL_RIM_T / 2)
        _box(stage, f"{prim_path}/rim_{tag}", size,
             (sx * off, sy * off, HOL_T + HOL_RIM_H / 2), yellow, 0.002, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazy: module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            mu_body_s: float = 0.60
            mu_body_d: float = 0.55

        @configclass
        class PostSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_post)
            y_side: float = 1.0
            color: tuple = (0.85, 0.10, 0.10)
            mu_post_s: float = 1.30
            mu_post_d: float = 1.20

        @configclass
        class PadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            y_side: float = 1.0
            x_side: float = 1.0
            clamp_k: float = 22000.0
            clamp_c: float = 250.0
            clamp_target: float = 0.015
            mu_pad_s: float = 1.30
            mu_pad_d: float = 1.20

        @configclass
        class SlugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slug)
            mu_slug_s: float = 0.40
            mu_slug_d: float = 0.35

        @configclass
        class HolsterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_holster)

        _SPAWNER_CACHE.update(rig=RigSpawnerCfg, post=PostSpawnerCfg, pad=PadSpawnerCfg,
                              slug=SlugSpawnerCfg, holster=HolsterSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PileDriverSceneCfg(BaseCfg):
    """Config for `PileDriverScene`. The honesty knobs are asserted in
    `__post_init__`: the clamp's static grip sits far above arm-scale quasi-static
    pushing and the resting tool weight (impacts are the ONLY way down), a single
    hand-height drop buys a useful but bounded advance, and the joint's hard stop
    coincides with the flush band's lower edge (no overdrive lottery)."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    flush_lo: float = tunable(-0.008)  # red head proud height lower bound (= joint stop)
    flush_hi: float = tunable(0.012)  # ... upper bound ("flush" = within 12 mm)
    white_tol: float = tunable(0.012)  # white post must stay within this of its start
    park_xy: float = tunable(0.040)  # slug center within this of the holster center
    park_z_win: tuple = tunable((-0.004, 0.020))  # slug bottom minus plate top
    park_up_min: float = tunable(0.85)  # slug local +z world-up cosine (resting flat)
    settle_lin: float = tunable(0.05)  # settle gates (m/s)
    settle_ang: float = tunable(1.0)  # (rad/s)
    near_r: float = tunable(0.22)  # score: "slug brought over the rig" xy radius

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    rig_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the rig at reset (m)
    rig_yaw_max: float = tunable(180.0)  # uniform +/- rig yaw (deg; FREE heading)
    proud_lo: float = tunable(0.060)  # per-post initial proud height range (m);
    proud_hi: float = tunable(0.078)  # the settle transient LIFTS a post ~settle_lift
    # above its written height (forge-measured pad-squeeze artifact), so the band
    # sits low enough that settled heights stay clear of the joint's upper stop
    # (PROUD_SPAWN + 0.5 mm) and the per-episode spread survives.
    hol_slot: float = tunable(150.0)  # holster world-frame arc slot center (deg)
    hol_jitter: float = tunable(80.0)  # uniform +/- arc-angle jitter (deg)
    hol_radius: float = tunable(0.33)  # arc radius around the rig (m)
    hol_rad_jitter: float = tunable(0.03)

    # --- info: structure -----------------------------------------------------------------------
    rig_mass: float = info(40.0)  # heavy dynamic fixture (joint anchors follow teleports)
    post_mass: float = info(0.25)
    pad_mass: float = info(0.08)
    slug_mass: float = info(1.2)
    hol_mass: float = info(3.0)
    clamp_k: float = info(22000.0)  # pad drive stiffness (N/m)
    clamp_c: float = info(250.0)  # pad drive damping
    clamp_target: float = info(0.015)  # pad drive crush target (m) -> ~319 N normal
    clamp_eff: float = info(0.24)  # MEASURED on the forge: PhysX patch friction
    # saturates sublinearly in NORMAL force — at this ~320 N normal it delivers
    # ~24% of the rigid-model tangential capacity (it stays linear in mu, which is
    # why the brake pads use a high-mu material). Every brake number below is
    # derated by this factor so the honesty asserts bind to what the sim does.
    mu_post_s: float = info(1.30)  # post/pad faces (high-mu brake-pad pair)
    mu_post_d: float = info(1.20)
    mu_slug_s: float = info(0.40)
    mu_slug_d: float = info(0.35)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)
    arm_push_ref: float = info(90.0)  # reference sustained arm push (N) the clamp must beat
    settle_lift: float = info(0.012)  # forge-MEASURED: (re)forming the pad squeeze
    # after a post state write lifts the post ~8-16 mm before the clamp latches;
    # proud0 re-latches from readback during the warmup window, so the rubric
    # always references the settled height.

    # Derived (filled in __post_init__).
    grip_normal: float = field(default=None, init=False)  # per-pad normal force (N)
    brake_static: float = field(default=None, init=False)  # static tangential grip (N)
    brake_dynamic: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.grip_normal = self.clamp_k * (self.clamp_target - PAD_GAP)
        # Effective (derated) brake forces — what the sim actually delivers.
        self.brake_static = self.clamp_eff * 2.0 * self.mu_post_s * self.grip_normal
        self.brake_dynamic = self.clamp_eff * 2.0 * self.mu_post_d * self.grip_normal
        # -- embodiment: the slug is graspable, everything within one base pose's reach --
        assert KNOB_S < 0.05, "grip stud must be a comfortable parallel-jaw pinch"
        assert SLUG_S < 0.078, "slug body must fit an ~80 mm jaw as a fallback grasp"
        assert DECK_TOP + PROUD_SPAWN + 0.25 < 0.70, \
            "drop hover must stay at a comfortable Franka height"
        # -- impacts are the ONLY way down --
        rest_w = (self.slug_mass + self.post_mass) * 9.81
        assert self.brake_static > 8.0 * rest_w, \
            f"static grip {self.brake_static:.0f} N must dwarf the resting tool weight"
        assert self.brake_static > 1.5 * self.arm_push_ref, \
            f"static grip {self.brake_static:.0f} N must beat a sustained arm push"
        # -- one hand-height drop buys a useful but bounded advance --
        ke = self.slug_mass * 9.81 * 0.14 * self.slug_mass / (self.slug_mass + self.post_mass)
        adv = ke / self.brake_dynamic
        assert 0.004 < adv < 0.035, f"per-drop advance estimate {adv * 1000:.1f} mm out of band"
        # -- the flush band is honest: hard stop at its lower edge, band inside travel --
        assert abs(-(TRAVEL - PROUD_SPAWN) - self.flush_lo) < 1e-9, \
            "joint hard stop must coincide with the flush band's lower edge"
        assert self.flush_hi > self.flush_lo + 0.010, "flush band must be >= 10 mm wide"
        assert self.proud_hi < PROUD_SPAWN, "reset proud height must sit below the joint zero"
        assert self.proud_lo > self.flush_hi + 0.040, \
            "posts must start far outside the flush band"
        # -- the slug face gauges the deck: wider than the aperture --
        assert SLUG_S > 2 * HOLE_HALF + 0.015, "slug face must bottom out on the deck"
        # -- null policy scores 0: holster spawns beyond the near-credit radius --
        assert self.hol_radius - self.hol_rad_jitter > self.near_r + 0.05, \
            "holster arc must start beyond the near-credit radius"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pile_driver")
class PileDriverScene(BaseScene):
    cfg: PileDriverSceneCfg

    def __init__(self, cfg: PileDriverSceneCfg | None = None) -> None:
        super().__init__(cfg or PileDriverSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        mass = sim_utils.MassPropertiesCfg
        rigid = sim_utils.RigidBodyPropertiesCfg

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: the rig MUST spawn before posts/pads (their joints target it).
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](mass_props=mass(mass=c.rig_mass), rigid_props=rigid()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
        }
        post_z = DECK_TOP + PROUD_SPAWN - L_POST / 2
        for name, y_side, color in (("post_red", 1.0, (0.85, 0.10, 0.10)),
                                    ("post_white", -1.0, (0.92, 0.92, 0.92))):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post_" + name.split("_")[1],
                spawn=sp["post"](mass_props=mass(mass=c.post_mass), rigid_props=rigid(),
                                 y_side=y_side, color=color,
                                 mu_post_s=c.mu_post_s, mu_post_d=c.mu_post_d),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, y_side * Y_POST, post_z)),
            )
        for name, y_side, x_side in (("pad_rn", 1.0, -1.0), ("pad_rp", 1.0, 1.0),
                                     ("pad_wn", -1.0, -1.0), ("pad_wp", -1.0, 1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad_" + name.split("_")[1],
                spawn=sp["pad"](mass_props=mass(mass=c.pad_mass), rigid_props=rigid(),
                                y_side=y_side, x_side=x_side,
                                clamp_k=c.clamp_k, clamp_c=c.clamp_c,
                                clamp_target=c.clamp_target,
                                mu_pad_s=c.mu_post_s, mu_pad_d=c.mu_post_d),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x_side * (A_POST / 2 + PAD_T / 2 + PAD_GAP),
                         y_side * Y_POST, PAD_Z)),
            )
        out["holster"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Holster",
            spawn=sp["holster"](mass_props=mass(mass=c.hol_mass), rigid_props=rigid()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.25, 0.25, 0.0)),
        )
        out["slug"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Slug",
            spawn=sp["slug"](mass_props=mass(mass=c.slug_mass), rigid_props=rigid(),
                             mu_slug_s=c.mu_slug_s, mu_slug_d=c.mu_slug_d),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.25, 0.25, HOL_T + SLUG_H / 2 + 0.002)),
        )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 240.0,
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
        for nm in ("rig", "post_red", "post_white", "pad_rn", "pad_rp", "pad_wn",
                   "pad_wp", "holster", "slug"):
            setattr(self, nm, env.iscene[nm])
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.proud0_red = torch.full((n,), self.cfg.proud_hi, device=dev)
        self.proud0_white = torch.full((n,), self.cfg.proud_hi, device=dev)
        # post-reset warmup countdown (steps): while > 0, proud0 re-latches from
        # readback (absorbing any settle transient) and score latches stay frozen.
        self.warmup = torch.zeros(n, dtype=torch.long, device=dev)
        # latched score stages (updated every physics substep in post_step)
        self.ever_near = torch.zeros(n, dtype=torch.bool, device=dev)
        self.best_drive = torch.zeros(n, device=dev)
        self.flush_latch = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the rig (heavy DYNAMIC teleport — every
        joint anchor follows), write posts at their sampled proud heights and pads at
        their rest poses IN THE RIG'S NEW FRAME, drop the holster on a jittered world
        arc with the slug standing in it, and clear the score latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- rig: xy jitter + free yaw ---
        rxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rig_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_max)
        q_rig = _qz(yaw)
        zeros = torch.zeros(m, 1, device=dev)
        write(self.rig, torch.cat([rxy, zeros], dim=-1), q_rig)

        # --- posts at sampled proud heights + pads at rest, all in the rig's frame ---
        proud_r = c.proud_lo + torch.rand(m, device=dev) * (c.proud_hi - c.proud_lo)
        proud_w = c.proud_lo + torch.rand(m, device=dev) * (c.proud_hi - c.proud_lo)
        self.proud0_red[env_ids] = proud_r
        self.proud0_white[env_ids] = proud_w
        for body, y_side, proud in ((self.post_red, 1.0, proud_r),
                                    (self.post_white, -1.0, proud_w)):
            local = torch.stack([torch.zeros(m, device=dev),
                                 torch.full((m,), y_side * Y_POST, device=dev),
                                 DECK_TOP + proud - L_POST / 2], dim=-1)
            write(body, torch.cat([rxy, zeros], dim=-1) + _qapply(q_rig, local), q_rig)
        # pads at ZERO-GAP contact equilibrium (face touching the post face): the
        # drive is at its steady-state stretch immediately — no slam-in transient
        # that would hammer the posts upward during the settle.
        for body, y_side, x_side in ((self.pad_rn, 1.0, -1.0), (self.pad_rp, 1.0, 1.0),
                                     (self.pad_wn, -1.0, -1.0), (self.pad_wp, -1.0, 1.0)):
            local = torch.tensor([x_side * (A_POST / 2 + PAD_T / 2),
                                  y_side * Y_POST, PAD_Z], device=dev).expand(m, 3)
            write(body, torch.cat([rxy, zeros], dim=-1) + _qapply(q_rig, local), q_rig)

        # --- holster on a jittered world arc, slug standing in it ---
        ang = math.radians(c.hol_slot) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hol_jitter)
        rad = c.hol_radius + (torch.rand(m, device=dev) * 2 - 1) * c.hol_rad_jitter
        hxy = rxy + torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=-1)
        hyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.holster, torch.cat([hxy, zeros], dim=-1), _qz(hyaw))
        slug_pos = torch.cat([hxy, torch.full((m, 1), HOL_T + SLUG_H / 2 + 0.002,
                                              device=dev)], dim=-1)
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.slug, slug_pos, _qz(syaw))

        # --- clear the score latches; arm the warmup re-latch window ---
        self.ever_near[env_ids] = False
        self.best_drive[env_ids] = 0.0
        self.flush_latch[env_ids] = False
        self.warmup[env_ids] = 60

    # ----- state (full, restorable) -----------------------------------------------------------
    _BODIES = ("rig", "post_red", "post_white", "pad_rn", "pad_rp", "pad_wn", "pad_wp",
               "holster", "slug")

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
               for nm in self._BODIES}
        out["proud0_red"] = self.proud0_red[env_ids].clone()
        out["proud0_white"] = self.proud0_white[env_ids].clone()
        out["warmup"] = self.warmup[env_ids].clone()
        out["ever_near"] = self.ever_near[env_ids].clone()
        out["best_drive"] = self.best_drive[env_ids].clone()
        out["flush_latch"] = self.flush_latch[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in self._BODIES:
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        self.proud0_red[env_ids] = state["proud0_red"]
        self.proud0_white[env_ids] = state["proud0_white"]
        self.warmup[env_ids] = state["warmup"]
        self.ever_near[env_ids] = state["ever_near"]
        self.best_drive[env_ids] = state["best_drive"]
        self.flush_latch[env_ids] = state["flush_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray pile-driver rig stands on the floor: a heavy base carrying a flat "
            f"DECK plate {DECK_TOP * 100:.0f} cm up, with two square steel posts "
            f"({A_POST * 1000:.0f} mm across) rising side by side through apertures in "
            f"the deck — one post RED, one WHITE, each sticking up roughly "
            f"{(c.proud_lo + c.settle_lift) * 100:.0f}-"
            f"{(c.proud_hi + c.settle_lift) * 100:.0f} cm above the deck. Below "
            f"the deck each post runs through a spring-loaded FRICTION CLAMP: the "
            f"posts slide only vertically, and the clamp grips with about "
            f"{c.brake_static:.0f} N of static friction — steady pushing on a post "
            f"head, even leaning on it with the whole arm, will not move it, and a "
            f"post never springs back up (its guide cannot rise past its start). "
            f"Sharp IMPACTS are different: a collision's force spike breaks the "
            f"friction for a few milliseconds and the post sinks a little, then the "
            f"clamp holds the new depth. On the floor nearby, a steel SLUG "
            f"({SLUG_S * 1000:.0f} mm square, {c.slug_mass:.1f} kg) with a narrow red "
            f"grip stud on top stands in a YELLOW walled holster tray. The rig's "
            f"position and heading, both posts' initial heights, and the holster's "
            f"position change every episode — read the scene by looking.\n"
            f"Goal: drive the RED post flush with the deck — its head top within "
            f"about {c.flush_hi * 1000:.0f} mm of the deck surface — by striking it "
            f"with the slug: lift the slug by its stud, hold it 10-20 cm above the "
            f"red head, and DROP it (or strike downward and release); repeat, each "
            f"blow sinks the post 4-10 mm. The slug's face is wider than the deck "
            f"aperture, so it bottoms on the deck at flush depth — you cannot "
            f"overdrive. Do NOT strike the WHITE post: it must stay within "
            f"{c.white_tol * 1000:.0f} mm of its initial height, and driving it is "
            f"IRREVERSIBLE — the task cannot be recovered. Finish by standing the "
            f"slug back in the yellow holster and leaving everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the steel slug from the yellow holster and drive the RED post "
            "flush with the rig deck by repeatedly dropping the slug on its head — "
            "steady pushing cannot move the post. Never strike the white post. When "
            "the red head sits flush, stand the slug back in the holster."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> rig frame."""
        return _qapply(_qinv(self.rig.data.root_quat_w), pos_w - self.rig.data.root_pos_w)

    def proud(self, body) -> torch.Tensor:
        """(N,) post head-top height above the deck surface (rig frame; joint
        readback via the post's root pose)."""
        loc = self.rig_local(body.data.root_pos_w)
        return loc[:, 2] + L_POST / 2 - DECK_TOP

    def drive_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: fraction of the red post's required travel achieved."""
        span = (self.proud0_red - self.cfg.flush_hi).clamp(min=1e-6)
        return ((self.proud0_red - self.proud(self.post_red)) / span).clamp(0.0, 1.0)

    def red_flush(self) -> torch.Tensor:
        p = self.proud(self.post_red)
        return (p >= self.cfg.flush_lo) & (p <= self.cfg.flush_hi)

    def white_ok(self) -> torch.Tensor:
        return (self.proud(self.post_white) - self.proud0_white).abs() <= self.cfg.white_tol

    def slug_parked(self) -> torch.Tensor:
        """(N,) bool: slug standing flat in the holster tray."""
        c = self.cfg
        d = (self.slug.data.root_pos_w[:, :2]
             - self.holster.data.root_pos_w[:, :2]).norm(dim=-1)
        bottom = self.slug.data.root_pos_w[:, 2] - SLUG_H / 2
        plate_top = self.holster.data.root_pos_w[:, 2] + HOL_T
        dz = bottom - plate_top
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = _qapply(self.slug.data.root_quat_w, ez)[:, 2]
        return (d <= c.park_xy) & (dz >= c.park_z_win[0]) & (dz <= c.park_z_win[1]) \
            & (up >= c.park_up_min)

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def all_settled(self) -> torch.Tensor:
        return self.settled(self.slug) & self.settled(self.post_red) \
            & self.settled(self.post_white) & self.settled(self.rig)

    def slug_near_rig(self) -> torch.Tensor:
        d = (self.slug.data.root_pos_w[:, :2] - self.rig.data.root_pos_w[:, :2]).norm(dim=-1)
        return d < self.cfg.near_r

    # ----- step-coupled latches ---------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch score stages at sim rate (a mid-flight or transient state must not
        be missed OR un-earned later): slug ever over the rig, best red drive
        fraction, flush achieved while the white post was still undisturbed.
        During the post-reset WARMUP window, proud0 re-latches from readback
        instead (absorbing the settle transient) and the latches stay frozen."""
        warm = self.warmup > 0
        if bool(warm.any()):
            self.proud0_red = torch.where(warm, self.proud(self.post_red),
                                          self.proud0_red)
            self.proud0_white = torch.where(warm, self.proud(self.post_white),
                                            self.proud0_white)
            self.warmup = self.warmup - warm.long()
        live = ~warm
        self.ever_near |= self.slug_near_rig() & live
        self.best_drive = torch.maximum(
            self.best_drive, torch.where(live, self.drive_frac(), self.best_drive))
        self.flush_latch |= self.red_flush() & self.white_ok() & live

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red post flush with the deck, white post undisturbed, slug
        parked flat in the holster, everything settled."""
        return self.red_flush() & self.white_ok() & self.slug_parked() \
            & self.all_settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.08 (slug ever over the rig) + 0.52 x best
        drive fraction + 0.15 (flush achieved with the white post undisturbed);
        1.0 iff success(). Latched credit never evaporates under correct behavior
        (readback latches update in post_step, travel through get/set_state);
        null policy ~0."""
        base = 0.08 * self.ever_near.float() + 0.52 * self.best_drive \
            + 0.15 * self.flush_latch.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="pile_driver", robot="null", env_spacing=3.0))
