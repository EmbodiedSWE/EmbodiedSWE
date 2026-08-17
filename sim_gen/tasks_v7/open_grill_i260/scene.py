"""ServingShelfScene — deploy the grill cart's drop-leaf serving shelf on the marked
side and set the platter on it: raise the leaf past vertical onto its keeper stop,
swing the fold-away brace out underneath, lower the leaf onto the brace, then move
the platter from the cart top onto the seated shelf. The other side stays folded.

Derived from the RLBench `open_grill` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's whole plan is ONE pull on the grill's hinged lid — grab the
handle, swing it open, and the articulation's final angle IS the goal. Here the
hinged leaf's final angle is only ONE ingredient of a built structure, and the
plan is a forced-order ASSEMBLY: the leaf must first be swung PAST its goal pose
(up onto the over-vertical keeper stop) so the brace underneath is uncovered,
because while the leaf hangs it physically covers the stowed brace's swing arc —
the brace tip jams on the hanging leaf after ~8 degrees (smoke proves ramming it
with the full working torque cannot deploy it). Only then can the brace swing
out, the leaf be lowered onto it (a real contact seat: without the brace a level
leaf simply falls — its hinge has no detent at level), and the platter be placed.
Success is judged on the settled STRUCTURE, not on any joint's history.

Mechanics (compound rigid bodies via custom spawn funcs — schemas authored in the
func — plus spawn-authored RevoluteJoints; the cart is a heavy DYNAMIC body so
every joint is dynamic-dynamic and the whole linkage can be teleported at reset):
  - cart: one 25 kg compound (body box, hinge lugs, brace bosses) standing on the
    ground; per-episode xy + yaw randomization is written to the WHOLE linkage;
  - leaf_l / leaf_r: drop-leaf shelves on horizontal RevoluteJoints (axis =
    leaf-local x, limits [-80 deg, +95 deg]); body origin ON the hinge line so a
    teleport to any hinge angle is a pure quat write; CoM/inertia authored. At
    -80 deg the leaf hangs against the lower stop; at +95 deg it rests stably on
    the upper stop (past vertical, gravity presses it there);
  - brace_l / brace_r: fold-away bars on vertical-axis RevoluteJoints (limits
    [-2 deg, 92 deg]); stowed flush along the cart face, deployed 90 deg outward
    so the bar tip sits 1 mm under the leaf's level underside;
  - platter: a free board + knob compound starting on the cart top;
  - two kinematic marker tiles floated 3 mm off the side faces (never touching):
    GREEN marks the serve side, GREY the side that must stay folded; swapped per
    episode with the sampled side.
`post_step` owns the wrench slots: hinge torques are applied in the BODY frame
along the body-local joint axis (drag-invariant per the forge-pod frame quirk),
the platter probe force is pre-encoded world->body per step, and hinge rates are
finite-differenced (root_ang_vel_w is phantom under external wrenches).

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  success() = target leaf LEVEL (|hinge angle| < leaf_level_tol) AND its brace
  DEPLOYED (> brace_deploy_min) AND the platter resting on the leaf (judged in
  the LEAF'S BODY FRAME) AND the decoy side still folded (leaf hanging below
  decoy_hang_max, brace under brace_stow_max) AND everything settled and finite.
  score() = 1.0 iff success(); else 0.15*up_latch (leaf seen on the keeper stop)
  + 0.15*brace_latch (brace seen deployed, slow) + 0.25*seat_latch (leaf seen
  level WITH its brace deployed, slow — a leaf falling through level latches
  nothing) + 0.20*plat_latch (platter seen at rest on the seated shelf).
  ~0 for the null policy; latched credit never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): the serve SIDE
(left/right, tiles physically swapped), the cart's ground pose (xy + yaw — the
whole linkage moves with it), and the platter's start pose on the cart top.

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
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- USD authoring helpers ---------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None):
    """One box child: translate + scale, displayColor, optional collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    return box.GetPrim()


def _rigid_body(root, *, mass, com, inertia, lin_damp, ang_damp, kinematic=False):
    """Author RigidBodyAPI + MassAPI (mass, CoM AND diagonal inertia — PhysX's
    shape-derived inertia is unauditable) + Physx solver/damping attrs."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    ma = UsdPhysics.MassAPI.Apply(root)
    ma.CreateMassAttr(float(mass))
    ma.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _mount_quat(side: float):
    """Mount orientation for a +-y side assembly: identity on +y, 180-deg yaw on -y
    (so 'outward' is the assembly's local +y and 'raise' is + about local x on BOTH
    sides, sharing joint limits and all angle math)."""
    from pxr import Gf

    if side > 0:
        return Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    return Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))


# ----- compound spawn funcs ------------------------------------------------------------------------
def _spawn_cart(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The grill cart: heavy DYNAMIC compound (dynamic so the leaf/brace joints are
    dynamic-dynamic — body-relative anchors — and the whole linkage teleports).
    Local frame: origin at the footprint centre ON the ground. Children: body box,
    2x2 hinge lugs (visual hinge mounts; jointed pairs are collision-filtered) and
    two brace pivot bosses."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, 0.0, c.height / 2),
                inertia=c.inertia, lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(c.contact_offset)
    body = c.body_color
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, c.height / 2),
             size=(c.depth, 2 * c.hw, c.height), color=body, collide=collide)
    lug_c = c.hw + c.lug_len / 2
    for sy in (1.0, -1.0):
        tag = "l" if sy > 0 else "r"
        for sx in (1.0, -1.0):
            _add_box(stage, f"{prim_path}/lug_{tag}_{'f' if sx > 0 else 'b'}",
                     center=(sx * c.lug_x, sy * lug_c, c.hinge_z),
                     size=(0.03, c.lug_len, 0.03), color=c.trim_color, collide=collide)
        _add_box(stage, f"{prim_path}/boss_{tag}",
                 center=(-sy * c.pivot_x_abs, sy * (c.hw + 0.0075), c.pivot_z),
                 size=(0.03, 0.015, 0.03), color=c.trim_color, collide=collide)
    return root


def _spawn_leaf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A drop-leaf shelf: body origin ON the hinge line (local x along the hinge,
    local +y outward when level). Children: main plate (starts 26 mm out so the
    swing clears the lug radius), two hinge tabs bridging to the hinge line, and a
    grasp bar on top of the outer edge. Spawn-authors the RevoluteJoint to the
    sibling cart: axis local X, limits [lo_deg, hi_deg]; joint-pair collision
    filtered — the hinge IS the mount."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, c.plate_c, 0.001),
                inertia=c.inertia, lin_damp=0.05, ang_damp=0.3)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/plate", center=(0.0, c.plate_c, 0.0),
             size=(c.width, c.plate_len, c.thick), color=c.color, collide=collide)
    for sx in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/tab_{'f' if sx > 0 else 'b'}",
                 center=(sx * 0.055, c.plate_s0 / 2, 0.0),
                 size=(0.02, c.plate_s0 + 0.004, c.thick), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/grasp",
             center=(0.0, c.plate_c + c.plate_len / 2 - 0.016, c.thick / 2 + 0.008),
             size=(0.16, 0.016, 0.016), color=c.trim_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Cart"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.side) * float(c.hinge_y), float(c.hinge_z)))
    j.CreateLocalRot0Attr(_mount_quat(c.side))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLowerLimitAttr(float(c.lo_deg))
    j.CreateUpperLimitAttr(float(c.hi_deg))
    return root


def _spawn_brace(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A fold-away brace: body origin at the vertical pivot (local +x along the
    bar). Children: the bar and a downward push tab at the tip (the finger
    target). Spawn-authors the vertical-axis RevoluteJoint to the sibling cart:
    stowed at 0 (bar along the cart face), deployed at +90 (bar outward under the
    leaf); limits [-2 deg, 92 deg]."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(c.bar_len / 2 + 0.005, 0.0, -0.003),
                inertia=c.inertia, lin_damp=0.05, ang_damp=1.0)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/bar", center=(c.bar_len / 2, 0.0, 0.0),
             size=(c.bar_len, c.bar_sq, c.bar_sq), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/tab", center=(c.bar_len - 0.01, 0.0, -c.bar_sq),
             size=(0.02, 0.02, c.bar_sq), color=c.trim_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Cart"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateCollisionEnabledAttr(False)
    s = float(c.side)
    j.CreateLocalPos0Attr(Gf.Vec3f(-s * abs(c.pivot_x_abs),
                                   s * (float(c.hinge_y) - 0.030), float(c.pivot_z)))
    j.CreateLocalRot0Attr(_mount_quat(c.side))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    j.CreateLowerLimitAttr(-2.0)
    j.CreateUpperLimitAttr(92.0)
    return root


def _spawn_platter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The platter: a board with a central grasp knob (one dynamic compound)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_body(root, mass=c.mass, com=(0.0, 0.0, 0.004),
                inertia=c.inertia, lin_damp=0.2, ang_damp=0.2)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/board", center=(0.0, 0.0, 0.0),
             size=(c.board_x, c.board_y, c.board_t), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/knob", center=(0.0, 0.0, c.board_t / 2 + c.knob / 2),
             size=(c.knob, c.knob, c.knob), color=c.trim_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cart" not in _SPAWNER_CACHE:

        @configclass
        class CartSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cart)
            depth: float = 0.50
            hw: float = 0.18
            height: float = 0.40
            hinge_z: float = 0.30
            lug_x: float = 0.10
            lug_len: float = 0.055
            pivot_x_abs: float = 0.10
            pivot_z: float = 0.283
            mass: float = 25.0
            inertia: tuple = (0.60, 0.85, 0.79)
            body_color: tuple = (0.25, 0.26, 0.30)
            trim_color: tuple = (0.55, 0.56, 0.60)
            contact_offset: float = 0.002

        @configclass
        class LeafSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_leaf)
            side: float = 1.0
            width: float = 0.24
            plate_len: float = 0.18
            plate_s0: float = 0.026
            plate_c: float = 0.116
            thick: float = 0.012
            hinge_y: float = 0.225
            hinge_z: float = 0.30
            lo_deg: float = -80.0
            hi_deg: float = 95.0
            mass: float = 0.35
            inertia: tuple = (9.5e-4, 1.7e-3, 2.6e-3)
            color: tuple = (0.72, 0.48, 0.22)
            trim_color: tuple = (0.20, 0.20, 0.24)
            contact_offset: float = 0.002

        @configclass
        class BraceSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_brace)
            side: float = 1.0
            bar_len: float = 0.14
            bar_sq: float = 0.02
            hinge_y: float = 0.225
            pivot_x_abs: float = 0.10
            pivot_z: float = 0.283
            mass: float = 0.06
            inertia: tuple = (1.0e-5, 1.3e-4, 1.3e-4)
            color: tuple = (0.80, 0.78, 0.72)
            trim_color: tuple = (0.85, 0.30, 0.10)
            contact_offset: float = 0.002

        @configclass
        class PlatterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_platter)
            board_x: float = 0.14
            board_y: float = 0.10
            board_t: float = 0.014
            knob: float = 0.024
            mass: float = 0.15
            inertia: tuple = (1.3e-4, 2.5e-4, 3.7e-4)
            color: tuple = (0.90, 0.90, 0.94)
            trim_color: tuple = (0.75, 0.10, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["cart"] = CartSpawnerCfg
        _SPAWNER_CACHE["leaf"] = LeafSpawnerCfg
        _SPAWNER_CACHE["brace"] = BraceSpawnerCfg
        _SPAWNER_CACHE["platter"] = PlatterSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class ServingShelfSceneCfg(BaseCfg):
    """Config for `ServingShelfScene`. Geometry is derived once in `__post_init__`
    so the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    leaf_level_tol: float = tunable(math.radians(5.0))  # |hinge angle| below this = level
    up_thresh: float = tunable(math.radians(60.0))  # leaf past this latches "was raised"
    brace_deploy_min: float = tunable(math.radians(75.0))  # brace past this = deployed
    brace_stow_max: float = tunable(math.radians(15.0))  # decoy brace must stay under this
    decoy_hang_max: float = tunable(math.radians(-60.0))  # decoy leaf must stay under this
    plat_x_tol: float = tunable(0.08)  # platter |x| window in the leaf frame (m)
    plat_y_win: tuple = tunable((0.045, 0.165))  # platter y window in the leaf frame (m)
    plat_z_win: tuple = tunable((0.006, 0.032))  # platter z window in the leaf frame (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_rate: float = tunable(0.5)  # max |FD hinge rate| when judging (rad/s)
    slow_gate: float = tunable(0.10)  # m/s: platter slow gate for the deposit latch

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    cart_xy_range: float = tunable(0.05)  # cart xy jitter (m)
    cart_yaw_range: float = tunable(0.45)  # cart yaw jitter (rad)
    plat_x_range: tuple = tunable((-0.15, 0.15))  # platter start x on the cart top (cart frame)
    plat_y_range: tuple = tunable((-0.08, 0.08))  # platter start y on the cart top (cart frame)

    # --- tunable: plant -------------------------------------------------------------------------
    leaf_mass: float = tunable(0.35)
    brace_mass: float = tunable(0.06)
    platter_mass: float = tunable(0.15)
    leaf_tau_max: float = tunable(0.8)  # N*m cap for the leaf hinge drive (one-hand lift)
    brace_tau_max: float = tunable(0.10)  # N*m cap for the brace drive — SHARED by solve
    # and by smoke's blocked-sweep probe, so the block proof is non-vacuous.

    # --- info: structure (cart-local coordinates; cart origin = footprint centre on ground) -----
    cart_depth: float = info(0.50)
    cart_hw: float = info(0.18)  # cart half-width (y)
    cart_h: float = info(0.40)  # cart top height
    hinge_y: float = info(0.225)  # |y| of the hinge line
    hinge_z: float = info(0.30)
    leaf_lo: float = info(math.radians(-80.0))  # hanging rest (lower joint stop)
    leaf_hi: float = info(math.radians(95.0))  # keeper rest (upper joint stop, past vertical)
    leaf_width: float = info(0.24)
    leaf_len: float = info(0.18)  # plate span from plate_s0 outward
    leaf_s0: float = info(0.026)  # plate starts this far from the hinge line
    leaf_thick: float = info(0.012)
    bar_len: float = info(0.14)
    bar_sq: float = info(0.02)
    pivot_x_abs: float = info(0.10)  # brace pivot |x| (mount frame: -x)
    pivot_z: float = info(0.283)  # bar top at 0.293; level leaf underside at 0.294
    board_x: float = info(0.14)
    board_y: float = info(0.10)
    board_t: float = info(0.014)
    knob: float = info(0.024)
    tile_size: tuple = info((0.10, 0.006, 0.06))
    tile_z: float = info(0.20)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    leaf_r_com: float = field(default=None, init=False)  # CoM distance from the hinge (m)
    leaf_mgr: float = field(default=None, init=False)  # m*g*r gravity torque scale (N*m)
    plat_seat_y: float = field(default=None, init=False)  # platter target y in the leaf frame
    plat_top_z: float = field(default=None, init=False)  # platter start height on the cart top

    def __post_init__(self) -> None:
        self.leaf_r_com = self.leaf_s0 + self.leaf_len / 2  # 0.116
        self.leaf_mgr = self.leaf_mass * 9.81 * self.leaf_r_com
        self.plat_seat_y = 0.105  # centred between hinge and grasp bar, over the brace tip
        self.plat_top_z = self.cart_h + self.board_t / 2 + 0.003


# ----- scene ----------------------------------------------------------------------------------------
class ServingShelfScene(BaseScene):
    cfg: ServingShelfSceneCfg

    def __init__(self, cfg: ServingShelfSceneCfg | None = None) -> None:
        super().__init__(cfg or ServingShelfSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the cart compound (FIRST — the leaves' and braces'
        spawn-authored joints target their sibling), two leaves, two braces, the
        platter, and two kinematic marker tiles (re-posed at reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            "cart": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cart",
                spawn=sp["cart"](mass=25.0, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002)),
            ),
            "platter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platter",
                spawn=sp["platter"](mass=c.platter_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.plat_top_z)),
            ),
        }
        for tag, s in (("l", 1.0), ("r", -1.0)):
            rot = (1.0, 0.0, 0.0, 0.0) if s > 0 else (0.0, 0.0, 0.0, 1.0)
            out[f"leaf_{tag}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Leaf" + tag.upper(),
                spawn=sp["leaf"](side=s, mass=c.leaf_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, s * c.hinge_y, c.hinge_z), rot=rot),
            )
            out[f"brace_{tag}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Brace" + tag.upper(),
                spawn=sp["brace"](side=s, mass=c.brace_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-s * c.pivot_x_abs, s * (c.hinge_y - 0.030), c.pivot_z), rot=rot),
            )
        for name, color, y0 in (("tile_green", (0.10, 0.75, 0.20), 1.0),
                                ("tile_grey", (0.55, 0.55, 0.58), -1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("TileGreen" if "green" in name else "TileGrey"),
                spawn=sim_utils.CuboidCfg(
                    size=(c.tile_size[0], c.tile_size[1], c.tile_size[2]),
                    rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, y0 * (c.cart_hw + 0.003 + c.tile_size[1] / 2), c.tile_z)),
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
        n = env.num_envs
        dev = env.device
        self.cart: RigidObject = env.iscene["cart"]
        self.leaves = {1.0: env.iscene["leaf_l"], -1.0: env.iscene["leaf_r"]}
        self.braces = {1.0: env.iscene["brace_l"], -1.0: env.iscene["brace_r"]}
        self.platter: RigidObject = env.iscene["platter"]
        self.tiles = {"green": env.iscene["tile_green"], "grey": env.iscene["tile_grey"]}
        self.env_origins = env.iscene.env_origins
        # Mount quats per stored column (col 0 = left/+y, col 1 = right/-y).
        self._q_mount = torch.zeros(2, 4, device=dev)
        self._q_mount[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        self._q_mount[1] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        # Episode state.
        self.side = torch.ones(n, device=dev)  # +1 = serve on +y (left), -1 on -y
        self.cart_pose0 = torch.zeros(n, 4, device=dev)  # sampled x, y, yaw, (pad)
        self.plat_start = torch.zeros(n, 3, device=dev)  # sampled cart-frame platter start
        self.up_latch = torch.zeros(n, device=dev)
        self.brace_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.plat_latch = torch.zeros(n, device=dev)
        # FD rate state (root_ang_vel_w is phantom under external wrenches).
        self._th_prev = torch.zeros(n, 2, device=dev)
        self._ph_prev = torch.zeros(n, 2, device=dev)
        self.leaf_rate = torch.zeros(n, 2, device=dev)
        self.brace_rate = torch.zeros(n, 2, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes +
        # owns the wrench slots — never call set_external_force_and_torque directly).
        self.leaf_tau = torch.zeros(n, 2, device=dev)  # hinge torque, cols (left, right)
        self.brace_tau = torch.zeros(n, 2, device=dev)  # pivot torque, cols (left, right)
        self.platter_f = torch.zeros(n, 3, device=dev)  # WORLD-frame probe force (encoded here)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the serve SIDE (tiles physically swapped), the
        cart's ground pose and the platter's start on the cart top; write the WHOLE
        linkage (cart + both leaves at hang + both braces stowed + platter + tiles)
        with the one sampled cart transform; clear latches, drives and FD state.
        Uses torch.rand comparisons (first randint after manual_seed is degenerate)."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        cx = (torch.rand(m, device=dev) * 2 - 1) * c.cart_xy_range
        cy = (torch.rand(m, device=dev) * 2 - 1) * c.cart_xy_range
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * c.cart_yaw_range
        px = c.plat_x_range[0] + torch.rand(m, device=dev) * (c.plat_x_range[1] - c.plat_x_range[0])
        py = c.plat_y_range[0] + torch.rand(m, device=dev) * (c.plat_y_range[1] - c.plat_y_range[0])
        pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi

        self.side[env_ids] = side
        self.cart_pose0[env_ids] = torch.stack([cx, cy, cyaw, torch.zeros_like(cx)], dim=1)
        self.plat_start[env_ids] = torch.stack([px, py, torch.full_like(px, c.plat_top_z)], dim=1)
        for latch in (self.up_latch, self.brace_latch, self.seat_latch, self.plat_latch):
            latch[env_ids] = 0.0
        self.leaf_tau[env_ids] = 0.0
        self.brace_tau[env_ids] = 0.0
        self.platter_f[env_ids] = 0.0

        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0] = torch.cos(cyaw / 2)
        q_yaw[:, 3] = torch.sin(cyaw / 2)
        cart_p = torch.stack([cx, cy, torch.full_like(cx, 0.002)], dim=1) + origin

        def write(body, local_pos, q_local) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = cart_p + quat_apply(q_yaw, local_pos)
            st[:, 3:7] = quat_mul(q_yaw, q_local)
            body.write_root_state_to_sim(st, env_ids)

        write(self.cart, torch.zeros(m, 3, device=dev),
              torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4).clone())

        th0 = torch.full((m,), c.leaf_lo + math.radians(2.0), device=dev)  # settles onto the stop
        q_hang = torch.zeros(m, 4, device=dev)
        q_hang[:, 0] = torch.cos(th0 / 2)
        q_hang[:, 1] = torch.sin(th0 / 2)
        for col, s in ((0, 1.0), (1, -1.0)):
            q_mount = self._q_mount[col].expand(m, 4)
            anchor = torch.zeros(m, 3, device=dev)
            anchor[:, 1] = s * c.hinge_y
            anchor[:, 2] = c.hinge_z
            write(self.leaves[s], anchor, quat_mul(q_mount, q_hang))
            pivot = torch.zeros(m, 3, device=dev)
            pivot[:, 0] = -s * c.pivot_x_abs
            pivot[:, 1] = s * (c.hinge_y - 0.030)
            pivot[:, 2] = c.pivot_z
            write(self.braces[s], pivot, q_mount.clone())
            self._th_prev[env_ids, col] = th0
            self._ph_prev[env_ids, col] = 0.0
        self.leaf_rate[env_ids] = 0.0
        self.brace_rate[env_ids] = 0.0

        q_plat = torch.zeros(m, 4, device=dev)
        q_plat[:, 0] = torch.cos(pyaw / 2)
        q_plat[:, 3] = torch.sin(pyaw / 2)
        write(self.platter, self.plat_start[env_ids], q_plat)

        for name, s_tile in (("green", side), ("grey", -side)):
            tp = torch.zeros(m, 3, device=dev)
            tp[:, 1] = s_tile * (c.cart_hw + 0.003 + c.tile_size[1] / 2)
            tp[:, 2] = c.tile_z
            write(self.tiles[name], tp,
                  torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4).clone())

    # ----- readings ------------------------------------------------------------------------------
    def _hinge_angle(self, body, col: int, axis: int) -> torch.Tensor:
        """(N,) joint angle of `body` about its mount-frame axis (1=x hinge, 3=z
        pivot): q_h = conj(q_cart * q_mount) * q_body, angle = 2*atan2(q_h[axis], q_h[w])."""
        from isaaclab.utils.math import quat_conjugate, quat_mul

        n = self.env.num_envs
        q_mount = self._q_mount[col].expand(n, 4)
        q_ref = quat_mul(self.cart.data.root_quat_w, q_mount)
        q_h = quat_mul(quat_conjugate(q_ref), body.data.root_quat_w)
        ang = 2.0 * torch.atan2(q_h[:, axis], q_h[:, 0])
        return torch.atan2(torch.sin(ang), torch.cos(ang))

    def leaf_angle(self, s: float) -> torch.Tensor:
        """(N,) hinge angle of the side-`s` leaf: -80 deg hang .. 0 level .. +95 keeper."""
        return self._hinge_angle(self.leaves[s], 0 if s > 0 else 1, 1)

    def brace_angle(self, s: float) -> torch.Tensor:
        """(N,) pivot angle of the side-`s` brace: 0 stowed .. +90 deployed."""
        return self._hinge_angle(self.braces[s], 0 if s > 0 else 1, 3)

    def _by_side(self, fn) -> torch.Tensor:
        """(N,) target-side reading (fn evaluated per fixed side, gathered by episode side)."""
        return torch.where(self.side > 0, fn(1.0), fn(-1.0))

    def target_leaf_angle(self) -> torch.Tensor:
        return self._by_side(self.leaf_angle)

    def target_brace_angle(self) -> torch.Tensor:
        return self._by_side(self.brace_angle)

    def decoy_leaf_angle(self) -> torch.Tensor:
        return torch.where(self.side > 0, self.leaf_angle(-1.0), self.leaf_angle(1.0))

    def decoy_brace_angle(self) -> torch.Tensor:
        return torch.where(self.side > 0, self.brace_angle(-1.0), self.brace_angle(1.0))

    def platter_local(self) -> torch.Tensor:
        """(N, 3) platter centre in the TARGET LEAF'S BODY FRAME (origin = hinge
        line, +y outward along the shelf, +z off the serving face)."""
        from isaaclab.utils.math import quat_apply_inverse

        leaf_pos = torch.where((self.side > 0).unsqueeze(1),
                               self.leaves[1.0].data.root_pos_w,
                               self.leaves[-1.0].data.root_pos_w)
        leaf_quat = torch.where((self.side > 0).unsqueeze(1),
                                self.leaves[1.0].data.root_quat_w,
                                self.leaves[-1.0].data.root_quat_w)
        return quat_apply_inverse(leaf_quat, self.platter.data.root_pos_w - leaf_pos)

    def platter_on_shelf(self) -> torch.Tensor:
        """(N,) bool: platter resting on the target leaf (leaf body frame; the z
        window rejects a platter under the leaf, on the grasp bar, or airborne
        high; the y window rejects the cart top and the ground)."""
        c = self.cfg
        p = self.platter_local()
        return ((p[:, 0].abs() < c.plat_x_tol)
                & (p[:, 1] > c.plat_y_win[0]) & (p[:, 1] < c.plat_y_win[1])
                & (p[:, 2] > c.plat_z_win[0]) & (p[:, 2] < c.plat_z_win[1]))

    def target_rate(self) -> torch.Tensor:
        return torch.where(self.side > 0, self.leaf_rate[:, 0], self.leaf_rate[:, 1])

    def settled(self) -> torch.Tensor:
        """(N,) bool: platter and both leaves slow (lin), all FD hinge rates slow."""
        c = self.cfg
        ok = self.platter.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for s in (1.0, -1.0):
            ok = ok & (self.leaves[s].data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        ok = ok & (self.leaf_rate.abs().max(dim=1).values < c.settle_rate)
        ok = ok & (self.brace_rate.abs().max(dim=1).values < c.settle_rate)
        return ok & (self.cart.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        return (torch.isfinite(self.platter.data.root_pos_w).all(dim=-1)
                & torch.isfinite(self.target_leaf_angle()))

    def success(self) -> torch.Tensor:
        """(N,) bool: the SETTLED STRUCTURE — target leaf level, its brace
        deployed under it, platter resting on the shelf, decoy side still folded
        (leaf hanging, brace stowed), everything slow and finite."""
        c = self.cfg
        return ((self.target_leaf_angle().abs() < c.leaf_level_tol)
                & (self.target_brace_angle() > c.brace_deploy_min)
                & self.platter_on_shelf()
                & (self.decoy_leaf_angle() < c.decoy_hang_max)
                & (self.decoy_brace_angle() < c.brace_stow_max)
                & self.settled() & self._finite())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.15*up_latch
        + 0.15*brace_latch + 0.25*seat_latch + 0.20*plat_latch (all latched
        through slow gates in post_step — see there). ~0 for the null policy;
        latched credit never evaporates under correct behavior (raising the leaf
        past level onto the keeper is REQUIRED before the brace can deploy, so
        the up latch must survive the later lowering)."""
        partial = (0.15 * self.up_latch + 0.15 * self.brace_latch
                   + 0.25 * self.seat_latch + 0.20 * self.plat_latch)
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self) -> None:
        """Apply the drive buffers (owns the wrench slots): hinge torques in the
        BODY frame along the body-local joint axis (drag-invariant on this stack),
        the platter probe force pre-encoded world->body per step. Then update the
        FD hinge rates and latch rubric progress through slow gates."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        for col, s in ((0, 1.0), (1, -1.0)):
            t = torch.zeros(n, 1, 3, device=dev)
            t[:, 0, 0] = self.leaf_tau[:, col].clamp(-c.leaf_tau_max, c.leaf_tau_max)
            self.leaves[s].set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), t)
            t = torch.zeros(n, 1, 3, device=dev)
            t[:, 0, 2] = self.brace_tau[:, col].clamp(-c.brace_tau_max, c.brace_tau_max)
            self.braces[s].set_external_force_and_torque(torch.zeros(n, 1, 3, device=dev), t)
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, :] = quat_apply_inverse(self.platter.data.root_quat_w, self.platter_f)
        self.platter.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))

        dt = self.env.dt
        for col, s in ((0, 1.0), (1, -1.0)):
            th = self.leaf_angle(s)
            ph = self.brace_angle(s)
            d = torch.atan2(torch.sin(th - self._th_prev[:, col]),
                            torch.cos(th - self._th_prev[:, col]))
            self.leaf_rate[:, col] = d / dt
            d = torch.atan2(torch.sin(ph - self._ph_prev[:, col]),
                            torch.cos(ph - self._ph_prev[:, col]))
            self.brace_rate[:, col] = d / dt
            self._th_prev[:, col] = th
            self._ph_prev[:, col] = ph

        th_t = self.target_leaf_angle()
        ph_t = self.target_brace_angle()
        rate_t = self.target_rate()
        brace_rate_t = torch.where(self.side > 0, self.brace_rate[:, 0], self.brace_rate[:, 1])
        up = (th_t > c.up_thresh) & (rate_t.abs() < 2.0) & torch.isfinite(th_t)
        self.up_latch = torch.maximum(self.up_latch, up.float())
        br = (ph_t > c.brace_deploy_min) & (brace_rate_t.abs() < 2.0)
        self.brace_latch = torch.maximum(self.brace_latch, br.float())
        seat = ((th_t.abs() < c.leaf_level_tol) & (ph_t > c.brace_deploy_min)
                & (rate_t.abs() < c.settle_rate))
        self.seat_latch = torch.maximum(self.seat_latch, seat.float())
        plat_slow = self.platter.data.root_lin_vel_w.norm(dim=-1) < c.slow_gate
        dep = self.platter_on_shelf() & plat_slow & seat
        self.plat_latch = torch.maximum(self.plat_latch, dep.float())

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "cart_pose0", "plat_start", "up_latch", "brace_latch",
                               "seat_latch", "plat_latch", "_th_prev", "_ph_prev",
                               "leaf_rate", "brace_rate", "leaf_tau", "brace_tau",
                               "platter_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"cart": self.cart, "leaf_l": self.leaves[1.0], "leaf_r": self.leaves[-1.0],
                "brace_l": self.braces[1.0], "brace_r": self.braces[-1.0],
                "platter": self.platter, "tile_green": self.tiles["green"],
                "tile_grey": self.tiles["grey"]}

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark grill cart stands on the ground (top at {c.cart_h * 100:.0f} cm; its "
            f"position and heading vary per episode). On EACH long side hangs a folded "
            f"wooden DROP-LEAF shelf on a horizontal hinge {c.hinge_z * 100:.0f} cm up, with "
            f"a dark grasp bar along its lower (outer) edge. Behind each hanging leaf, "
            f"folded flat against the cart, is a pale swing-out BRACE bar on a vertical "
            f"pivot with an orange push tab at its tip. A small GREEN tile floats on one "
            f"side face and a GREY tile on the other; which side is green changes per "
            f"episode. A white platter with a red knob sits on the cart top.\n"
            f"Goal: deploy the shelf on the GREEN side and serve the platter on it. The "
            f"hanging leaf covers the stowed brace, so the brace cannot swing out first. "
            f"Swing the green-side leaf UP by its grasp bar, past vertical, until it rests "
            f"leaning on its keeper stop and stays there on its own. Then push the brace's "
            f"orange tab to swing the bar 90 degrees out under the shelf. Lower the leaf "
            f"back down until it lies LEVEL, resting on the brace (without the brace a "
            f"level leaf just falls back down). Finally lift the platter by its knob from "
            f"the cart top and set it flat on the deployed shelf, roughly centred. Leave "
            f"the GREY side untouched: its leaf must still hang folded and its brace must "
            f"stay stowed. Done when the green-side shelf is level on its brace with the "
            f"platter resting on it, the grey side is still folded, and everything is "
            f"still."
        )

    def instruction(self) -> str:
        return (
            "Deploy the drop-leaf shelf on the grill cart's green-marked side: swing the "
            "leaf up past vertical onto its keeper, swing the brace bar out underneath, "
            "lower the leaf level onto the brace, then set the platter from the cart top "
            "onto the shelf. Leave the grey side folded."
        )


# Guarded registration: the forge may import this module under two names.
if "serving_shelf" not in SCENES.list():
    SCENES.register("serving_shelf", ServingShelfScene)
if "simgen.serving_shelf" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="serving_shelf", robot="null"))
