"""SwingArrestScene — the workshop's two pendulums were left swinging: arrest each
one through timed inelastic contact with a heavy arrestor block, ease it to plumb,
then take the blocks away so both pendulums hang dead still, unsupported
(sim_gen task `draw_svg_i49`).

Derived from maniskill/draw_svg, but STRATEGICALLY different: the seed drags a red
marker cube along a prescribed 2-D path — the ARM'S OWN CONTINUOUS MOTION is the
product, the scene is passive, and the judged quantity is a trace of positions the
robot itself generated. Here that relationship is INVERTED: the scene arrives with
STORED KINETIC ENERGY (two pendulums are released mid-swing at reset and, left
alone, keep swinging far beyond any episode horizon), the robot's job is to REMOVE
energy rather than inject motion, and the judged product is STILLNESS — a sustained
physical rest state the scene cannot reach on its own. The seed's dragged red
marker cube returns as the pendulum bob. A solver needs a different PLAN (read the
live phase of each pendulum, stage an inelastic interceptor in the swing path while
the bob is on the far side, let impact kill the swing, ease the bob to plumb
through quasi-static contact, then extract the interceptor without re-exciting the
pendulum) and a different code structure (phase-aware timing, impact monitoring,
residual-oscillation management) — not waypoint tracking. Against the tasks_v7
corpus (surveyed before design): every existing task starts at rest and ADDS or
routes energy; none starts with kinetic energy that must be dissipated, and none
judges arrest of an autonomously moving mechanism.

The scene (fully procedural, no external assets):
  - a light-gray kinematic BENCH slab (1.1 x 1.0 x 0.10 m, top at 0.10 m);
  - two PENDULUM GANTRIES side by side (which side holds which is randomized): a
    heavy dynamic BASE FOOT with two posts and a crossbar (dynamic, not kinematic —
    a joint anchored to a teleported kinematic body0 stays world-fixed at the spawn
    pose on this stack), and an amber ROD on a revolute pivot at the crossbar
    (axis = gantry-local X, so the rod swings in the gantry's y-z plane; limits
    +/- `joint_limit_deg`) ending in a RED cube BOB. The LONG pendulum hangs
    `0.38` m from a `0.55` m pivot, the SHORT one `0.27` m from `0.42` m. The
    pivot is nearly frictionless: only a small angular damping on the rod
    (time constant ~ 1/rod_ang_damp = 50 s) bleeds energy, so a released swing
    outlives any episode by minutes — but the sub-degree residual left after a
    proper arrest is already below the stillness gates (asserted);
  - both rods are RELEASED AT RESET from a random side at a random 50-75 deg
    amplitude: the scene is MOVING from the first frame of every episode;
  - two BLACK ARRESTOR BLOCKS staged on the bench in front: a wide heavy base slab
    carrying a 50 mm square strike post (graspable; the whole block is meant to be
    slid or carried into a bob's swing path and struck).

success() (all judged live):
  A. PLUMB — each rod within `plumb_tol_deg` of vertical;
  B. STILL — SUSTAINED rest: both rods' angular speed and both blocks' linear
     speed below threshold for `still_steps` consecutive steps (a swinging rod is
     momentarily slow at each turning point; the sustained gate is what makes
     "judged only at rest" real);
  C. CLEAR — each arrestor block's xy at least `exclusion_r` from EACH gantry
     centre (rejects "leave the block propping the bob at plumb" and any parking
     of blocks on or against a gantry: even a block LYING on its side, post tip
     extended, cannot reach a within-tolerance bob from beyond the radius —
     asserted in `__post_init__`);
  D. GANTRIES HOME — each gantry within 5 cm and 10 deg of its spawn pose
     (rejects knocking a gantry over or dragging it to stop its pendulum).
Given C and D, physics itself enforces that a plumb, sustained-still rod is
hanging freely in equilibrium — the swing energy truly went away through contact.

score() (latched, anchored in the demonstrated solve trajectory): 0.20 per
pendulum ever ARRESTED (rod sustained-still and plumb, blocks anywhere) + 0.15 per
pendulum ever CLEAN (arrested while both blocks are clear of that gantry and
still), capped at 0.70; exactly 1.0 iff success() now. Null policy scores ~0: the
pendulums keep swinging (exponential decay from a 50-75 deg release takes minutes,
far longer than any episode) and the sustained-still gate never opens.

Per-episode randomization (readback-verifiable): release side and amplitude per
pendulum, long/short side swap, gantry xy + yaw jitter, arrestor staging-slot
permutation + xy jitter + free yaw.

No execution order is required (the pendulums may be arrested in either order).
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _rot_xy(ang: torch.Tensor, xy: tuple) -> torch.Tensor:
    """(m, 2) world offset of a body-local xy point under yaw `ang`."""
    ca, sa = torch.cos(ang), torch.sin(ang)
    x, y = float(xy[0]), float(xy[1])
    return torch.stack([ca * x - sa * y, sa * x + ca * y], dim=-1)


# ----- custom compound spawners (gantry / rod / arrestor block) ---------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 32, com: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass, damping, no
    sleeping while velocities are judged, depenetration cap. `com` authors an
    explicit local centre of mass — REQUIRED for the rods: with only a mass on the
    root, PhysX keeps the CoM at the body ORIGIN (here the pivot), which kills the
    pendulum's restoring torque entirely (forge-verified quirk)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_mat(stage_path: str, mat_path: str, static: float, dynamic: float,
              restitution: float = 0.0) -> None:
    """Author (once) and bind a physics material (custom spawner colliders
    otherwise get the ~0.5-friction default with no restitution control)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=restitution))
    bind_physics_material(stage_path, mat_path)


def _spawn_gantry(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one pendulum GANTRY: heavy DYNAMIC compound (a joint anchored to a
    teleported kinematic body0 stays world-fixed at spawn on this stack — the
    anchor must be a heavy dynamic fixture that follows its teleports). Local
    origin: footprint centre at bench-top level; the pendulum swings in the local
    y-z plane between the posts at local x = +/- `post_x`."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.gantry_mass, lin_damp=0.5, ang_damp=0.5, iters=16)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/foot", center=(0.0, 0.0, c.foot_h / 2),
             size=(c.foot_x, c.foot_y, c.foot_h), color=c.frame_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/post_{tag}",
                 center=(sgn * c.post_x, 0.0, c.foot_h + (c.pivot_h - c.foot_h) / 2),
                 size=(c.post_w, c.post_w, c.pivot_h - c.foot_h),
                 color=c.frame_color, collide=collide)
    _add_box(stage, f"{prim_path}/crossbar", center=(0.0, 0.0, c.pivot_h),
             size=(2 * c.post_x + c.post_w, c.post_w, c.post_w),
             color=c.frame_color, collide=collide)
    return root


def _spawn_rod(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one pendulum ROD: dynamic compound whose origin IS the pivot.
    Children: the thin bar hanging down local -z and the RED cube BOB at its end.
    Plus the REVOLUTE pivot joint to the sibling gantry (authored at spawn;
    post-play joints are dead). The pivot itself is frictionless on this stack
    (USD jointFriction only applies to articulations — forge-verified warning);
    the honest decay device is the rod's small angular damping, slow enough that
    a released swing outlives any episode. The joint pair is collision-filtered
    by USD default (the bar overlaps the crossbar at the pivot)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    # Explicit CoM (bob-dominated, from collider volumes at uniform density): with
    # only MassAPI mass, PhysX leaves the CoM at the origin = the PIVOT — no
    # restoring torque, no pendulum.
    v_bar = c.rod_w * c.rod_w * c.rod_len
    v_bob = c.bob_size ** 3
    com_z = -c.rod_len * (v_bob + v_bar / 2) / (v_bob + v_bar)
    _rigid_dynamic(root, c.rod_mass, lin_damp=0.0, ang_damp=c.rod_ang_damp,
                   com=(0.0, 0.0, com_z))
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, -c.rod_len / 2),
             size=(c.rod_w, c.rod_w, c.rod_len), color=c.rod_color, collide=collide)
    _add_box(stage, f"{prim_path}/bob", center=(0.0, 0.0, -c.rod_len),
             size=(c.bob_size,) * 3, color=c.bob_color, collide=collide)
    _bind_mat(f"{prim_path}/bob", "/World/simgenArrestBobMat", 0.60, 0.50, 0.0)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/{c.body0_name}"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.joint_limit_deg))
    j.CreateUpperLimitAttr(float(c.joint_limit_deg))
    return root


def _spawn_arrestor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one ARRESTOR BLOCK: dynamic compound — a heavy base slab (high grip
    on the bench, dense: low authored CoM) carrying a 50 mm square strike POST at
    the slab's FRONT edge (local -y): the strike face is flush with the slab's
    front, so the slab trails BEHIND the post, away from whatever the post faces —
    the block can stand close to a gantry foot, and a strike pushes it onto its
    long anti-tip lever. Local origin: footprint centre at the base's bottom."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.block_mass, lin_damp=0.2, ang_damp=1.0,
                   com=(0.0, -0.02, 0.030))
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, c.slab_h / 2),
             size=(c.slab_x, c.slab_y, c.slab_h), color=c.block_color, collide=collide)
    _add_box(stage, f"{prim_path}/post",
             center=(0.0, -(c.slab_y - c.post_wy) / 2, c.slab_h + c.post_h / 2),
             size=(c.post_wx, c.post_wy, c.post_h), color=c.block_color, collide=collide)
    _bind_mat(f"{prim_path}/slab", "/World/simgenArrestGripMat", 0.95, 0.90, 0.0)
    _bind_mat(f"{prim_path}/post", "/World/simgenArrestFaceMat", 0.60, 0.50, 0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gantry" not in _SPAWNER_CACHE:

        @configclass
        class GantrySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gantry)
            gantry_mass: float = 28.0
            foot_x: float = 0.30
            foot_y: float = 0.05
            foot_h: float = 0.020
            post_x: float = 0.10
            post_w: float = 0.040
            pivot_h: float = 0.55
            frame_color: tuple = (0.32, 0.32, 0.36)
            contact_offset: float = 0.002

        @configclass
        class RodSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rod)
            rod_mass: float = 0.42
            rod_ang_damp: float = 0.02
            rod_w: float = 0.016
            rod_len: float = 0.38
            bob_size: float = 0.055
            pivot_h: float = 0.55
            joint_limit_deg: float = 100.0
            body0_name: str = "GantryLong"
            rod_color: tuple = (0.85, 0.65, 0.20)
            bob_color: tuple = (0.85, 0.12, 0.10)
            contact_offset: float = 0.002

        @configclass
        class ArrestorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_arrestor)
            block_mass: float = 2.4
            slab_x: float = 0.10
            slab_y: float = 0.14
            slab_h: float = 0.035
            post_wx: float = 0.050
            post_wy: float = 0.050
            post_h: float = 0.20
            block_color: tuple = (0.08, 0.08, 0.09)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["gantry"] = GantrySpawnerCfg
        _SPAWNER_CACHE["rod"] = RodSpawnerCfg
        _SPAWNER_CACHE["block"] = ArrestorSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SwingArrestSceneCfg(BaseCfg):
    """Config for `SwingArrestScene`. Honesty knobs asserted in `__post_init__`:
    releases stay inside the joint limits, the strike post covers the bob band at
    plumb, a prop supporting a bob inside the plumb tolerance must sit well inside
    the exclusion radius, and the staging slots are clause-C clean at spawn."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    plumb_tol_deg: float = tunable(6.0)      # |rod angle from vertical| below this = plumb
    rod_still_avel: float = tunable(0.12)    # max rod |ang vel| when judging (rad/s)
    block_still_speed: float = tunable(0.06)  # max arrestor |lin vel| when judging (m/s)
    still_steps: int = tunable(60)           # consecutive sub-threshold steps (0.5 s) at rest
    exclusion_r: float = tunable(0.36)       # blocks must end farther than this from gantries
    home_xy_tol: float = tunable(0.05)       # gantry drift tolerance (clause D)
    home_tilt_deg: float = tunable(10.0)     # gantry uprightness tolerance (clause D)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rel_min_deg: float = tunable(50.0)       # release amplitude ~ U[rel_min, rel_max], side +/-
    rel_max_deg: float = tunable(75.0)
    side_swap: bool = tunable(True)          # long/short gantries swap bench sides
    frame_jitter: float = tunable(0.025)     # gantry xy jitter (+/- m)
    frame_yaw_deg: float = tunable(8.0)      # gantry yaw jitter (+/- deg)
    slot_shuffle: bool = tunable(True)       # permute which slot holds which arrestor
    block_jitter: float = tunable(0.02)      # arrestor xy jitter (+/- m)
    block_yaw_deg: float = tunable(180.0)    # arrestor free yaw (+/- deg)

    # --- tunable: placement (bench frame) ----------------------------------------------------
    frame_x: float = tunable(0.26)           # gantry centres at (+/- frame_x, frame_y)
    frame_y: float = tunable(0.10)
    slot_xy: tuple = tunable(((-0.12, -0.34), (0.12, -0.34)))  # arrestor staging slots

    # --- info: bench -------------------------------------------------------------------------
    bench_size: tuple = info((1.10, 1.00, 0.10))
    bench_top: float = info(0.10)
    bench_color: tuple = info((0.75, 0.75, 0.78))
    # --- info: pendulums (name, pivot height above bench, rod length) ------------------------
    pend_specs: tuple = info((("long", 0.55, 0.38), ("short", 0.42, 0.27)))
    rod_mass: float = info(0.42)
    rod_w: float = info(0.016)
    bob_size: float = info(0.055)
    joint_limit_deg: float = info(100.0)
    rod_ang_damp: float = info(0.02)         # sole decay device (pivot is frictionless)
    # --- info: gantry structure --------------------------------------------------------------
    foot_x: float = info(0.30)
    foot_y: float = info(0.05)               # narrow: an arrestor stands at the strike gap
    foot_h: float = info(0.020)
    post_x: float = info(0.10)
    post_w: float = info(0.040)
    gantry_mass: float = info(28.0)
    # --- info: arrestor blocks ---------------------------------------------------------------
    block_mass: float = info(2.4)
    slab_x: float = info(0.10)
    slab_y: float = info(0.14)
    slab_h: float = info(0.035)
    block_post_w: float = info(0.050)
    block_post_h: float = info(0.20)
    strike_gap: float = info(0.004)          # solver's bob-face-to-post-face standoff at plumb
    contact_offset: float = info(0.002)
    # rubric weights (2*0.20 + 2*0.15 = 0.70 = the non-success cap)
    w_arrest: float = info(0.20)
    w_clean: float = info(0.15)

    # Derived (filled in __post_init__).
    bob_z: tuple = field(default=None, init=False)  # bob centre height above bench, plumb

    def __post_init__(self) -> None:
        self.bob_z = tuple(hp - ln for _n, hp, ln in self.pend_specs)
        # releases never reach the joint limits (energy is conserved short of the top)
        assert self.rel_max_deg < self.joint_limit_deg - 15.0
        # ANY-orientation reach of an arrestor from its root: the full 3-D corner
        # distance (a block lying on its side reaches its post tip this far out)
        reach_any = math.sqrt((self.slab_x / 2) ** 2 + (self.slab_y / 2) ** 2
                              + (self.slab_h + self.block_post_h) ** 2)
        for (_n, hp, ln), bz in zip(self.pend_specs, self.bob_z):
            # the strike post covers the bob band at plumb ...
            lo, hi = bz - self.bob_size / 2, bz + self.bob_size / 2
            assert lo > self.slab_h + 0.02, f"bob must clear the arrestor slab ({_n})"
            assert hi < self.slab_h + self.block_post_h - 0.01, \
                f"strike post must cover the bob at plumb ({_n})"
            # ... and NO part of a clause-C-clean block — upright, tipped, or lying —
            # can touch a bob inside the plumb tolerance (rejects post-tip propping)
            prop_reach = ln * math.sin(math.radians(self.plumb_tol_deg)) \
                + self.bob_size / 2 + reach_any
            assert self.exclusion_r > prop_reach + 0.03, \
                f"exclusion radius must cover every plumb-prop pose ({_n})"
            # the bob swings between the posts with clearance
            assert self.post_x - self.post_w / 2 > self.bob_size / 2 + 0.02
        # an arrestor standing at the strike gap (front face at bob face + gap) does
        # not touch the narrow gantry foot
        assert self.foot_y / 2 + 0.002 < self.bob_size / 2 + self.strike_gap, \
            "gantry foot must clear an arrestor placed at the strike gap"
        # gantries far enough apart that a bob's swing plane clears the other gantry
        assert 2 * self.frame_x > self.foot_x + self.bob_size + 0.05
        # staging slots are clause-C clean at spawn (both gantries, worst-case jitter)
        # and on the bench with any block yaw
        for sx, sy in self.slot_xy:
            for gx in (self.frame_x, -self.frame_x):
                d = math.hypot(sx - gx, sy - self.frame_y) \
                    - self.block_jitter - self.frame_jitter
                assert d > self.exclusion_r + 0.03, "staging slots must clear the exclusion"
            hd = math.hypot(self.slab_x, self.slab_y) / 2 + self.block_jitter
            assert abs(sx) + hd < self.bench_size[0] / 2
            assert abs(sy) + hd < self.bench_size[1] / 2
        # the two slots are far enough apart to grasp either block
        (x0, y0), (x1, y1) = self.slot_xy
        assert math.hypot(x1 - x0, y1 - y0) > 0.14
        # the strike post fits the Franka jaw
        assert self.block_post_w < 0.078
        # the residual swing left after a proper arrest is judged still: a bob that
        # came to rest leaning on a post face at the strike gap swings freely, once
        # the block is removed, with peak angular speed below the still gate (the
        # pivot is frictionless on this stack; damping only shrinks it further)
        for _n, hp, ln in self.pend_specs:
            lean = math.asin(self.strike_gap / ln)  # rest lean against the post face
            w_peak = 1.2 * lean * math.sqrt(9.81 / ln)
            assert w_peak < self.rod_still_avel, \
                f"post-arrest residual swing would defeat the still gate ({_n})"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("swing_arrest")
class SwingArrestScene(BaseScene):
    cfg: SwingArrestSceneCfg

    def __init__(self, cfg: SwingArrestSceneCfg | None = None) -> None:
        super().__init__(cfg or SwingArrestSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.bench_top
        sp = _spawner_classes()
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.60, dynamic_friction=0.50,
                                             restitution=0.0)
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
            "bench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=c.bench_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.bench_size[2] / 2)),
            ),
        }
        # spawn order matters: each rod joint's body0 (its gantry) must already exist
        for i, (name, hp, ln) in enumerate(c.pend_specs):
            x = c.frame_x if i == 0 else -c.frame_x
            gname = f"Gantry{name.capitalize()}"
            out[f"gantry_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + gname,
                spawn=sp["gantry"](pivot_h=hp, foot_x=c.foot_x, foot_y=c.foot_y,
                                   foot_h=c.foot_h, post_x=c.post_x, post_w=c.post_w,
                                   gantry_mass=c.gantry_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, c.frame_y, z0)),
            )
            out[f"rod_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod" + name.capitalize(),
                spawn=sp["rod"](pivot_h=hp, rod_len=ln, rod_mass=c.rod_mass,
                                rod_w=c.rod_w, bob_size=c.bob_size,
                                joint_limit_deg=c.joint_limit_deg,
                                rod_ang_damp=c.rod_ang_damp, body0_name=gname,
                                contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, c.frame_y, z0 + hp)),
            )
        for j, (sx, sy) in enumerate(c.slot_xy):
            out[f"block_{j}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Arrestor_" + str(j),
                spawn=sp["block"](block_mass=c.block_mass, slab_x=c.slab_x, slab_y=c.slab_y,
                                  slab_h=c.slab_h, post_wx=c.block_post_w,
                                  post_wy=c.block_post_w, post_h=c.block_post_h,
                                  contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, z0)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.gantries: list[RigidObject] = [
            env.iscene[f"gantry_{nm}"] for nm, _hp, _ln in c.pend_specs]
        self.rods: list[RigidObject] = [
            env.iscene[f"rod_{nm}"] for nm, _hp, _ln in c.pend_specs]
        self.blocks: list[RigidObject] = [
            env.iscene[f"block_{j}"] for j in range(len(c.slot_xy))]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.frame_home = torch.zeros(n, 2, 2, device=dev)   # spawn xy per gantry
        self.theta0 = torch.zeros(n, 2, device=dev)          # signed release angle (rad)
        self.frame_yaw = torch.zeros(n, 2, device=dev)       # spawn yaw per gantry
        self.arrest_latch = torch.zeros(n, 2, device=dev)
        self.clean_latch = torch.zeros(n, 2, device=dev)
        self.rod_still_count = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.still_count = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the two gantries (long/short possibly side-swapped,
        xy + yaw jittered), write each rod RELEASED at a random signed 50-75 deg
        angle with zero velocity — the pendulums start FALLING on the first step —
        and stage the arrestor blocks on shuffled slots; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.bench_top

        swap = (torch.rand(m, device=dev) < 0.5) if c.side_swap \
            else torch.zeros(m, dtype=torch.bool, device=dev)
        side = torch.where(swap, -torch.ones(m, device=dev), torch.ones(m, device=dev))

        def write(body, xy: torch.Tensor, z, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for i, (_nm, hp, _ln) in enumerate(c.pend_specs):
            gx = side * c.frame_x * (1.0 if i == 0 else -1.0)
            xy = torch.stack([gx, torch.full((m,), c.frame_y, device=dev)], dim=-1)
            xy = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.frame_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.frame_yaw_deg)
            q = _qz(yaw)
            th = (torch.rand(m, device=dev)
                  * math.radians(c.rel_max_deg - c.rel_min_deg)
                  + math.radians(c.rel_min_deg))
            th = th * torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
            write(self.gantries[i], xy, z0, q)
            write(self.rods[i], xy, z0 + hp, _qmul(q, _qx(th)))
            self.frame_home[env_ids, i] = xy
            self.frame_yaw[env_ids, i] = yaw
            self.theta0[env_ids, i] = th

        perm = torch.rand(m, 2, device=dev).argsort(dim=1) if c.slot_shuffle \
            else torch.arange(2, device=dev).unsqueeze(0).expand(m, 2)
        slots = torch.tensor(list(c.slot_xy), device=dev)
        for j, body in enumerate(self.blocks):
            xy = slots[perm[:, j]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.block_jitter
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg)
            write(body, xy, z0 + 0.002, _qz(byaw))

        self.arrest_latch[env_ids] = 0.0
        self.clean_latch[env_ids] = 0.0
        self.rod_still_count[env_ids] = 0
        self.still_count[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gantries": [b.data.root_state_w[env_ids].clone() for b in self.gantries],
            "rods": [b.data.root_state_w[env_ids].clone() for b in self.rods],
            "blocks": [b.data.root_state_w[env_ids].clone() for b in self.blocks],
            "frame_home": self.frame_home[env_ids].clone(),
            "frame_yaw": self.frame_yaw[env_ids].clone(),
            "theta0": self.theta0[env_ids].clone(),
            "arrest_latch": self.arrest_latch[env_ids].clone(),
            "clean_latch": self.clean_latch[env_ids].clone(),
            "rod_still_count": self.rod_still_count[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.gantries, state["gantries"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.rods, state["rods"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.blocks, state["blocks"]):
            b.write_root_state_to_sim(st, env_ids)
        self.frame_home[env_ids] = state["frame_home"]
        self.frame_yaw[env_ids] = state["frame_yaw"]
        self.theta0[env_ids] = state["theta0"]
        self.arrest_latch[env_ids] = state["arrest_latch"]
        self.clean_latch[env_ids] = state["clean_latch"]
        self.rod_still_count[env_ids] = state["rod_still_count"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Two PENDULUM GANTRIES stand side by side on a light-gray bench: gray frames "
            f"(two posts and a crossbar) each carrying an amber ROD on a free pivot, "
            f"ending in a RED cube BOB ({c.bob_size * 1000:.0f} mm). One pendulum is LONG "
            f"({c.pend_specs[0][2] * 100:.0f} cm rod), the other SHORT "
            f"({c.pend_specs[1][2] * 100:.0f} cm); which side each stands on changes "
            f"between episodes. BOTH PENDULUMS ARE SWINGING — they were released "
            f"mid-swing and their pivots are nearly frictionless, so left alone they keep "
            f"swinging far beyond the episode. On the bench in front lie two identical "
            f"BLACK ARRESTOR BLOCKS: a heavy wide base slab carrying a "
            f"{c.block_post_w * 1000:.0f} mm square strike post — sized so a bob strikes "
            f"the post face flat, and narrow enough to grasp.\n"
            f"Goal: bring BOTH pendulums to a COMPLETE REST hanging PLUMB (within "
            f"{c.plumb_tol_deg:.0f} degrees of vertical). The intended tool is the "
            f"arrestor blocks: stood in a bob's swing path, a block soaks up the swing "
            f"energy on impact and the bob settles against its post near plumb. When both "
            f"pendulums hang still, take the blocks AWAY: every block must end at least "
            f"{c.exclusion_r * 100:.0f} cm from both gantry centres — a pendulum still "
            f"leaning on a block, or a block left against a gantry, voids the result. Do "
            f"not shift or topple the gantries themselves. The scene is judged only at "
            f"sustained rest: a rod merely pausing at the top of a swing does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Both pendulums are swinging. Use the black arrestor blocks to stop them: "
            "stand a block in each bob's path, let the bob strike it and settle, then "
            "remove both blocks well away from the gantries so both pendulums hang "
            "straight down, dead still and unsupported. Do not move the gantries."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def angle(self) -> torch.Tensor:
        """(N, 2) signed rod angle from vertical (radians); positive = bob toward
        the gantry's local +y."""
        from isaaclab.utils.math import quat_apply

        out = []
        for i, rod in enumerate(self.rods):
            q = rod.data.root_quat_w
            down = torch.tensor([0.0, 0.0, -1.0], device=q.device).expand(q.shape[0], 3)
            v = quat_apply(q, down)                      # rod's down axis in world
            gq = self.gantries[i].data.root_quat_w
            ey = torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(q.shape[0], 3)
            ydir = quat_apply(gq, ey)                    # swing direction in world
            out.append(torch.atan2((v * ydir).sum(-1), -v[:, 2]))
        return torch.stack(out, dim=1)

    def rod_avel(self) -> torch.Tensor:
        """(N, 2) rod angular speed (rad/s)."""
        return torch.stack(
            [r.data.root_ang_vel_w.norm(dim=-1) for r in self.rods], dim=1)

    def bob_pos(self) -> torch.Tensor:
        """(N, 2, 3) bob centre world positions."""
        from isaaclab.utils.math import quat_apply

        out = []
        for (_nm, _hp, ln), rod in zip(self.cfg.pend_specs, self.rods):
            q = rod.data.root_quat_w
            off = torch.tensor([0.0, 0.0, -ln], device=q.device).expand(q.shape[0], 3)
            out.append(rod.data.root_pos_w + quat_apply(q, off))
        return torch.stack(out, dim=1)

    # ----- predicates -------------------------------------------------------------------------
    def plumb(self) -> torch.Tensor:
        """(N, 2) bool: rod within `plumb_tol_deg` of vertical."""
        return self.angle().abs() <= math.radians(self.cfg.plumb_tol_deg)

    def _block_xy(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w[:, :2] for b in self.blocks], dim=1)

    def _block_vel(self) -> torch.Tensor:
        return torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.blocks], dim=1)

    def blocks_clear(self) -> torch.Tensor:
        """(N, 2) bool per GANTRY: both arrestor blocks farther than `exclusion_r`
        from that gantry's centre (clause C, per gantry)."""
        bxy = self._block_xy()                                    # (N, B, 2)
        out = []
        for g in self.gantries:
            gxy = g.data.root_pos_w[:, None, :2]
            out.append(((bxy - gxy).norm(dim=-1) > self.cfg.exclusion_r).all(dim=1))
        return torch.stack(out, dim=1)

    def gantries_home(self) -> torch.Tensor:
        """(N,) bool: each gantry within `home_xy_tol` of its spawn xy and upright
        within `home_tilt_deg` (clause D)."""
        from isaaclab.utils.math import quat_apply

        ok = None
        for i, g in enumerate(self.gantries):
            xy = g.data.root_pos_w[:, :2] - self.env_origins[:, :2]
            near = (xy - self.frame_home[:, i]).norm(dim=-1) < self.cfg.home_xy_tol
            q = g.data.root_quat_w
            ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
            up = quat_apply(q, ez)[:, 2] > math.cos(math.radians(self.cfg.home_tilt_deg))
            good = near & up
            ok = good if ok is None else (ok & good)
        return ok

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: both rods, both blocks and both gantries below their velocity
        thresholds at THIS instant (also true for a moment at a swing's turning
        point — hence the sustained counter)."""
        c = self.cfg
        ok = (self.rod_avel() < c.rod_still_avel).all(dim=1)
        ok = ok & (self._block_vel() < c.block_still_speed).all(dim=1)
        for g in self.gantries:
            ok = ok & (g.data.root_lin_vel_w.norm(dim=-1) < c.block_still_speed)
        return ok

    def still(self) -> torch.Tensor:
        """(N,) bool: SUSTAINED rest — sub-threshold for `still_steps` consecutive
        sim steps (counter kept by `post_step`)."""
        return self.still_count >= self.cfg.still_steps

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Per-step: advance the sustained-stillness counters, then latch
        per-pendulum ARRESTED (rod sustained-still and plumb) and CLEAN (arrested
        while both blocks are clear of that gantry and still). Latched credit does
        not evaporate if a later action re-excites a pendulum."""
        c = self.cfg
        rod_ok = self.rod_avel() < c.rod_still_avel
        self.rod_still_count = torch.where(
            rod_ok, self.rod_still_count + 1, torch.zeros_like(self.rod_still_count))
        now = self._still_now()
        self.still_count = torch.where(
            now, self.still_count + 1, torch.zeros_like(self.still_count))

        arrested = (self.rod_still_count >= c.still_steps) & self.plumb()
        self.arrest_latch = torch.maximum(self.arrest_latch, arrested.float())
        blocks_still = (self._block_vel() < c.block_still_speed).all(dim=1, keepdim=True)
        clean = arrested & self.blocks_clear() & blocks_still
        self.clean_latch = torch.maximum(self.clean_latch, clean.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both rods plumb AND everything sustained-still AND both
        blocks beyond the exclusion radius of both gantries AND the gantries at
        their spawn poses. Physics then guarantees both pendulums hang freely —
        their swing energy was truly dissipated."""
        return (self.plumb().all(dim=1) & self.still()
                & self.blocks_clear().all(dim=1) & self.gantries_home())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20 per pendulum ever arrested + 0.15 per pendulum
        ever cleanly arrested, capped at 0.70; exactly 1.0 iff success() now. Null
        policy: the pendulums never stop swinging -> 0."""
        c = self.cfg
        base = (c.w_arrest * self.arrest_latch.sum(dim=1)
                + c.w_clean * self.clean_latch.sum(dim=1)).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="swing_arrest", robot="null"))
