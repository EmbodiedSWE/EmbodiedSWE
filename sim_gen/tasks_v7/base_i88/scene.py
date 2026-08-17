"""BallastLiftScene — tip a see-saw with ballast cubes to raise an ungraspable ball,
then push it onto a shelf dock (sim_gen task `base_i88`).

Derived from pick_place/base, but STRATEGICALLY different: the seed is direct
grasp-and-carry — close the jaw on a 4 cm cube and translate it through a sequence of
free-space waypoints to a goal pose (one grasp, one guided transport, judged by
waypoint tracking). Here the transported cargo is a 90 mm ball that CANNOT be grasped
(wider than the Franka's 80 mm jaw span) and the goal surface is a shelf top that
cannot be reached from below: the ball starts in a fenced tray on the cargo end of a
SEE-SAW beam (revolute hinge on a pylon, +-15 deg) resting cargo-end-DOWN. The solver
must (1) drop three loose 40 mm ballast cubes into a chimney-shaped hopper on the
OPPOSITE end of the beam until the accumulated counterweight out-moments the ball and
the beam tips, raising the tray to shelf height, and only then (2) push the ball
sideways — the hinge axis direction stays level at every beam angle — over the tray's
low lip, across a 10 mm gap, into the walled dock on the shelf top. A solver needs a
different PLAN from the seed (indirect actuation by accumulating mass on a lever, then
a push-only lateral transfer — the target object is never held) and a different code
structure (beam-angle + hopper-containment latches and a dock predicate — not
gripper-distance waypoint tracking).

The mechanism is real physics, not a scripted flag, and the margins are asserted in
`__post_init__` from the cfg's own numbers:
  - the ball's gravity moment about the hinge is in [ball_mom_min, ball_mom_max]
    (its rest band in the tray, over the full +-15 deg sweep);
  - ONE cube on the hopper floor gives at most `cube1_mom_max` < ball_mom_min: a
    single cube can NEVER tip the beam (smoke proves it with a settled probe);
  - THREE stacked cubes give at least `cube3_mom_min` > ball_mom_max at the worst
    angle: a full hopper ALWAYS tips it (solve demonstrates it);
  - the shelf top sits below the raised tray lip (the transfer is downhill) and
    over 0.13 m above a ground-resting ball's top (no path up from below).
The beam's own mass is authored with MassAPI CoM at the hinge, so the plank
contributes no gravity moment and the ballast-vs-ball ledger above is the whole story.

success(): the ball rests INSIDE the shelf dock — env-frame containment window that
accepts every physically-in-dock resting pose (dock cavity minus ball radius) and
rejects the raised-tray pose (x window), the ground beside the pedestal (z window) and
gap-bridging perches (x window) — with the ball settled. The identity is enforced by
name: only the ball counts, a ballast cube in the dock counts for nothing.

score() is graded and latched (credit never evaporates): 0.10 per cube ever contained
in the hopper + 0.25 beam ever raised + 0.10 ball ever across onto the shelf = 0.65
cap; 1.0 iff success(). The null policy scores ~0 (beam spawns cargo-end-down, cubes
scattered on the ground, ball in the tray).

Per-episode randomization (readback-verified in smoke): ballast cube ground slots are
PERMUTED per env with xy jitter + free yaw, and the ball's resting spot in the tray
jitters along the hinge-axis direction. Assets are fully procedural compound-box /
sphere spawners (one rigid body each; decorations authored idempotently). Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
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
                kinematic: bool = False, com_at_origin: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on). `com_at_origin` pins the center of
    mass to the body origin explicitly (the beam's moment ledger depends on it)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com_at_origin:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_pylon(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC hinge pylon: a post whose top stays below the plank's swing
    envelope (the hinge anchor floats `hinge_h` above the post top by design). One
    rigid body; origin = post footprint center at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 5.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/post", (0.06, 0.06, cfg.post_h),
         (0.0, 0.0, cfg.post_h / 2), (0.25, 0.25, 0.28), cfg.contact_offset,
         material=mat)
    _box(stage, f"{prim_path}/base", (0.16, 0.16, 0.02), (0.0, 0.0, 0.01),
         (0.25, 0.25, 0.28), cfg.contact_offset, material=mat)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The see-saw beam, one dynamic rigid body, origin AT the hinge point (and the
    authored CoM pinned there, so the plank adds no gravity moment): plank, cargo
    tray at +y (tall fences on three sides, a LOW push-over lip on the +x side),
    ballast hopper chimney at -y (four tall walls, open top; the plank is its
    floor)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.beam_mass,
                             com_at_origin=True)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(1.0)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c = cfg
    co = c.contact_offset
    wood, dark, lipc = (0.72, 0.52, 0.28), (0.30, 0.30, 0.34), (0.85, 0.70, 0.20)
    boxes = [
        ("plank", (0.13, 0.56, 0.016), (0.0, 0.0, 0.008), wood),
        # cargo tray (+y): floor is the plank top (local z 0.016)
        ("tray_in", (0.13, 0.014, 0.040), (0.0, 0.148, 0.036), wood),   # face y 0.155
        ("tray_out", (0.13, 0.014, 0.040), (0.0, 0.262, 0.036), wood),  # face y 0.255
        ("tray_wall", (0.014, 0.100, 0.040), (-0.058, 0.205, 0.036), wood),
        ("tray_lip", (0.014, 0.100, 0.012), (0.058, 0.205, 0.022), lipc),  # top 0.028
        # ballast hopper chimney (-y): cavity 0.062 x 0.062 (cube diag 0.057 fits),
        # y in [-0.261, -0.199], walls z in [0.016, 0.156]
        ("hop_in", (0.076, 0.014, 0.140), (0.0, -0.192, 0.086), dark),
        ("hop_out", (0.076, 0.014, 0.140), (0.0, -0.268, 0.086), dark),
        ("hop_px", (0.014, 0.062, 0.140), (+0.038, -0.230, 0.086), dark),
        ("hop_nx", (0.014, 0.062, 0.140), (-0.038, -0.230, 0.086), dark),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    return root


def _spawn_shelf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC shelf: a tall pedestal whose top carries the dock — far wall and
    two side walls, open toward the see-saw (-x). Origin = pedestal footprint center
    at ground level."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 8.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c = cfg
    co = c.contact_offset
    green, rim = (0.16, 0.55, 0.30), (0.10, 0.38, 0.20)
    t = c.shelf_top  # pedestal top height
    boxes = [
        ("pedestal", (0.15, 0.15, t), (0.0, 0.0, t / 2), green),
        ("dock_far", (0.014, 0.150, 0.040), (+0.068, 0.0, t + 0.020), rim),
        ("dock_py", (0.150, 0.014, 0.040), (0.0, +0.068, t + 0.020), rim),
        ("dock_ny", (0.150, 0.014, 0.040), (0.0, -0.068, t + 0.020), rim),
        # low retention sill on the open (-x) side: the push crosses it like the tray
        # lip (~2.3 N << the servo's 4 N stall force); once inside, the ball cannot
        # roll back out (an in-dock rest against the sill sits at x >= 0.569, inside
        # the dock_x window)
        ("dock_sill", (0.014, 0.150, 0.012), (-0.068, 0.0, t + 0.006), rim),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cargo ball: one 90 mm sphere — wider than the Franka's 80 mm jaw span, so
    it can be pushed and rolled but never grasped."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)
    mat = _material(stage, f"{prim_path}/phys_mat")
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/body")
    sph.CreateRadiusAttr(float(cfg.radius))
    sph.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.12, 0.12)])
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(sph.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return root


def _spawn_cube(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One 40 mm ballast cube (graspable by any parallel jaw)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.10)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/body", (cfg.edge, cfg.edge, cfg.edge), (0.0, 0.0, 0.0),
         (0.15, 0.35, 0.85), cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pylon" not in _SPAWNER_CACHE:

        @configclass
        class PylonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pylon)
            post_h: float = 0.125
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            beam_mass: float = 1.0
            contact_offset: float = 0.002

        @configclass
        class ShelfSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shelf)
            shelf_top: float = 0.225
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            mass: float = 0.25
            radius: float = 0.045
            contact_offset: float = 0.002

        @configclass
        class CubeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cube)
            mass: float = 0.15
            edge: float = 0.04
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pylon=PylonSpawnerCfg, beam=BeamSpawnerCfg,
                              shelf=ShelfSpawnerCfg, ball=BallSpawnerCfg,
                              cube=CubeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BallastLiftSceneCfg(BaseCfg):
    """Config for `BallastLiftScene`. The mechanism honesty (one cube can never tip
    the beam; three always do; the shelf is reachable only from the raised tray) is
    asserted in `__post_init__` from these numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    raise_deg: float = tunable(12.0)  # beam counts as raised at this tilt (limit 15)
    dock_x: tuple = tunable((0.545, 0.645))  # ball-center window on the shelf, env x
    dock_y: tuple = tunable((0.115, 0.225))  # ... env y
    dock_z: tuple = tunable((0.245, 0.300))  # ... env z (shelf_top + ball_r = 0.270)
    cross_x: float = tunable(0.545)  # ball ever past this x at height -> crossed latch
    cross_z: float = tunable(0.240)
    settle_lin: float = tunable(0.05)  # max ball |lin vel| at judging (m/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cube_slots: tuple = tunable(((0.16, -0.34), (0.30, -0.26), (0.14, -0.16)))
    cube_jitter: float = tunable(0.025)  # +-xy jitter per cube at its permuted slot
    cube_yaw_deg: float = tunable(180.0)  # +- free yaw per cube
    ball_jitter: float = tunable(0.018)  # +-x (hinge-axis) jitter of the ball in tray

    # --- info: structure (the geometry the spawners author) ----------------------------------
    pylon_xy: tuple = info((0.45, 0.0))  # hinge pylon footprint center (env frame)
    hinge_h: float = info(0.15)  # hinge (= beam origin) height above ground
    post_h: float = info(0.125)  # pylon post top (clears the plank swing by 14 mm)
    tilt_max_deg: float = info(15.0)  # revolute limits +- this
    plank_half: float = info(0.28)  # plank half-length (y)
    plank_top: float = info(0.016)  # plank top, beam local z
    tray_y: tuple = info((0.155, 0.255))  # tray cavity y span (beam local)
    tray_lip_x: float = info(0.051)  # lip inner face, beam local x (outer 0.065)
    tray_lip_top: float = info(0.028)  # lip top, beam local z (12 mm above the floor)
    hop_y: tuple = info((-0.261, -0.199))  # hopper cavity y span (beam local)
    hop_x_half: float = info(0.031)  # hopper cavity half-width, x
    hop_top: float = info(0.156)  # hopper wall top, beam local z
    beam_mass: float = info(1.0)  # authored with CoM at the hinge -> zero moment
    ball_r: float = info(0.045)
    ball_mass: float = info(0.25)
    cube_edge: float = info(0.04)
    cube_mass: float = info(0.15)
    n_cubes: int = info(3)
    shelf_xy: tuple = info((0.60, 0.17))  # pedestal footprint center
    shelf_top: float = info(0.225)  # pedestal top = dock floor height
    shelf_near_x: float = info(0.525)  # pedestal near face (10 mm from the beam edge)
    dock_far_x: float = info(0.661)  # dock far-wall inner face
    dock_y_faces: tuple = info((0.109, 0.231))  # dock side-wall inner faces
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        s, cmax = math.sin(math.radians(c.tilt_max_deg)), math.cos(math.radians(c.tilt_max_deg))
        g = 9.81
        # -- the ball's gravity-moment band about the hinge (rest band in the tray:
        # against the outer fence when down, against the inner fence when up) --
        ball_y_out = c.tray_y[1] - c.ball_r * 0.93  # ~ touching the outer fence
        ball_y_in = c.tray_y[0] + c.ball_r * 0.93
        ball_zc = c.plank_top + c.ball_r
        ball_mom_max = c.ball_mass * g * (ball_y_out * cmax + ball_zc * s)
        ball_mom_min = c.ball_mass * g * (ball_y_in * cmax - ball_zc * s)
        # -- one cube on the hopper floor: max possible moment < ball min moment --
        cube_y_far = abs(c.hop_y[0]) - c.cube_edge / 2
        cube_z0 = c.plank_top + c.cube_edge / 2
        cube1_max = c.cube_mass * g * (cube_y_far * 1.0 + cube_z0 * s)
        assert cube1_max < ball_mom_min * 0.95, \
            f"one cube must NEVER tip the beam ({cube1_max:.3f} vs {ball_mom_min:.3f})"
        # -- three stacked cubes at the WORST angle (bucket end raised, arms
        # shortened by the stack height): min total moment > ball max moment --
        cube_y_near = abs(c.hop_y[1]) - c.cube_edge / 2
        cube3_min = sum(
            c.cube_mass * g * (cube_y_near * cmax - (cube_z0 + k * c.cube_edge) * s)
            for k in range(3))
        assert cube3_min > ball_mom_max * 1.10, \
            f"three cubes must ALWAYS tip the beam ({cube3_min:.3f} vs {ball_mom_max:.3f})"
        # -- the hopper swallows a cube in any yaw, and no two fit side by side --
        diag = c.cube_edge * math.sqrt(2.0)
        assert diag < 2 * c.hop_x_half - 0.004, "yawed cube must not jam the hopper"
        assert 2 * c.cube_edge > c.hop_y[1] - c.hop_y[0], "cubes must stack, not spread"
        # -- the transfer is downhill and only exists at the raised limit --
        lip_y_mid = (c.tray_y[0] + c.tray_y[1]) / 2
        lip_top_raised = c.hinge_h + lip_y_mid * s * (0.97) + c.tray_lip_top * cmax
        assert c.shelf_top <= lip_top_raised + 0.002, \
            f"shelf top {c.shelf_top:.3f} must sit at/below the raised lip {lip_top_raised:.3f}"
        assert c.shelf_top > 2 * c.ball_r + 0.10, \
            "shelf top must be far above a ground-resting ball (no path from below)"
        # -- the swinging beam clears the pedestal --
        assert c.pylon_xy[0] + 0.065 + 2 * c.contact_offset < c.shelf_near_x, \
            "beam +x edge must clear the pedestal near face at every angle"
        # -- the dock window is honest by construction --
        assert c.dock_x[0] > c.shelf_near_x + 0.012, \
            "gap-bridging / edge-hanging poses must fall below the x window"
        assert c.dock_x[1] >= c.dock_far_x - c.ball_r, \
            "the against-the-far-wall rest pose must be inside the x window"
        assert c.dock_y[0] <= c.dock_y_faces[0] + c.ball_r <= c.dock_y[1], \
            "the against-a-side-wall rest pose must be inside the y window"
        assert c.dock_z[0] <= c.shelf_top + c.ball_r <= c.dock_z[1], \
            "the on-dock-floor rest pose must be inside the z window"
        assert c.dock_z[0] > 2 * c.ball_r + 0.03, \
            "a ground-resting ball must be below the z window"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ballast_lift")
class BallastLiftScene(BaseScene):
    cfg: BallastLiftSceneCfg

    def __init__(self, cfg: BallastLiftSceneCfg | None = None) -> None:
        super().__init__(cfg or BallastLiftSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        px, py = c.pylon_xy

        out: dict[str, Any] = {
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
            "pylon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pylon",
                spawn=sp["pylon"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    post_h=c.post_h, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0)),
            ),
            "shelf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                spawn=sp["shelf"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    shelf_top=c.shelf_top, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_xy[0], c.shelf_xy[1], 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=sp["beam"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    beam_mass=c.beam_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, c.hinge_h)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sp["ball"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.ball_mass, radius=c.ball_r,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py + 0.205, c.hinge_h + 0.10)),
            ),
        }
        for i in range(c.n_cubes):
            sx, sy = c.cube_slots[i]
            out[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube" + str(i),
                spawn=sp["cube"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.cube_mass, edge=c.cube_edge,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(sx, sy, c.cube_edge / 2 + 0.002)),
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
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.pylon: RigidObject = env.iscene["pylon"]
        self.beam: RigidObject = env.iscene["beam"]
        self.ball: RigidObject = env.iscene["ball"]
        self.shelf: RigidObject = env.iscene["shelf"]
        self.cubes: list[RigidObject] = [env.iscene[f"cube_{i}"] for i in range(c.n_cubes)]
        self.env_origins = env.iscene.env_origins
        # progress latches (post_step): per-cube ever-in-hopper, beam ever raised,
        # ball ever across onto the shelf
        self.cube_latch = torch.zeros(n, c.n_cubes, device=dev)
        self.raise_latch = torch.zeros(n, device=dev)
        self.cross_latch = torch.zeros(n, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per env: the beam's revolute hinge on the kinematic pylon — pair collision
        disabled (the joint limits are the mechanical stops), SYMMETRIC limits
        +-tilt_max about the hinge anchor. Axis X (the hinge axis is the direction
        that stays level; positive rotation raises the +y cargo end)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/Beam_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Pylon"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.hinge_h))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.tilt_max_deg)
            j.CreateUpperLimitAttr(c.tilt_max_deg)
            lim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "angular")
            if hasattr(lim, "CreateContactDistanceAttr"):  # removed in Isaac 5.1
                lim.CreateContactDistanceAttr(0.5)

    def _beam_pose_at(self, phi_deg: float, m: int) -> torch.Tensor:
        """Root state (m, 13) for the beam pinned at hinge with tilt `phi_deg`."""
        c = self.cfg
        dev = self.env.device
        half = math.radians(phi_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pylon_xy[0]
        st[:, 1] = c.pylon_xy[1]
        st[:, 2] = c.hinge_h
        st[:, 3] = math.cos(half)
        st[:, 4] = math.sin(half)  # rotation about +x
        return st

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: beam pinned near the cargo-down limit, ball dropped into
        the tray with hinge-axis jitter, ballast cubes at PERMUTED ground slots with
        xy jitter + free yaw; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        phi0 = -(c.tilt_max_deg - 2.0)  # near the lower stop; settles onto it

        st = self._beam_pose_at(phi0, m)
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        # ball: just above the tray floor at its jittered hinge-axis spot, in the
        # beam's tilted frame (drops the last few mm under gravity)
        s0, c0 = math.sin(math.radians(phi0)), math.cos(math.radians(phi0))
        jx = (torch.rand(m, device=dev) * 2 - 1) * c.ball_jitter
        y_l = 0.205
        z_l = c.plank_top + c.ball_r + 0.006
        bst = torch.zeros(m, 13, device=dev)
        bst[:, 0] = c.pylon_xy[0] + jx
        bst[:, 1] = c.pylon_xy[1] + y_l * c0 - z_l * s0
        bst[:, 2] = c.hinge_h + y_l * s0 + z_l * c0
        bst[:, 3] = 1.0
        bst[:, 0:3] += origin
        self.ball.write_root_state_to_sim(bst, env_ids)

        # cubes: permuted slots + jitter + free yaw
        slots = torch.tensor(c.cube_slots, device=dev)  # (3, 2)
        perm_rank = torch.rand(m, c.n_cubes, device=dev).argsort(dim=1)  # (m, 3)
        yaw_amp = math.radians(c.cube_yaw_deg)
        for i, cube in enumerate(self.cubes):
            sl = slots[perm_rank[:, i]]  # (m, 2)
            cst = torch.zeros(m, 13, device=dev)
            cst[:, 0:2] = sl + (torch.rand(m, 2, device=dev) * 2 - 1) * c.cube_jitter
            cst[:, 2] = c.cube_edge / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            cst[:, 3] = torch.cos(half)
            cst[:, 6] = torch.sin(half)
            cst[:, 0:3] += origin
            cube.write_root_state_to_sim(cst, env_ids)

        self.cube_latch[env_ids] = 0.0
        self.raise_latch[env_ids] = 0.0
        self.cross_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"beam": self.beam, "ball": self.ball,
                  **{f"cube_{i}": b for i, b in enumerate(self.cubes)}}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "cube_latch": self.cube_latch[env_ids].clone(),
            "raise_latch": self.raise_latch[env_ids].clone(),
            "cross_latch": self.cross_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"beam": self.beam, "ball": self.ball,
                  **{f"cube_{i}": b for i, b in enumerate(self.cubes)}}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.cube_latch[env_ids] = state["cube_latch"]
        self.raise_latch[env_ids] = state["raise_latch"]
        self.cross_latch[env_ids] = state["cross_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A wooden SEE-SAW beam pivots on a dark pylon in the middle of the floor; "
            "its hinge lets it tilt about 15 degrees each way. The beam's far end "
            "carries an open-topped TRAY holding one large RED ball "
            f"({int(2000 * c.ball_r)} mm across — too wide for a parallel-jaw gripper "
            "to grasp; it can only be pushed). The beam's near end carries a tall "
            "dark square CHIMNEY-shaped hopper, open at the top. The beam rests with "
            "the heavy ball end DOWN and the empty hopper end UP. Scattered on the "
            f"floor near the hopper lie {c.n_cubes} BLUE cubes "
            f"({int(1000 * c.cube_edge)} mm, easily graspable); their positions and "
            "orientations change every episode — find them by looking. Beside the "
            "tray stands a tall GREEN pedestal shelf; its top carries a walled DOCK "
            "with dark-green rims, open on the side facing the see-saw except for a "
            "low sill ridge. The shelf top is far above the lowered tray and cannot "
            "be reached from the floor.\n"
            "Goal: drop the blue cubes one at a time into the beam's hopper chimney. "
            "Their accumulated weight tips the see-saw: the hopper end sinks and the "
            "tray end rises until the tray's low YELLOW lip comes level with the "
            "shelf top (one cube is never enough; all "
            f"{c.n_cubes} guarantee the lift). Then push the red ball sideways — "
            "toward the shelf, across the tray's yellow lip — so it rolls off the "
            "raised tray into the green dock, and leave it there at rest, well "
            "inside the dock walls. Only the red ball in the dock counts: a cube "
            "placed in the dock, the ball dropped to the floor beside the pedestal, "
            "the ball still sitting in the raised tray, or a still-moving ball all "
            "count for nothing. The order is forced by the mechanism: without "
            "enough ballast in the hopper the tray stays low and the shelf top is "
            "an unreachable wall."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the blue cubes into the see-saw's hopper chimney until the beam "
            "tips and the tray with the red ball rises to the green shelf, then push "
            "the red ball off the tray into the shelf's dock and leave it resting "
            "there. The ball is too big to grasp — push it, never lift it."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def beam_tilt(self) -> torch.Tensor:
        """(N,) beam tilt in degrees: positive = cargo (tray) end raised. Read from
        the beam's +y axis z-component (robust to numeric roll/yaw noise)."""
        from isaaclab.utils.math import quat_apply

        ey = torch.tensor([0.0, 1.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        ey_w = quat_apply(self.beam.data.root_quat_w, ey)
        return torch.rad2deg(torch.asin(ey_w[:, 2].clamp(-1.0, 1.0)))

    def raised(self) -> torch.Tensor:
        """(N,) bool: tray end raised to at least `raise_deg`."""
        return self.beam_tilt() >= self.cfg.raise_deg

    def lowered(self) -> torch.Tensor:
        """(N,) bool: tray end down at the lower stop band."""
        return self.beam_tilt() <= -self.cfg.raise_deg

    def _cube_pos_beam_frame(self) -> torch.Tensor:
        """(N, K, 3) cube centers in the beam's body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        n, k = self.env.num_envs, len(self.cubes)
        pos = torch.stack([b.data.root_pos_w for b in self.cubes], dim=1)  # (N,K,3)
        bq = self.beam.data.root_quat_w[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        bp = self.beam.data.root_pos_w[:, None, :]
        return quat_apply_inverse(bq, (pos - bp).reshape(n * k, 3)).reshape(n, k, 3)

    def cubes_in_hopper(self) -> torch.Tensor:
        """(N, K) bool, geometric: cube center inside the hopper chimney cavity
        (beam frame)."""
        c = self.cfg
        rel = self._cube_pos_beam_frame()
        return (rel[:, :, 0].abs() < c.hop_x_half + 0.02) \
            & (rel[:, :, 1] > c.hop_y[0] - 0.01) & (rel[:, :, 1] < c.hop_y[1] + 0.01) \
            & (rel[:, :, 2] > c.plank_top - 0.01) & (rel[:, :, 2] < c.hop_top + 0.03)

    def _ball_rel(self) -> torch.Tensor:
        return self.ball.data.root_pos_w - self.env_origins

    def ball_in_dock(self) -> torch.Tensor:
        """(N,) bool, geometric: ball center inside the dock window (env frame; the
        shelf is static). Accepts every physically-in-dock resting pose, rejects the
        raised-tray pose (x), the floor beside the pedestal (z), and gap-bridging
        perches (x) — asserted in __post_init__."""
        c = self.cfg
        rel = self._ball_rel()
        return (rel[:, 0] >= c.dock_x[0]) & (rel[:, 0] <= c.dock_x[1]) \
            & (rel[:, 1] >= c.dock_y[0]) & (rel[:, 1] <= c.dock_y[1]) \
            & (rel[:, 2] >= c.dock_z[0]) & (rel[:, 2] <= c.dock_z[1])

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball at rest."""
        return self.ball.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: per-cube containment
        in the hopper, beam raised, ball across onto the shelf."""
        c = self.cfg
        self.cube_latch = torch.maximum(self.cube_latch, self.cubes_in_hopper().float())
        self.raise_latch = torch.maximum(self.raise_latch, self.raised().float())
        rel = self._ball_rel()
        crossed = (rel[:, 0] > c.cross_x) & (rel[:, 2] > c.cross_z)
        self.cross_latch = torch.maximum(self.cross_latch, crossed.float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED ball rests inside the shelf dock, settled. Identity is
        by name — a ballast cube in the dock counts for nothing; the mechanism makes
        the ballast->raise->push chain physically necessary."""
        return self.ball_in_dock() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 per cube ever contained in the hopper + 0.25
        beam ever raised + 0.10 ball ever across onto the shelf (cap 0.65); 1.0 iff
        success(). Latched — credit never evaporates; the null policy scores ~0
        (beam spawns cargo-down, cubes on the ground)."""
        base = (0.10 * self.cube_latch.sum(dim=1) + 0.25 * self.raise_latch
                + 0.10 * self.cross_latch).clamp(0.0, 0.65)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="ballast_lift", robot="null", env_spacing=3.0))
