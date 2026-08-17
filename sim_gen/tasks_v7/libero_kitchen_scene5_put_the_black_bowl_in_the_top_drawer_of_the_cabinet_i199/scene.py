"""BatonStowScene — thread the too-long black baton into the drawer, then shut it.

Derived from libero_90/kitchen_scene5 "put the black bowl in the top drawer of the
cabinet" (pull the drawer open, pick the black object off the cabinet, drop it in —
judged by an object-in-open-drawer bbox test). The seed's terminal relation is kept
(the BLACK item ends inside the cabinet's sliding drawer) but the manipulation is
STRATEGICALLY inverted from "open, drop in, done":

  1. The cargo is a LONG BLACK BATON (20 x 3 x 3 cm) that starts on the CABINET ROOF,
     and the drawer's exposed opening is metrically SHORTER than the baton: the roof
     plate overhangs the drawer bay, so at full travel only ~15.7 cm of the drawer
     cavity is ever exposed (and the widest strip 3 cm wide that fits diagonally in
     the 15.7 x 12 cm aperture is ~16.3 cm). A flat drop-in — the seed's whole plan —
     is geometrically impossible at ANY yaw. The baton must be THREADED: tipped up
     ~40-50 deg, low end first into the exposed cavity, then slid aft UNDER the roof
     lip while pitching down along the clearance curve, and released only when its
     high end has also passed under the lip. Then it falls flat on the drawer floor.
  2. The goal state has the drawer PUSHED FULLY SHUT again with the baton enclosed —
     the seed (and its whole LIBERO family) ends with the drawer open. Success is
     only judged with travel <= 1 cm, so "open, insert" is not enough.
  3. A SHORT WHITE DECOY baton (12 cm — it DOES fit through the opening flat) starts
     next to the black one on the roof. It must end OUTSIDE the drawer: while the
     decoy sits in the cavity the score is capped near zero, so "shove both in" or
     "grab the easy one" fails.

Execution order is physically forced: nothing can enter the shut drawer (every gap
around the closed face is <= 4-6 mm, far under the 30 mm baton cross-section), so it
must be OPEN(drawer) -> THREAD(baton) -> SHUT(drawer). The slide is springless
(heavy damping = slide friction): open is a persistent state the solver must create
and then explicitly undo.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - shell: KINEMATIC compound — plinth (15 cm tall), side walls, back wall, and the
    ROOF PLATE (z = 22.8-24.0 cm, front edge exactly over the closed drawer's face)
    whose overhang creates the short aperture. The batons spawn on the roof top.
  - drawer: DYNAMIC compound — floor + four walls (interior 240 x 120 x 60 mm), a
    face plate sealing the front, and a protruding blue HANDLE (two posts + 14 mm
    crossbar). Origin at the front-bottom-centre of the box body.
  - drawer slide: bind-time UsdPhysics.PrismaticJoint shell->drawer along +X, limits
    [-travel, 0] (0 = shut), joint-pair collision disabled; no spring.
  - black baton (cargo) and white baton (decoy): dynamic boxes.

Per-episode randomization (readback-verifiable): Bernoulli LEFT/RIGHT roof-slot swap
of the two batons + per-baton xy jitter + per-baton random yaw (+/- 20 deg).

Rubric (0..1; progress latched so correct behavior never loses credit):
  0.15 * open_max   — latched max of drawer travel / open_norm (clamped 0..1)
  0.25 * thread     — latched bool: any black-baton END inside the drawer cavity
  0.30 * inside     — latched bool: BOTH black-baton ends inside the cavity
  0.15 * close_max  — latched max of shut-progress WHILE the baton is inside (live)
  cap 0.85 non-success; cap 0.10 while the white decoy currently sits in the cavity
  1.0 iff success() — both black ends in the cavity box, drawer shut (<= 1 cm),
                      decoy out, everything at rest. Physical outcomes only.

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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _spawn_shell(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet: KINEMATIC compound. Origin at the bay's front-bottom-centre
    (the closed drawer's box-front plane, x=0); the drawer pulls out toward -x.
    Parts: plinth (the drawer rides just above its top), side walls, back wall, and
    the ROOF PLATE whose front edge sits exactly at x=0 — the aperture maker."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    d = c.bay_depth
    xc = d / 2
    wall_z0, wall_z1 = c.plinth_h, c.roof_z0
    _add_box(stage, f"{prim_path}/plinth", center=(xc, 0.0, c.plinth_h / 2),
             size=(d, 2 * c.plinth_hw, c.plinth_h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(xc, sgn * (c.wall_y_in + c.wall_t / 2), (wall_z0 + wall_z1) / 2),
                 size=(d, c.wall_t, wall_z1 - wall_z0), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/back_wall",
             center=(d - c.wall_t / 2, 0.0, (wall_z0 + wall_z1) / 2),
             size=(c.wall_t, 2 * c.wall_y_in, wall_z1 - wall_z0), color=c.color,
             collide=collide)
    _add_box(stage, f"{prim_path}/roof",
             center=(c.roof_depth / 2, 0.0, (c.roof_z0 + c.roof_z1) / 2),
             size=(c.roof_depth, 2 * c.roof_hw, c.roof_z1 - c.roof_z0),
             color=c.roof_color, collide=collide)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer: DYNAMIC compound — open-top box (origin at the box's
    front-bottom-centre), face plate sealing the front, blue handle (posts +
    crossbar). Sleep/stabilization zeroed (force-driven on its slide)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(cfg.slide_damping))
    pxrb.CreateAngularDampingAttr(1.0)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    c = cfg
    t, D, W, H = c.t, c.depth, c.width, c.wall_top  # outer depth/width, wall-top height
    _add_box(stage, f"{prim_path}/floor", center=(D / 2, 0.0, t / 2),
             size=(D, W, t), color=c.color, collide=collide)
    wall_h = H - t
    _add_box(stage, f"{prim_path}/wall_front", center=(t / 2, 0.0, t + wall_h / 2),
             size=(t, W, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back", center=(D - t / 2, 0.0, t + wall_h / 2),
             size=(t, W, wall_h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(D / 2, sgn * (W / 2 - t / 2), t + wall_h / 2),
                 size=(D, t, wall_h), color=c.color, collide=collide)
    # face plate: seals the front while shut (all peripheral gaps << baton width)
    _add_box(stage, f"{prim_path}/face",
             center=((c.face_x0 + c.face_x1) / 2, 0.0, c.face_h / 2),
             size=(c.face_x1 - c.face_x0, c.face_w, c.face_h), color=c.face_color,
             collide=collide)
    # handle: two posts + crossbar (14 mm bar, graspable by a parallel jaw)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/post_{'l' if sgn > 0 else 'r'}",
                 center=((c.face_x0 + c.bar_x0) / 2, sgn * c.post_y, c.handle_z),
                 size=(c.face_x0 - c.bar_x0, 0.012, 0.012), color=c.handle_color,
                 collide=collide)
    _add_box(stage, f"{prim_path}/handle_bar",
             center=((c.bar_x0 + c.bar_x1) / 2, 0.0, c.handle_z),
             size=(c.bar_x1 - c.bar_x0, 2 * c.post_y + 0.024, 0.014),
             color=c.handle_color, collide=collide)
    return root


def _spawn_baton(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one baton: DYNAMIC single box (length along local +x)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
             size=(cfg.length, cfg.cross, cfg.cross), color=cfg.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shell" not in _SPAWNER_CACHE:

        @configclass
        class ShellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shell)
            bay_depth: float = 0.268
            plinth_h: float = 0.150
            plinth_hw: float = 0.080
            wall_y_in: float = 0.070
            wall_t: float = 0.010
            roof_z0: float = 0.228
            roof_z1: float = 0.240
            roof_depth: float = 0.280
            roof_hw: float = 0.140
            color: tuple = (0.45, 0.42, 0.40)
            roof_color: tuple = (0.58, 0.55, 0.52)
            contact_offset: float = 0.002

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            depth: float = 0.256
            width: float = 0.136
            wall_top: float = 0.068
            t: float = 0.008
            face_x0: float = -0.014
            face_x1: float = -0.002
            face_w: float = 0.140
            face_h: float = 0.070
            post_y: float = 0.035
            bar_x0: float = -0.054
            bar_x1: float = -0.040
            handle_z: float = 0.035
            handle_color: tuple = (0.10, 0.25, 0.80)
            face_color: tuple = (0.72, 0.62, 0.45)
            color: tuple = (0.80, 0.70, 0.50)
            slide_damping: float = 6.0
            contact_offset: float = 0.002

        @configclass
        class BatonSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_baton)
            length: float = 0.20
            cross: float = 0.030
            color: tuple = (0.05, 0.05, 0.05)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(shell=ShellSpawnerCfg, drawer=DrawerSpawnerCfg,
                              baton=BatonSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BatonStowSceneCfg(BaseCfg):
    """Config for `BatonStowScene`. The interlock is metric: the roof plate's front
    edge sits exactly over the shut drawer's box front, so the exposed cavity at full
    travel is (travel - t) = 157 mm long by 120 mm wide — and the longest 30 mm-wide
    strip that fits that aperture diagonally is ~163 mm, under the 200 mm black
    baton at every yaw. Flat insertion is impossible; the baton must be pitched up
    and threaded under the roof lip. Shut, every peripheral gap is 4-6 mm."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    open_norm: float = tunable(0.14)  # drawer travel (m) that counts as "fully open" credit
    close_tol: float = tunable(0.010)  # success: drawer travel must be <= this (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| of batons when judging (m/s)
    drawer_settle: float = tunable(0.03)  # max drawer |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    slot_jitter: float = tunable(0.02)  # baton spawn xy jitter (+/- m)
    yaw_max_deg: float = tunable(20.0)  # baton spawn yaw range (+/- deg)
    swap_slots: bool = tunable(True)  # Bernoulli left/right roof slot swap (demo sets False)

    # --- info: layout (single Franka base at the world origin; radii 0.20-0.70 m) ---------------
    front_x: float = info(0.42)  # shut drawer box-front plane (shell origin x)
    travel: float = info(0.165)  # slide travel (m); open = drawer at front_x - travel
    drawer_z: float = info(0.154)  # drawer origin height (rides the joint here)
    slot_x: float = info(0.15)  # roof slot centre, shell-local x
    slot_y: float = info(0.06)  # roof slots at y = +/- slot_y
    spawn_z: float = info(0.257)  # baton spawn centre z (roof top 0.240 + 15 + 2 mm)

    # --- info: cabinet structure ----------------------------------------------------------------
    bay_depth: float = info(0.268)
    plinth_h: float = info(0.150)
    plinth_hw: float = info(0.080)
    wall_y_in: float = info(0.070)  # side wall inner face |y|
    wall_t: float = info(0.010)
    roof_z0: float = info(0.228)  # roof plate bottom (world z; shell sits at z=0)
    roof_z1: float = info(0.240)  # roof plate top (batons spawn here)
    roof_depth: float = info(0.280)
    roof_hw: float = info(0.140)

    # --- info: drawer structure -----------------------------------------------------------------
    drawer_depth: float = info(0.256)  # outer (x); interior 0.240
    drawer_w: float = info(0.136)  # outer (y); interior 0.120
    drawer_wall_top: float = info(0.068)  # wall top above drawer origin; interior height 0.060
    drawer_t: float = info(0.008)
    drawer_mass: float = info(1.0)
    slide_damping: float = info(6.0)
    face_x0: float = info(-0.014)
    face_x1: float = info(-0.002)
    face_w: float = info(0.140)
    face_h: float = info(0.070)
    bar_x0: float = info(-0.054)  # handle crossbar front/back x (drawer-local)
    bar_x1: float = info(-0.040)
    post_y: float = info(0.035)
    handle_z: float = info(0.035)

    # --- info: batons ---------------------------------------------------------------------------
    black_len: float = info(0.20)
    decoy_len: float = info(0.12)
    baton_cross: float = info(0.030)
    black_mass: float = info(0.12)
    decoy_mass: float = info(0.07)
    end_off: float = info(0.094)  # black end-point sample offset from centre (m)

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.30 + 0.15 = 0.85 = the non-success cap)
    w_open: float = info(0.15)
    w_thread: float = info(0.25)
    w_inside: float = info(0.30)
    w_close: float = info(0.15)
    decoy_cap: float = info(0.10)  # score ceiling while the decoy sits in the cavity

    # Derived (filled in __post_init__).
    cavity_lo: tuple = field(default=None, init=False)  # drawer-local goal box for black ends
    cavity_hi: tuple = field(default=None, init=False)
    decoy_lo: tuple = field(default=None, init=False)  # expanded drawer-local decoy-in box
    decoy_hi: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        t, D, W, H = self.drawer_t, self.drawer_depth, self.drawer_w, self.drawer_wall_top
        m = 0.004
        self.cavity_lo = (t + m, -(W / 2 - t - m), t + 0.002)
        self.cavity_hi = (D - t - m, (W / 2 - t - m), H - 0.008)
        # decoy-in box: cavity grown 30 mm in xy; z capped at 0.090 so a baton resting
        # on the ROOF (drawer-local z ~0.101) or on the plinth/floor stays OUTSIDE.
        self.decoy_lo = (self.cavity_lo[0] - 0.03, self.cavity_lo[1] - 0.03, 0.0)
        self.decoy_hi = (self.cavity_hi[0] + 0.03, self.cavity_hi[1] + 0.03, 0.090)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("baton_stow")
class BatonStowScene(BaseScene):
    cfg: BatonStowSceneCfg

    def __init__(self, cfg: BatonStowSceneCfg | None = None) -> None:
        super().__init__(cfg or BatonStowSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        shell_spawn = spawners["shell"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bay_depth=c.bay_depth, plinth_h=c.plinth_h, plinth_hw=c.plinth_hw,
            wall_y_in=c.wall_y_in, wall_t=c.wall_t, roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            roof_depth=c.roof_depth, roof_hw=c.roof_hw, contact_offset=c.contact_offset,
        )
        drawer_spawn = spawners["drawer"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            depth=c.drawer_depth, width=c.drawer_w, wall_top=c.drawer_wall_top,
            t=c.drawer_t, face_x0=c.face_x0, face_x1=c.face_x1, face_w=c.face_w,
            face_h=c.face_h, post_y=c.post_y, bar_x0=c.bar_x0, bar_x1=c.bar_x1,
            handle_z=c.handle_z, slide_damping=c.slide_damping,
            contact_offset=c.contact_offset,
        )
        black_spawn = spawners["baton"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.black_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            length=c.black_len, cross=c.baton_cross, color=(0.05, 0.05, 0.05),
            contact_offset=c.contact_offset,
        )
        decoy_spawn = spawners["baton"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            length=c.decoy_len, cross=c.baton_cross, color=(0.92, 0.92, 0.92),
            contact_offset=c.contact_offset,
        )
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
            "shell": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shell",
                spawn=shell_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.front_x, 0.0, 0.0)),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.front_x, 0.0, c.drawer_z)),
            ),
            "black": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlackBaton",
                spawn=black_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.front_x + c.slot_x, c.slot_y, c.spawn_z)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhiteBaton",
                spawn=decoy_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.front_x + c.slot_x, -c.slot_y, c.spawn_z)),
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
        self.shell: RigidObject = env.iscene["shell"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.black: RigidObject = env.iscene["black"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_slide()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements
        self._open_max = torch.zeros(n, device=dev)
        self._thread = torch.zeros(n, dtype=torch.bool, device=dev)
        self._inside = torch.zeros(n, dtype=torch.bool, device=dev)
        self._close_max = torch.zeros(n, device=dev)
        # External drive inputs (solve.py writes; post_step consumes and OWNS the
        # bodies' external-wrench slots).
        self.drawer_drive = torch.zeros(n, device=dev)  # force along +x (N); pull = negative
        self.black_force_w = torch.zeros(n, 3, device=dev)  # world force on the black baton
        self.black_torque_w = torch.zeros(n, 3, device=dev)  # world torque on the black baton

    def _author_slide(self) -> None:
        """Per env: a +X prismatic joint shell->drawer, limits [-travel, 0]
        (0 = shut), joint-pair collision disabled (the joint owns the drawer-shell
        relation; the shell still blocks the batons everywhere)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Shell"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.drawer_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.travel))
            j.CreateUpperLimitAttr(0.0)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: drawer shut, the two batons on the roof at Bernoulli-swapped
        slots with xy jitter and random yaw, latches cleared. The shell CANNOT be
        pose-randomized (kinematic joint anchors stay world-fixed), so the batons
        carry all the episode-to-episode variation."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- shell (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.front_x
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shell.write_root_state_to_sim(st, env_ids)

        # --- drawer: shut (joint coordinate 0) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 2] = c.front_x, c.drawer_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.drawer.write_root_state_to_sim(st, env_ids)

        # --- batons: slot swap + jitter + yaw ---
        # (torch.rand-based draws: the first randint after manual_seed is degenerate)
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        y_sign = torch.where(swap, -torch.ones(m, device=dev), torch.ones(m, device=dev))
        for baton, sgn in ((self.black, 1.0), (self.decoy, -1.0)):
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.front_x + c.slot_x + jit[:, 0]
            st[:, 1] = sgn * y_sign * c.slot_y + jit[:, 1]
            st[:, 2] = c.spawn_z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            baton.write_root_state_to_sim(st, env_ids)

        # --- clear latches + drives ---
        self._open_max[env_ids] = 0.0
        self._thread[env_ids] = False
        self._inside[env_ids] = False
        self._close_max[env_ids] = 0.0
        self.drawer_drive[env_ids] = 0.0
        self.black_force_w[env_ids] = 0.0
        self.black_torque_w[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shell": self.shell.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "black": self.black.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "open_max": self._open_max[env_ids].clone(),
            "thread": self._thread[env_ids].clone(),
            "inside": self._inside[env_ids].clone(),
            "close_max": self._close_max[env_ids].clone(),
            "drawer_drive": self.drawer_drive[env_ids].clone(),
            "black_force_w": self.black_force_w[env_ids].clone(),
            "black_torque_w": self.black_torque_w[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shell.write_root_state_to_sim(state["shell"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.black.write_root_state_to_sim(state["black"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._open_max[env_ids] = state["open_max"]
        self._thread[env_ids] = state["thread"]
        self._inside[env_ids] = state["inside"]
        self._close_max[env_ids] = state["close_max"]
        self.drawer_drive[env_ids] = state["drawer_drive"]
        self.black_force_w[env_ids] = state["black_force_w"]
        self.black_torque_w[env_ids] = state["black_torque_w"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        ap = c.travel - c.drawer_t
        return (
            f"A gray floor cabinet faces you. Its single sliding DRAWER (tan, interior "
            f"{(c.drawer_depth - 2 * c.drawer_t) * 100:.0f} x "
            f"{(c.drawer_w - 2 * c.drawer_t) * 100:.0f} x "
            f"{(c.drawer_wall_top - c.drawer_t) * 100:.0f} cm) starts fully SHUT behind "
            f"a face plate with a BLUE HANDLE BAR; it slides straight out toward you "
            f"(travel {c.travel * 100:.1f} cm) and stays wherever it is left. A fixed "
            f"ROOF PLATE overhangs the whole drawer bay with its front edge directly "
            f"above the shut drawer's face, so even fully open, only the front "
            f"~{ap * 100:.1f} cm of the drawer cavity is ever exposed. On TOP of the "
            f"roof lie two batons ({c.baton_cross * 100:.0f} cm square cross-section): "
            f"a LONG BLACK baton ({c.black_len * 100:.0f} cm) and a SHORT WHITE baton "
            f"({c.decoy_len * 100:.0f} cm); which lies left and which lies right varies "
            f"between episodes, as do their exact positions and yaws.\n"
            f"Goal: the LONG BLACK baton must end up lying INSIDE the drawer, and the "
            f"drawer must be pushed FULLY SHUT again (within {c.close_tol * 1000:.0f} mm). "
            f"The black baton is LONGER than the exposed opening ever gets — it cannot "
            f"be dropped in flat at any angle. Thread it: tip it up steeply, lower its "
            f"low end into the exposed cavity, then slide that end back UNDER the roof "
            f"lip while leveling out, and let it drop flat only once it fits. Nothing "
            f"passes into the shut drawer (all gaps are a few mm), so the drawer must "
            f"be opened first and shut last. The SHORT WHITE baton is a decoy: it "
            f"would fit through the opening flat, but it must stay OUT of the drawer — "
            f"leaving it inside forfeits nearly all credit. Leave everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the drawer open by its blue handle, take the long black baton off "
            "the cabinet roof, thread it tilted through the opening under the roof "
            "lip so it lies flat inside the drawer, then push the drawer fully shut. "
            "Leave the short white baton out of the drawer."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def drawer_open(self) -> torch.Tensor:
        """(N,) drawer travel in m (0 = shut, travel = fully out)."""
        x = (self.drawer.data.root_pos_w - self.env_origins)[:, 0]
        return (self.cfg.front_x - x).clamp(min=0.0)

    def black_endpoints(self) -> torch.Tensor:
        """(N, 2, 3) world positions of the two black-baton end points (centre
        +/- end_off along the baton's long axis)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        ax = quat_apply(self.black.data.root_quat_w, ex)
        p = self.black.data.root_pos_w
        return torch.stack([p + c.end_off * ax, p - c.end_off * ax], dim=1)

    def endpoints_in_cavity(self) -> torch.Tensor:
        """(N, 2) bool: black end points inside the drawer's goal box (drawer frame;
        the prismatic joint locks rotation, so this is a translation)."""
        c = self.cfg
        loc = self.black_endpoints() - self.drawer.data.root_pos_w.unsqueeze(1)
        lo = torch.tensor(c.cavity_lo, device=loc.device)
        hi = torch.tensor(c.cavity_hi, device=loc.device)
        return ((loc >= lo) & (loc <= hi)).all(dim=-1)

    def black_inside(self) -> torch.Tensor:
        """(N,) bool: BOTH black end points inside the goal box (live)."""
        return self.endpoints_in_cavity().all(dim=1)

    def decoy_in_cavity(self) -> torch.Tensor:
        """(N,) bool: decoy centre inside the expanded drawer-local cavity box —
        the forbidden state (in the cavity or perched on the drawer's rim)."""
        c = self.cfg
        loc = self.decoy.data.root_pos_w - self.drawer.data.root_pos_w
        lo = torch.tensor(c.decoy_lo, device=loc.device)
        hi = torch.tensor(c.decoy_hi, device=loc.device)
        return ((loc >= lo) & (loc <= hi)).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        op = self.drawer_open()
        prog = torch.nan_to_num((op / c.open_norm).clamp(0.0, 1.0), nan=0.0)
        self._open_max = torch.maximum(self._open_max, prog)
        ep_in = self.endpoints_in_cavity()
        self._thread |= ep_in.any(dim=1)
        inside = ep_in.all(dim=1)
        self._inside |= inside
        close = (1.0 - op / c.open_norm).clamp(0.0, 1.0) * inside.float()
        self._close_max = torch.maximum(self._close_max, torch.nan_to_num(close, nan=0.0))

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Drive plants: the drawer's slide force (the applied-wrench emulation of
        the arm on the blue handle) and the black baton's held wrench (the emulation
        of the arm gripping the baton), then latch rubric progress. Owns both
        bodies' external-wrench slots. Baton wrench is commanded in the WORLD frame
        and converted per-substep (external wrenches apply in the body frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        dev = self.env.device
        force = torch.zeros(n, 1, 3, device=dev)
        force[:, 0, 0] = self.drawer_drive
        self.drawer.set_external_force_and_torque(
            force, torch.zeros(n, 1, 3, device=dev))
        q = self.black.data.root_quat_w
        f_b = quat_apply_inverse(q, self.black_force_w)
        t_b = quat_apply_inverse(q, self.black_torque_w)
        self.black.set_external_force_and_torque(f_b.unsqueeze(1), t_b.unsqueeze(1))
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: both black end points inside the drawer cavity box, drawer shut
        (travel <= close_tol), decoy OUT of the cavity, batons and drawer at rest.
        Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        shut = self.drawer_open() <= c.close_tol
        still = (
            (self.black.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
            & (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.drawer_settle)
        )
        return self.black_inside() & shut & ~self.decoy_in_cavity() & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*open_max + 0.25*thread + 0.30*inside +
        0.15*close_max (all latched, ~0 for doing nothing), capped 0.85; capped at
        decoy_cap while the decoy currently sits in the cavity; exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._open_max + c.w_thread * self._thread.float()
                + c.w_inside * self._inside.float()
                + c.w_close * self._close_max).clamp(max=0.85)
        base = torch.where(self.decoy_in_cavity(), base.clamp(max=c.decoy_cap), base)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="baton_stow", robot="null"))
