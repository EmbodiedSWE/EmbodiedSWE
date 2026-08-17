"""DetergentDrawerScene — dose the washer: open the detergent drawer, drop the pod
into the BLUE-marked main-wash compartment, slide the drawer fully shut.

Derived from the RLBench `open_washing_machine` seed but STRATEGICALLY DIFFERENT
(see TASK.md): the seed's whole plan is ONE pull on a hinged door — grab the
handle, swing it open, done; the articulation's final angle IS the goal. Here the
articulation is a PRISMATIC detergent drawer and moving it is never the goal —
it is the gate for a selective deposit and must end where it STARTED (shut). The
plan has three ordered phases: (1) pull the drawer out of the console mouth,
(2) drop the green detergent pod into the correct one of its two compartments —
the side marked by the BLUE tile on the console face (the white tile marks the
softener side, which must stay empty), (3) push the drawer fully shut so the pod
ends up sealed under the console hood. The ordering is enforced by GEOMETRY, not
by rubric fiat: when the drawer is shut the hood leaves a 6 mm slit over the
cells, so a 22 mm pod physically cannot enter (smoke proves the drop bounces off
the hood), and a pod deposited correctly but left in an open drawer is not a
success either.

Mechanics (compound rigid bodies via custom spawn funcs — the schemas-in-func
pattern — plus a spawn-authored PrismaticJoint):
  - console: ONE kinematic compound (bottom block, two pillars, hood, rear wall)
    forming a rectangular mouth; never teleported, so the joint anchor is safe;
  - drawer: ONE dynamic compound (base plate, four walls, center divider — two
    open-top cells — and a front handle plate on a neck) hung on a PrismaticJoint
    (console -> drawer, local +x, limits [0, stroke]); the joint carries it, so
    the only degree of freedom is the pull axis; linear damping parks it where
    released (no spring: shut is a place you must PUSH it back to);
  - pod: a plain dynamic cube on the table;
  - two kinematic marker tiles on the console face, SWAPPED per episode: blue
    over the sampled main-wash cell, white over the softener cell.
`post_step` owns the wrench slots: it applies the `drive_f` (drawer, pull axis)
and `pod_f` (probe) buffers and latches rubric progress.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  success() = drawer SHUT (opening < shut_tol) AND pod resting inside the
  target cell (judged in the DRAWER'S BODY FRAME, so the pod rides the moving
  drawer without losing containment) AND everything settled.
  score() = 1.0 iff success(); else 0.20*opened_latch (drawer was pulled past
  open_thresh — latched, because correct behavior must UNDO the opening)
  + 0.25*deposit_latch (pod seen at rest in the target cell — latched through a
  slow gate so a pod flying through the cell volume earns nothing)
  + 0.20*in_cell_now (live containment; rides through the closing push).
  ~0 for the null policy; latched credit never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): the main-wash SIDE
(left/right, tiles physically swapped), the pod's start pose (x, y incl. side,
yaw), and the drawer's initial crack (0..12 mm — always well short of the 90 mm
open latch and too narrow to admit the pod).

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


# ----- compound spawn funcs ------------------------------------------------------------------------
def _spawn_console(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The washer console: KINEMATIC compound around a rectangular drawer mouth.
    Local frame: origin at the footprint centre ON the table top; front face at
    local +x = depth/2. Children: bottom block (mouth floor level), two side
    pillars, the HOOD (roof sealing the cells when the drawer is shut) and a rear
    wall (no dosing from behind)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    collide = _make_collide(c.contact_offset)
    dx, dy = c.depth, c.width
    body = c.body_color
    _add_box(stage, f"{prim_path}/bottom", center=(0.0, 0.0, c.bottom_h / 2),
             size=(dx, dy, c.bottom_h), color=body, collide=collide)
    pil_y = c.mouth_w / 2 + c.pillar_w / 2
    for tag, sy in (("pillar_l", -1.0), ("pillar_r", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}",
                 center=(0.0, sy * pil_y, c.bottom_h + c.mouth_h / 2),
                 size=(dx, c.pillar_w, c.mouth_h), color=body, collide=collide)
    _add_box(stage, f"{prim_path}/hood",
             center=(0.0, 0.0, c.bottom_h + c.mouth_h + c.hood_h / 2),
             size=(dx, dy, c.hood_h), color=c.hood_color, collide=collide)
    _add_box(stage, f"{prim_path}/rear",
             center=(-dx / 2 + c.rear_t / 2, 0.0, c.bottom_h + c.mouth_h / 2),
             size=(c.rear_t, c.mouth_w, c.mouth_h), color=body, collide=collide)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The detergent drawer: ONE dynamic compound — base plate (body origin), four
    walls, a center divider splitting the tray into LEFT/RIGHT open-top cells, and
    the front handle (neck + upright grip plate). Spawn-authors the PrismaticJoint
    to the sibling console: axis local +x, limits [0, stroke] (0 = shut, flush
    with the console face). Joint-pair collision filtered — the joint IS the rail."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(c.lin_damping))
    pxrb.CreateAngularDampingAttr(5.0)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    D, W, t, h = c.tray_d, c.tray_w, c.wall_t, c.wall_h
    zc = t / 2 + h / 2  # wall centre above the plate mid-plane
    grey = c.tray_color
    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(D, W, t), color=grey, collide=collide)
    _add_box(stage, f"{prim_path}/wall_f", center=(D / 2 - t / 2, 0.0, zc),
             size=(t, W, h), color=grey, collide=collide)
    _add_box(stage, f"{prim_path}/wall_b", center=(-D / 2 + t / 2, 0.0, zc),
             size=(t, W, h), color=grey, collide=collide)
    for tag, sy in (("wall_l", -1.0), ("wall_r", 1.0)):
        _add_box(stage, f"{prim_path}/{tag}", center=(0.0, sy * (W / 2 - t / 2), zc),
                 size=(D - 2 * t, t, h), color=grey, collide=collide)
    _add_box(stage, f"{prim_path}/divider", center=(0.0, 0.0, zc),
             size=(D - 2 * t, t, h), color=(0.5, 0.5, 0.54), collide=collide)
    # handle: neck out of the front wall + upright grip plate (the jaw target)
    _add_box(stage, f"{prim_path}/neck",
             center=(D / 2 + c.neck_len / 2, 0.0, c.handle_z),
             size=(c.neck_len, 0.016, 0.012), color=grey, collide=collide)
    _add_box(stage, f"{prim_path}/grip",
             center=(D / 2 + c.neck_len + c.grip_t / 2, 0.0, c.handle_z),
             size=(c.grip_t, c.grip_w, c.grip_h), color=c.grip_color, collide=collide)
    # --- the rail: PrismaticJoint to the sibling console, axis +x, [0, stroke] ---
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/rail")
    j.CreateBody0Rel().SetTargets([f"{base}/Console"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.anchor_x), 0.0, float(c.anchor_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(c.stroke))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "console" not in _SPAWNER_CACHE:

        @configclass
        class ConsoleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_console)
            depth: float = 0.22
            width: float = 0.34
            bottom_h: float = 0.06
            mouth_w: float = 0.21
            mouth_h: float = 0.06
            pillar_w: float = 0.065
            hood_h: float = 0.08
            rear_t: float = 0.012
            body_color: tuple = (0.82, 0.83, 0.86)
            hood_color: tuple = (0.30, 0.32, 0.38)
            contact_offset: float = 0.002

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            tray_d: float = 0.16
            tray_w: float = 0.19
            wall_t: float = 0.008
            wall_h: float = 0.045
            neck_len: float = 0.020
            grip_t: float = 0.008
            grip_w: float = 0.056
            grip_h: float = 0.032
            handle_z: float = 0.024
            mass: float = 0.30
            lin_damping: float = 5.0
            anchor_x: float = 0.03  # drawer-shut origin, in console-local coords
            anchor_z: float = 0.065
            stroke: float = 0.13
            tray_color: tuple = (0.62, 0.63, 0.66)
            grip_color: tuple = (0.20, 0.20, 0.24)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["console"] = ConsoleSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class DetergentDrawerSceneCfg(BaseCfg):
    """Config for `DetergentDrawerScene`. Geometry is derived once in `__post_init__`
    so the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    shut_tol: float = tunable(0.006)  # opening below this = drawer shut (m)
    open_thresh: float = tunable(0.090)  # opening past this latches "was opened" (m)
    cell_margin: float = tunable(0.006)  # containment inset from the cell walls (m)
    slow_gate: float = tunable(0.10)  # m/s: pod/drawer count as slow below this (latch gate)
    settle_lin: float = tunable(0.05)  # max |lin vel| (drawer AND pod) when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    crack_max: float = tunable(0.012)  # drawer initial opening sampled in [0, this] (m)
    pod_x_range: tuple = tunable((0.13, 0.25))  # pod spawn x band (env-local, m)
    pod_y_band: tuple = tunable((0.13, 0.23))  # pod spawn |y| band (side sampled +-)

    # --- tunable: plant -------------------------------------------------------------------------
    drawer_mass: float = tunable(0.30)
    drawer_damping: float = tunable(5.0)  # parks the drawer where released
    pod_mass: float = tunable(0.020)

    # --- info: structure (env-local coordinates) ------------------------------------------------
    table_center: tuple = info((0.10, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.00, 0.08))  # top at z = 0.40
    console_pos: tuple = info((-0.02, 0.0))  # console origin on the table top
    console_depth: float = info(0.22)
    console_width: float = info(0.34)
    bottom_h: float = info(0.06)  # mouth floor at table + this
    mouth_h: float = info(0.06)
    hood_h: float = info(0.08)
    tray_d: float = info(0.16)
    tray_w: float = info(0.19)
    wall_t: float = info(0.008)
    wall_h: float = info(0.045)
    stroke: float = info(0.13)
    neck_len: float = info(0.020)
    grip_t: float = info(0.008)
    pod_size: float = info(0.022)
    marker_size: tuple = info((0.006, 0.050, 0.030))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    table_top_z: float = field(default=None, init=False)
    drawer_z: float = field(default=None, init=False)  # drawer body-origin height
    drawer_shut_x: float = field(default=None, init=False)  # body-origin x when shut
    front_face_x: float = field(default=None, init=False)  # console front face plane
    hood_bot_z: float = field(default=None, init=False)
    hood_top_z: float = field(default=None, init=False)
    cell_y_c: float = field(default=None, init=False)  # |cell centre y| (drawer-local)
    rim_z: float = field(default=None, init=False)  # cell rim height (world, drawer level)
    marker_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.table_top_z = self.table_center[2] + self.table_size[2] / 2
        self.front_face_x = self.console_pos[0] + self.console_depth / 2
        self.drawer_z = self.table_top_z + 0.065  # = console anchor_z above the table top
        self.drawer_shut_x = self.console_pos[0] + 0.03  # front wall flush with the face
        self.hood_bot_z = self.table_top_z + self.bottom_h + self.mouth_h
        self.hood_top_z = self.hood_bot_z + self.hood_h
        # cells: y from wall_t/2 (divider face) to tray_w/2 - wall_t (side wall face)
        self.cell_y_c = (self.wall_t / 2 + self.tray_w / 2 - self.wall_t) / 2 + 0.0
        self.rim_z = self.drawer_z + self.wall_t / 2 + self.wall_h
        self.marker_z = self.hood_bot_z + self.hood_h / 2


# ----- scene ----------------------------------------------------------------------------------------
class DetergentDrawerScene(BaseScene):
    cfg: DetergentDrawerSceneCfg

    def __init__(self, cfg: DetergentDrawerSceneCfg | None = None) -> None:
        super().__init__(cfg or DetergentDrawerSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table, the console + drawer compounds (console
        FIRST — the drawer's spawn-authored joint targets its sibling), the pod, and
        the two kinematic marker tiles (re-posed at reset to the sampled side)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        console_spawn = spawners["console"](
            depth=c.console_depth, width=c.console_width, bottom_h=c.bottom_h,
            mouth_h=c.mouth_h, hood_h=c.hood_h, contact_offset=c.contact_offset)
        drawer_spawn = spawners["drawer"](
            tray_d=c.tray_d, tray_w=c.tray_w, wall_t=c.wall_t, wall_h=c.wall_h,
            neck_len=c.neck_len, grip_t=c.grip_t, mass=c.drawer_mass,
            lin_damping=c.drawer_damping, stroke=c.stroke, contact_offset=c.contact_offset)

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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "console": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Console",
                spawn=console_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.console_pos[0], c.console_pos[1], c.table_top_z)),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.drawer_shut_x, 0.0, c.drawer_z)),
            ),
            "pod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pod",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pod_size, c.pod_size, c.pod_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        linear_damping=0.2, angular_damping=0.2,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pod_mass),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.72, 0.30))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.20, 0.18, c.table_top_z + c.pod_size / 2 + 0.003)),
            ),
        }
        for name, color, y0 in (("marker_blue", (0.10, 0.30, 0.90), -1.0),
                                ("marker_white", (0.92, 0.92, 0.92), 1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("MarkerBlue" if "blue" in name else "MarkerWhite"),
                spawn=sim_utils.CuboidCfg(
                    size=c.marker_size, rigid_props=kin, collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.front_face_x + c.marker_size[0] / 2, y0 * c.cell_y_c, c.marker_z)),
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
        self.drawer: RigidObject = env.iscene["drawer"]
        self.pod: RigidObject = env.iscene["pod"]
        self.console: RigidObject = env.iscene["console"]
        self.markers = {"blue": env.iscene["marker_blue"], "white": env.iscene["marker_white"]}
        self.env_origins = env.iscene.env_origins
        # Episode state.
        self.side = torch.ones(n, device=dev)  # +1 = main-wash cell on +y, -1 on -y
        self.crack0 = torch.zeros(n, device=dev)  # sampled initial opening
        self.pod_start = torch.zeros(n, 3, device=dev)  # sampled pod start (env-local)
        self.opened_latch = torch.zeros(n, device=dev)  # was pulled past open_thresh
        self.deposit_latch = torch.zeros(n, device=dev)  # pod seen at rest in the cell
        # External drive input (solve.py and smoke probes write; post_step consumes +
        # owns the wrench slots — never call set_external_force_and_torque directly).
        self.drive_f = torch.zeros(n, device=dev)  # force on the drawer along +x (N)
        self.pod_f = torch.zeros(n, 3, device=dev)  # probe force on the pod (N)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the main-wash SIDE (tiles physically swapped), the
        drawer's initial crack and the pod's start pose; clear latches and drives.
        Uses torch.rand comparisons throughout (the first randint after manual_seed
        is degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        crack = torch.rand(m, device=dev) * c.crack_max
        px = c.pod_x_range[0] + torch.rand(m, device=dev) * (c.pod_x_range[1] - c.pod_x_range[0])
        pod_side = torch.where(torch.rand(m, device=dev) < 0.5,
                               -torch.ones(m, device=dev), torch.ones(m, device=dev))
        py = pod_side * (c.pod_y_band[0] + torch.rand(m, device=dev)
                         * (c.pod_y_band[1] - c.pod_y_band[0]))
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi

        self.side[env_ids] = side
        self.crack0[env_ids] = crack
        self.opened_latch[env_ids] = 0.0
        self.deposit_latch[env_ids] = 0.0
        self.drive_f[env_ids] = 0.0
        self.pod_f[env_ids] = 0.0

        # drawer: shut + sampled crack, identity orientation, zero velocity
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.drawer_shut_x + crack
        st[:, 2] = c.drawer_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.drawer.write_root_state_to_sim(st, env_ids)

        # pod: sampled table pose
        pz = c.table_top_z + c.pod_size / 2 + 0.003
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = pz
        half = yaw / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.pod.write_root_state_to_sim(st, env_ids)
        self.pod_start[env_ids] = torch.stack([px, py, torch.full_like(px, pz)], dim=1)

        # marker tiles: blue over the target cell, white over the other
        for name, s in (("blue", side), ("white", -side)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.front_face_x + c.marker_size[0] / 2
            st[:, 1] = s * c.cell_y_c
            st[:, 2] = c.marker_z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.markers[name].write_root_state_to_sim(st, env_ids)

    # ----- readings ------------------------------------------------------------------------------
    def opening(self) -> torch.Tensor:
        """(N,) drawer opening in m (0 = shut; the joint caps it at `stroke`)."""
        return (self.drawer.data.root_pos_w[:, 0] - self.env_origins[:, 0]
                - self.cfg.drawer_shut_x)

    def pod_local(self) -> torch.Tensor:
        """(N, 3) pod centre in the DRAWER'S BODY FRAME (origin = base-plate centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.pod.data.root_pos_w - self.drawer.data.root_pos_w
        return quat_apply_inverse(self.drawer.data.root_quat_w, rel)

    def in_cell(self) -> torch.Tensor:
        """(N,) bool: pod centre inside the TARGET cell's inner box (drawer frame).
        z window rejects a pod perched on the rim/divider; the containment rides
        the moving drawer by construction."""
        c = self.cfg
        p = self.pod_local()
        m = c.cell_margin
        x_ok = p[:, 0].abs() < (c.tray_d / 2 - c.wall_t - m)
        y_s = p[:, 1] * self.side  # >0 on the target side
        y_ok = (y_s > c.wall_t / 2 + 0.002) & (y_s < c.tray_w / 2 - c.wall_t - 0.002)
        z_ok = (p[:, 2] > 0.004) & (p[:, 2] < c.wall_t / 2 + c.wall_h + 0.008)
        return x_ok & y_ok & z_ok

    def shut_now(self) -> torch.Tensor:
        """(N,) bool: drawer opening below `shut_tol`."""
        return self.opening() < self.cfg.shut_tol

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer AND pod |lin vel| below `settle_lin`."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.pod.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def success(self) -> torch.Tensor:
        """(N,) bool: drawer SHUT + pod resting inside the blue-marked cell +
        everything settled (current, physical state)."""
        return self.shut_now() & self.in_cell() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.20*opened_latch
        + 0.25*deposit_latch + 0.20*in_cell (live). ~0 for the null policy;
        latched credit never evaporates under correct behavior (the deposit
        latch survives the closing push; the opened latch survives re-shutting,
        which correct behavior REQUIRES)."""
        partial = 0.20 * self.opened_latch + 0.25 * self.deposit_latch \
            + 0.20 * self.in_cell().float()
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self) -> None:
        """Apply the `drive_f` / `pod_f` buffers (owns the wrench slots), then latch
        rubric progress: `opened_latch` on geometric opening past `open_thresh`;
        `deposit_latch` only while pod AND drawer are slow (a pod flying through
        the cell volume latches nothing)."""
        n = self.env.num_envs
        dev = self.env.device
        f = torch.zeros(n, 1, 3, device=dev)
        f[:, 0, 0] = self.drive_f
        self.drawer.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))
        fp = torch.zeros(n, 1, 3, device=dev)
        fp[:, 0, :] = self.pod_f
        self.pod.set_external_force_and_torque(fp, torch.zeros(n, 1, 3, device=dev))

        g = self.opening()
        opened = (g >= self.cfg.open_thresh) & torch.isfinite(g)
        self.opened_latch = torch.maximum(self.opened_latch, opened.float())
        slow = (self.pod.data.root_lin_vel_w.norm(dim=-1) < self.cfg.slow_gate) \
            & (self.drawer.data.root_lin_vel_w.norm(dim=-1) < self.cfg.slow_gate)
        dep = self.in_cell() & slow
        self.deposit_latch = torch.maximum(self.deposit_latch, dep.float())

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "crack0", "pod_start", "opened_latch",
                               "deposit_latch", "drive_f", "pod_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"drawer": self.drawer, "pod": self.pod,
                "marker_blue": self.markers["blue"], "marker_white": self.markers["white"]}

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pale washing-machine console stands on a wooden table. Set into its front "
            f"face, under a dark hood, is a sliding DETERGENT DRAWER with a dark handle "
            f"plate sticking straight out at you; the drawer pulls OUT toward you by up to "
            f"{c.stroke * 100:.0f} cm and slides back flush. Its tray is split by a center "
            f"divider into two identical open-top compartments, left and right. On the "
            f"console face just above the drawer sit two small tiles: a BLUE tile over one "
            f"compartment and a WHITE tile over the other. The blue tile marks the "
            f"MAIN-WASH compartment; which side it is on changes every episode. A green "
            f"detergent pod (a {c.pod_size * 1000:.0f} mm cube) lies on the table nearby; "
            f"its position also changes every episode, and the drawer may start shut or "
            f"cracked open by a centimetre.\n"
            f"Goal: pull the drawer out by its handle, drop the green pod into the "
            f"compartment on the BLUE tile's side, then push the drawer fully shut (handle "
            f"flush against the console face) and let go. When the drawer is shut the hood "
            f"seals the compartments, so the pod can only go in while the drawer is open. "
            f"A pod in the white-marked softener compartment fails; a correctly dosed but "
            f"open or half-open drawer is not done — only a shut drawer with the pod "
            f"resting inside the blue-marked compartment counts."
        )

    def instruction(self) -> str:
        return (
            "Pull the washer's detergent drawer open by its handle, drop the green pod "
            "into the compartment on the blue tile's side (not the white side), then "
            "push the drawer fully shut."
        )


# Guarded registration: the forge may import this module under two names.
if "detergent_drawer" not in SCENES.list():
    SCENES.register("detergent_drawer", DetergentDrawerScene)
if "simgen.detergent_drawer" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="detergent_drawer", robot="null"))
