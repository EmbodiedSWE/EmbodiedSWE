"""WardedSpindleScene — thread a finned sleeve DOWN a post past two notched ward
shelves (rotate to each notch, lower through, seat on the base plinth).

Derived from the RLBench `beat_the_buzz` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed is the buzz-wire game — grasp a wand and guide its loop along a
bent wire END-TO-END while NEVER touching it; the whole skill is clearance
maximization along a continuous curve, and contact = instant failure. Here the
constraint is INVERTED into a warded lock: the captive sleeve CANNOT move down the
post at all unless its fin is rotated into the open notch of the next ward shelf —
contact with the wards is the normal, expected state (the sleeve RESTS on them),
and progress happens only through a discrete align-then-lower engagement executed
twice, at two independently randomized notch angles. Avoidance and clearance
control are worthless; the skill is reading two notch angles, a rotation servo
about the post axis, and a controlled lower through each notch. No buzzer, no
keep-out band, no continuous guarded sweep.

Mechanics (all real rigid-body contact; nothing scripted): the post and plinth are
kinematic; the two ward shelves are kinematic annular collars (12 tangential boxes
covering 300 deg, one 60 deg gap) whose YAW is sampled per episode; the sleeve is
one dynamic compound body — an octagonal hub (8 boxes) that stays threaded around
the post plus a radial fin. Misaligned, the fin lands on a shelf top and the
sleeve simply rests there (a stable, settled state); aligned, the fin passes
through the gap and the sleeve descends to the next rest. The bottom seat is
reachable ONLY through both wards: the hub is a closed loop around the post, so
there is no lateral path — the topology forces upper ward, then lower ward, then
seat (execution order is geometric, not conventional).

Rubric (graded 0..1, latching transient achievement — anchored in the solve.py
trajectory: pass ward 1, pass ward 2, seat):
  - `p1_latch`  (0.30): sleeve centre has been below the upper-ward pass plane
    while THREADED on the post;
  - `p2_latch`  (0.30): same for the lower-ward pass plane;
  - `seat_latch` (0.20): sleeve has rested seated on the plinth (threaded,
    upright, settled);
  - success (-> 1.0): sleeve currently seated on the plinth, threaded, upright,
    and settled. score() == 1.0 iff success(); null policy ~0. Latched credit
    never evaporates under correct behavior. All latches require the THREADED
    readback, so a sleeve taken off the post top and dropped near the base (the
    only bypass the topology leaves) earns nothing.

Per-episode randomization (readback-verified in smoke): tower xy, upper-ward
notch angle th1 (full circle), lower-ward notch offset th2-th1 (|.|
in [55, 180] deg, either sign), and the sleeve's initial fin angle (misaligned
from th1 by >= 55 deg, either sign).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----- custom compound spawners -----------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable, yaw: float = 0.0):
    """One box child: translate (+ optional yaw) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        xf.AddRotateZOp().Set(math.degrees(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _bind_mat(prim_path: str, child: str, static: float, dynamic: float) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_path = f"{prim_path}/physMat_{child.rsplit('/', 1)[-1]}"
    sim_utils.spawn_rigid_body_material(
        mat_path,
        sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                       restitution=0.0))
    bind_physics_material(child, mat_path)


def _spawn_ward(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one ward shelf: KINEMATIC annular collar of 12 tangential boxes covering
    (360 - gap_deg) deg; the gap is centred on local +x. Local origin on the post axis
    at the collar's mid-thickness."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    n = 12
    span = math.radians(360.0 - c.gap_deg) / n
    r_mid = (c.inner_r + c.outer_r) / 2
    length = 2.0 * c.outer_r * math.sin(span / 2)  # flush coverage at the outer radius
    a0 = math.radians(c.gap_deg / 2)
    for i in range(n):
        ang = a0 + span * (i + 0.5)
        _add_box(stage, f"{prim_path}/seg_{i}",
                 center=(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0),
                 size=(c.outer_r - c.inner_r, length, c.thick),
                 color=c.color, collide=collide, yaw=ang)
        _bind_mat(prim_path, f"{prim_path}/seg_{i}", c.friction, max(c.friction - 0.02, 0.01))
    return root


def _spawn_sleeve(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sleeve: DYNAMIC compound — octagonal hub (8 tangential wall boxes,
    a closed loop around the post) + one radial fin along local +x. Local origin on
    the hub axis at mid-height. Mass + CoM authored explicitly (compound roots keep
    CoM at the origin otherwise on this stack)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(float(c.com_x), 0.0, 0.0))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.15)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateEnableCCDAttr(True)
    collide = _make_collide(c.contact_offset)

    # hub: 8 tangential wall boxes, inner inradius hub_in, wall hub_wall, height hub_h
    r_wall = c.hub_in + c.hub_wall / 2
    length = 2.0 * (c.hub_in + c.hub_wall) * math.tan(math.pi / 8)  # overlap at corners
    for i in range(8):
        ang = 2 * math.pi * i / 8
        _add_box(stage, f"{prim_path}/hub_{i}",
                 center=(r_wall * math.cos(ang), r_wall * math.sin(ang), 0.0),
                 size=(c.hub_wall, length, c.hub_h),
                 color=c.hub_color, collide=collide, yaw=ang)
        _bind_mat(prim_path, f"{prim_path}/hub_{i}", c.hub_friction, c.hub_friction - 0.02)
    # fin: radial handle along +x, overlapping the hub wall so it is one solid piece
    _add_box(stage, f"{prim_path}/fin",
             center=((c.fin_r0 + c.fin_r1) / 2, 0.0, 0.0),
             size=(c.fin_r1 - c.fin_r0, c.fin_w, c.fin_h),
             color=c.fin_color, collide=collide)
    _bind_mat(prim_path, f"{prim_path}/fin", c.fin_friction, c.fin_friction - 0.02)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ward" not in _SPAWNER_CACHE:

        @configclass
        class WardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ward)
            inner_r: float = 0.032
            outer_r: float = 0.075
            thick: float = 0.016
            gap_deg: float = 60.0
            color: tuple = (0.75, 0.15, 0.12)
            friction: float = 0.12
            contact_offset: float = 0.004

        @configclass
        class SleeveSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sleeve)
            hub_in: float = 0.019
            hub_wall: float = 0.006
            hub_h: float = 0.060
            fin_r0: float = 0.020
            fin_r1: float = 0.105
            fin_w: float = 0.014
            fin_h: float = 0.016
            mass: float = 0.15
            com_x: float = 0.016
            hub_color: tuple = (0.15, 0.30, 0.80)
            fin_color: tuple = (0.95, 0.85, 0.10)
            hub_friction: float = 0.10
            fin_friction: float = 0.12
            contact_offset: float = 0.003

        _SPAWNER_CACHE["ward"] = WardSpawnerCfg
        _SPAWNER_CACHE["sleeve"] = SleeveSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WardedSpindleSceneCfg(BaseCfg):
    """Config for `WardedSpindleScene`. Geometry is derived once in `__post_init__` so
    the scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    threaded_tol: float = tunable(0.008)  # sleeve axis within this (xy) of the post axis
    seat_z_tol: float = tunable(0.006)  # sleeve centre within this of the seated height
    upright_max_deg: float = tunable(15.0)  # hub axis within this of world-up
    settle_lin: float = tunable(0.04)  # max |lin vel| (m/s) when judging
    settle_ang: float = tunable(0.6)  # max |ang vel| (rad/s) when judging
    p1_margin: float = tunable(0.040)  # pass plane 1 = ward1 top - this
    p2_margin: float = tunable(0.040)  # pass plane 2 = ward2 top - this

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    tower_jitter: float = tunable(0.030)  # +/- xy jitter of the whole tower per episode
    misalign_min_deg: float = tunable(55.0)  # min |initial fin - notch1| and |notch2 - notch1|
    misalign_max_deg: float = tunable(180.0)

    # --- info: structure (env-local coordinates; bench top at z0) ----------------------------
    bench_center: tuple = info((0.10, 0.0, 0.36))
    bench_size: tuple = info((0.85, 0.85, 0.08))  # top at z0 = 0.40
    tower_xy0: tuple = info((0.10, 0.0))
    plinth_size: tuple = info((0.16, 0.16, 0.024))  # seat top at z0 + 0.024
    post_r: float = info(0.016)
    post_h: float = info(0.340)  # post from seat top to z0 + 0.364
    ward_inner_r: float = info(0.032)
    ward_outer_r: float = info(0.075)
    ward_thick: float = info(0.016)
    gap_deg: float = info(60.0)  # notch angular width (effective fin pass window ~ +/-12 deg)
    w1_top: float = info(0.200)  # upper ward top face, above z0
    w2_top: float = info(0.120)  # lower ward top face, above z0
    hub_in: float = info(0.019)  # hub inner inradius (post clearance 3 mm at flats)
    hub_wall: float = info(0.006)
    hub_h: float = info(0.060)
    fin_r1: float = info(0.105)  # fin tip radius (30 mm graspable beyond the ward rim)
    fin_w: float = info(0.014)
    fin_h: float = info(0.016)
    sleeve_mass: float = info(0.15)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    z0: float = field(default=None, init=False)  # bench top
    seat_top: float = field(default=None, init=False)  # plinth top face
    z_seat: float = field(default=None, init=False)  # sleeve centre z when seated
    z_rest_w1: float = field(default=None, init=False)  # sleeve centre z resting on ward1
    z_rest_w2: float = field(default=None, init=False)
    z_p1: float = field(default=None, init=False)  # pass plane 1 (below = past ward1)
    z_p2: float = field(default=None, init=False)
    post_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.z0 = self.bench_center[2] + self.bench_size[2] / 2
        self.seat_top = self.z0 + self.plinth_size[2]
        self.z_seat = self.seat_top + self.hub_h / 2
        self.z_rest_w1 = self.z0 + self.w1_top + self.fin_h / 2
        self.z_rest_w2 = self.z0 + self.w2_top + self.fin_h / 2
        self.z_p1 = self.z0 + self.w1_top - self.p1_margin
        self.z_p2 = self.z0 + self.w2_top - self.p2_margin
        self.post_top = self.seat_top + self.post_h


# ----- scene -----------------------------------------------------------------------------------
class WardedSpindleScene(BaseScene):
    cfg: WardedSpindleSceneCfg

    def __init__(self, cfg: WardedSpindleSceneCfg | None = None) -> None:
        super().__init__(cfg or WardedSpindleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic bench, the kinematic tower (green plinth + gray post +
        two red/orange ward collars) and the dynamic blue/yellow sleeve threaded on the
        post above the upper ward."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.33, 0.20))
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.50, 0.15))
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.64, 0.68))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        slick = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.10, dynamic_friction=0.08, restitution=0.0)
        grippy = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.45, dynamic_friction=0.40, restitution=0.0)

        tx, ty = c.tower_xy0
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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=c.bench_size, rigid_props=kin, collision_props=coll,
                    visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.bench_center),
            ),
            "plinth": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plinth",
                spawn=sim_utils.CuboidCfg(
                    size=c.plinth_size, rigid_props=kin, collision_props=coll,
                    physics_material=grippy, visual_material=green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx, ty, c.z0 + c.plinth_size[2] / 2)),
            ),
            "post": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post",
                spawn=sim_utils.CylinderCfg(
                    radius=c.post_r, height=c.post_h, axis="Z",
                    rigid_props=kin, collision_props=coll,
                    physics_material=slick, visual_material=steel),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx, ty, c.seat_top + c.post_h / 2)),
            ),
            "ward1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ward1",
                spawn=spawners["ward"](
                    inner_r=c.ward_inner_r, outer_r=c.ward_outer_r, thick=c.ward_thick,
                    gap_deg=c.gap_deg, color=(0.75, 0.15, 0.12)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx, ty, c.z0 + c.w1_top - c.ward_thick / 2)),
            ),
            "ward2": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ward2",
                spawn=spawners["ward"](
                    inner_r=c.ward_inner_r, outer_r=c.ward_outer_r, thick=c.ward_thick,
                    gap_deg=c.gap_deg, color=(0.90, 0.45, 0.10)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx, ty, c.z0 + c.w2_top - c.ward_thick / 2)),
            ),
            "sleeve": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sleeve",
                spawn=spawners["sleeve"](
                    hub_in=c.hub_in, hub_wall=c.hub_wall, hub_h=c.hub_h,
                    fin_r1=c.fin_r1, fin_w=c.fin_w, fin_h=c.fin_h,
                    mass=c.sleeve_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(tx, ty, c.z_rest_w1 + 0.006)),
            ),
        }
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.plinth: RigidObject = env.iscene["plinth"]
        self.post: RigidObject = env.iscene["post"]
        self.ward1: RigidObject = env.iscene["ward1"]
        self.ward2: RigidObject = env.iscene["ward2"]
        self.sleeve: RigidObject = env.iscene["sleeve"]
        self.env_origins = env.iscene.env_origins
        # Episode state (sampled at reset; readback-verified in smoke).
        self.th1 = torch.zeros(n, device=dev)  # upper notch angle (world yaw)
        self.th2 = torch.zeros(n, device=dev)  # lower notch angle
        self.th_s0 = torch.zeros(n, device=dev)  # sleeve initial fin angle
        # Rubric latches.
        self.p1_latch = torch.zeros(n, device=dev)
        self.p2_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        # External drive input (solve.py / smoke probes write; post_step consumes and OWNS
        # the sleeve's wrench slot — never call set_external_force_and_torque directly).
        self.drive_f = torch.zeros(n, 1, 3, device=dev)  # world force on the sleeve
        self.drive_t = torch.zeros(n, 1, 3, device=dev)  # world torque on the sleeve

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample tower xy + both notch angles + the sleeve's initial fin
        angle (guaranteed misaligned), write the kinematic tower and the sleeve resting
        just above the upper ward; clear latches and drives."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # NOTE: the FIRST randint draw after manual_seed is degenerate on this stack —
        # all discrete draws come from torch.rand comparisons instead.
        txy = torch.tensor(c.tower_xy0, device=dev) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.tower_jitter
        th1 = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        lo = math.radians(c.misalign_min_deg)
        hi = math.radians(c.misalign_max_deg)
        sgn2 = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        d2 = lo + (hi - lo) * torch.rand(m, device=dev)
        th2 = wrap_pi(th1 + sgn2 * d2)
        sgns = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        ds = lo + (hi - lo) * torch.rand(m, device=dev)
        th_s = wrap_pi(th1 + sgns * ds)
        self.th1[env_ids] = th1
        self.th2[env_ids] = th2
        self.th_s0[env_ids] = th_s
        self.p1_latch[env_ids] = 0.0
        self.p2_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0
        self.drive_f[env_ids] = 0.0
        self.drive_t[env_ids] = 0.0

        def write(body, z: float, yaw: torch.Tensor | None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = txy
            st[:, 2] = z
            if yaw is None:
                st[:, 3] = 1.0
            else:
                st[:, 3] = torch.cos(yaw / 2)
                st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.plinth, c.z0 + c.plinth_size[2] / 2, None)
        write(self.post, c.seat_top + c.post_h / 2, None)
        write(self.ward1, c.z0 + c.w1_top - c.ward_thick / 2, th1)
        write(self.ward2, c.z0 + c.w2_top - c.ward_thick / 2, th2)
        # sleeve: threaded on the post, fin 6 mm above the upper ward top (falls to rest)
        write(self.sleeve, c.z_rest_w1 + 0.006, th_s)

    # ----- readings -----------------------------------------------------------------------------
    def sleeve_z(self) -> torch.Tensor:
        """(N,) sleeve centre height above the env origin (env-local z)."""
        return self.sleeve.data.root_pos_w[:, 2] - self.env_origins[:, 2]

    def tab_yaw(self) -> torch.Tensor:
        """(N,) world yaw of the fin direction (sleeve local +x)."""
        q = self.sleeve.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # ex rotated: (1-2(y^2+z^2), 2(xy+wz), .)
        return torch.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

    def threaded(self) -> torch.Tensor:
        """(N,) bool: sleeve axis within `threaded_tol` (xy) of the post axis — true only
        while the closed hub is actually around the post."""
        d = self.sleeve.data.root_pos_w[:, 0:2] - self.post.data.root_pos_w[:, 0:2]
        return d.norm(dim=-1) < self.cfg.threaded_tol

    def upright(self) -> torch.Tensor:
        """(N,) bool: hub axis within `upright_max_deg` of world-up."""
        q = self.sleeve.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return up_z >= math.cos(math.radians(self.cfg.upright_max_deg))

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.sleeve.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.sleeve.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def seated(self) -> torch.Tensor:
        """(N,) bool: sleeve resting at the seat height, threaded, upright."""
        c = self.cfg
        return ((self.sleeve_z() - c.z_seat).abs() < c.seat_z_tol) \
            & self.threaded() & self.upright()

    def success(self) -> torch.Tensor:
        """(N,) bool: sleeve currently seated on the plinth (threaded through BOTH wards
        by topology), upright and settled."""
        return self.seated() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success(); else 0.3*p1 + 0.3*p2 + 0.2*seat
        (latched transient achievement — credit never evaporates under correct
        behavior). Null policy ~0."""
        partial = 0.3 * self.p1_latch + 0.3 * self.p2_latch + 0.2 * self.seat_latch
        return torch.where(self.success(), torch.ones_like(partial), partial)

    # ----- step-coupled mechanics (every step) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the solver's drive wrench on the sleeve (owns that slot), then latch
        rubric progress. Latches require the THREADED readback so off-post states earn
        nothing."""
        c = self.cfg
        self.sleeve.set_external_force_and_torque(self.drive_f, self.drive_t)
        z = self.sleeve_z()
        thr = self.threaded()
        p1 = ((z < c.z_p1) & thr).float()
        p2 = ((z < c.z_p2) & thr).float()
        st = (self.seated() & self.settled()).float()
        # A diverged step must not latch: torch.maximum propagates NaN. Garbage earns 0.
        p1 = torch.nan_to_num(p1, nan=0.0, posinf=0.0, neginf=0.0)
        p2 = torch.nan_to_num(p2, nan=0.0, posinf=0.0, neginf=0.0)
        st = torch.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
        self.p1_latch = torch.maximum(self.p1_latch, p1)
        self.p2_latch = torch.maximum(self.p2_latch, p2)
        self.seat_latch = torch.maximum(self.seat_latch, st)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("th1", "th2", "th_s0", "p1_latch", "p2_latch", "seat_latch",
                               "drive_f", "drive_t")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"plinth": self.plinth, "post": self.post, "ward1": self.ward1,
                "ward2": self.ward2, "sleeve": self.sleeve}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On a wooden bench stands a vertical steel post ({c.post_r * 200:.1f} cm thick, "
            f"about {c.post_h * 100:.0f} cm tall) rising from a flat GREEN BASE PLINTH. Two "
            f"horizontal ring-shaped shelves are fixed around the post: an upper RED shelf "
            f"and a lower ORANGE shelf (each about {c.ward_outer_r * 200:.0f} cm across, "
            f"{c.ward_thick * 100:.1f} cm thick). Each shelf has ONE open notch — a missing "
            f"{c.gap_deg:.0f}-degree sector, clearly visible as a gap in the ring — and the "
            f"two notches point in different, randomized directions every episode.\n"
            f"A BLUE COLLAR (a sleeve threaded around the post — it cannot come off "
            f"sideways) with a single YELLOW FIN sticking out is resting on top of the "
            f"upper red shelf: the fin lies on the shelf ring because it is not lined up "
            f"with the notch. The fin extends about "
            f"{(c.fin_r1 - c.ward_outer_r) * 100:.0f} cm beyond the shelf rim, so it can "
            f"be grasped or pushed at its tip.\n"
            f"Goal: lower the collar all the way down the post and SEAT it on the green "
            f"base plinth. The collar can only descend through a shelf when its yellow fin "
            f"is rotated (about the post axis) into that shelf's open notch; misaligned, "
            f"the fin simply rests on the shelf ring. So: rotate the collar until the fin "
            f"points into the RED shelf's notch and lower it through, then rotate it to the "
            f"ORANGE shelf's notch (a different direction) and lower it through, and let it "
            f"come to rest, still threaded on the post, sitting on the plinth. Alignment "
            f"tolerance is roughly +/-10 degrees. The collar must remain threaded on the "
            f"post the whole way — lifting it off the top of the post and dropping it near "
            f"the base does NOT count."
        )

    def instruction(self) -> str:
        return (
            "Rotate the blue collar on the post so its yellow fin lines up with the open "
            "notch of the upper red shelf and lower it through, then align the fin with "
            "the lower orange shelf's notch and lower it through, seating the collar on "
            "the green base plinth. The collar must stay threaded on the post — taking it "
            "off the top fails the task."
        )


# Guarded registration: the forge may import this module under two names.
if "warded_spindle" not in SCENES.list():
    SCENES.register("warded_spindle", WardedSpindleScene)
if "simgen.warded_spindle" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="warded_spindle", robot="null"))
