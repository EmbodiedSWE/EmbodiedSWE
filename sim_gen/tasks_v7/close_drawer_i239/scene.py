"""DrawerRerailScene — put a fallen drawer back on its rails (sim_gen task
`close_drawer_i239`).

Derived from rlbench/close_drawer, but STRATEGICALLY different: the seed's drawer
rides an articulation and the whole skill is one guided push on its front until the
prismatic joint reads closed. Here THERE IS NO JOINT AND NO PUSHABLE DRAWER TO
BEGIN WITH — the drawer box has FALLEN OUT of the cabinet and lies on its SIDE on
the ground in front of it. The cabinet has TWO identical open bays (a lower and an
upper); a green BEACON lamp mounted over one bay's mouth marks the bay the drawer
belongs to (which bay varies per episode). The task is a recovery/installation:

  1. retrieve the fallen drawer from the ground (it starts on its side — a real
     6-DoF reorientation, not a yaw fix),
  2. present it upright (open top up, handle facing out) at the mouth of the
     MARKED bay,
  3. insert it through the mouth and slide it home until its front face sits
     flush at the cabinet front, resting on the bay floor.

Plan-level contrast with the seed: the seed is a single constrained translation of
an articulated part; here the judged object is a free rigid body that must be
re-captured by the fixture — pick it up, reorient it in free space, discriminate
between two identical apertures by an external marker, thread it through the
chosen aperture, and seat it against the back stop. Executing the seed's skill for
real (pushing the drawer along the ground toward the cabinet) merely jams it
against the plinth and scores ~0 (smoke proves this with a real force).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): cabinet (KINEMATIC: solid plinth, two side panels, back wall, mid
shelf, top slab — forming two open bays), drawer (DYNAMIC open-top box with a
handle bar standing off its front face), beacon (KINEMATIC green lamp cube,
re-mounted over the target bay's mouth at every reset).

Honesty by construction (asserted in `__post_init__`):
  - side clearance 8 mm and headroom 34 mm: the drawer really fits through the
    mouth, with room for closed-loop arm control;
  - the back wall IS the seated pose: bay depth minus drawer depth (10 mm) is
    inside the `front_tol` flush window, so seating against the hard stop
    succeeds;
  - a BACKWARDS drawer (handle in) is geometrically proud: handle standoff
    (40 mm) makes its front face stand 30 mm out — beyond `front_tol` — so the
    facing clause is backed by geometry;
  - an UPSIDE-DOWN drawer still fits the bay, so the upright clause of the rubric
    (open top up) is load-bearing, not vacuous;
  - the two bay bands are far apart and off the ground, so credit gated to the
    target bay can never leak from the decoy bay or from ground poses.

Per-episode randomization (readback-verified by smoke): cabinet yaw FREE
(+/-180 deg) + xy jitter, WHICH bay is marked (the beacon tracks it), drawer
spawn spot, WHICH side it lies on, and its yaw.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  hold  — drawer ever carried upright, clear of the ground, into the
                approach window of the MARKED bay's mouth (latched)
  0.55  ins   — latched max insertion fraction of the drawer into the MARKED bay
                (tracked only while upright, facing out, centred in that bay)
non-success total <= 0.70; exactly 1.0 iff success(): drawer resting on the
MARKED bay's floor, upright, handle out, front face flush (within `front_tol`),
centred, settled, finite. Null policy ~0 (the drawer lies where it fell). The
seed's strategy — push the drawer — earns ~0.

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


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0))
    return mat_path


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet at `prim_path`: KINEMATIC compound. Local frame: origin at
    the FRONT face centre on the ground (z=0); +x points OUT toward the apron; the
    two bays extend to -x. Solid plinth, side panels, back wall, mid shelf, top."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    Y = c.bay_w_in / 2 + c.panel_t                # outer half-width
    W2 = c.bay_w_in / 2                           # bay interior half-width
    x0, x1 = -(c.bay_d + c.back_t), 0.0           # depth extent
    z_b0, z_c0 = c.bay_z0, c.bay_z0 + c.bay_h_in          # lower bay floor / ceiling
    z_b1, z_c1 = z_c0 + c.shelf_t, z_c0 + c.shelf_t + c.bay_h_in  # upper bay
    z_top = z_c1 + c.top_t

    _span(stage, f"{prim_path}/plinth", x=(x0, x1), y=(-Y, Y), z=(0.0, z_b0),
          color=c.plinth_color, collide=collide)
    _span(stage, f"{prim_path}/side_p", x=(x0, x1), y=(W2, Y), z=(z_b0, z_top),
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(x0, x1), y=(-Y, -W2), z=(z_b0, z_top),
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(x0, -c.bay_d), y=(-W2, W2), z=(z_b0, z_top),
          color=c.back_color, collide=collide)
    _span(stage, f"{prim_path}/shelf", x=(x0, x1), y=(-W2, W2), z=(z_c0, z_b1),
          color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/top", x=(x0, x1), y=(-W2, W2), z=(z_c1, z_top),
          color=c.frame_color, collide=collide)

    smooth = _mk_material(prim_path, "smooth", c.cab_mu_s, c.cab_mu_d)
    bind_physics_material(prim_path, smooth)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at `prim_path`: DYNAMIC compound open-top box. Local frame:
    origin at the CENTRE of the main box; +x is the front (handle) direction; +z is
    the open top. The handle bar stands off the front face on a centre post."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L2, B2, H2 = c.d_l / 2, c.d_w / 2, c.d_h / 2
    t = c.box_t
    _span(stage, f"{prim_path}/floor", x=(-L2, L2), y=(-B2, B2), z=(-H2, -H2 + t),
          color=c.drawer_color, collide=collide)
    zw = (-H2 + t, H2)
    _span(stage, f"{prim_path}/w_front", x=(L2 - t, L2), y=(-B2, B2), z=zw,
          color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(-L2, -L2 + t), y=(-B2, B2), z=zw,
          color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-L2, L2), y=(B2 - t, B2), z=zw,
          color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-L2, L2), y=(-B2, -B2 + t), z=zw,
          color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/post", x=(L2, L2 + c.handle_post),
          y=(-0.006, 0.006), z=(-0.006, 0.006), color=c.handle_color, collide=collide)
    _span(stage, f"{prim_path}/bar", x=(L2 + c.handle_post, L2 + c.handle_post + c.handle_bar),
          y=(-c.handle_hw, c.handle_hw), z=(-0.006, 0.006),
          color=c.handle_color, collide=collide)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.drawer_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.30)
    prb.CreateAngularDampingAttr(1.00)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    mat = _mk_material(prim_path, "boxmat", c.drawer_mu_s, c.drawer_mu_d)
    bind_physics_material(prim_path, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            bay_w_in: float = 0.176
            panel_t: float = 0.012
            bay_d: float = 0.210
            back_t: float = 0.012
            bay_z0: float = 0.100
            bay_h_in: float = 0.104
            shelf_t: float = 0.030
            top_t: float = 0.030
            body_color: tuple = (0.33, 0.36, 0.43)
            frame_color: tuple = (0.42, 0.45, 0.52)
            back_color: tuple = (0.20, 0.22, 0.27)
            plinth_color: tuple = (0.24, 0.25, 0.28)
            contact_offset: float = 0.0015
            cab_mu_s: float = 0.25
            cab_mu_d: float = 0.20

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            d_l: float = 0.200
            d_w: float = 0.160
            d_h: float = 0.070
            box_t: float = 0.008
            handle_post: float = 0.028
            handle_bar: float = 0.012
            handle_hw: float = 0.030
            drawer_mass: float = 0.18
            drawer_color: tuple = (0.72, 0.68, 0.60)
            front_color: tuple = (0.80, 0.52, 0.15)
            handle_color: tuple = (0.88, 0.78, 0.12)
            contact_offset: float = 0.0015
            drawer_mu_s: float = 0.30
            drawer_mu_d: float = 0.25

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DrawerRerailSceneCfg(BaseCfg):
    """Config for `DrawerRerailScene`. The geometric honesty contract is asserted in
    `__post_init__`: the drawer fits its bay with real clearance, the back wall IS
    the flush pose, a backwards drawer is geometrically proud, an upside-down
    drawer still fits (so the upright clause is load-bearing), and the credit
    bands of the two bays and the ground never overlap."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    front_tol: float = tunable(0.012)     # front face within this of the cabinet front plane (m)
    y_tol: float = tunable(0.020)         # drawer centred in its bay (m)
    z_tol: float = tunable(0.012)         # drawer bottom within this of the bay floor (m)
    upright_max_deg: float = tunable(12.0)   # open top within this of the cabinet up
    facing_max_deg: float = tunable(15.0)    # handle direction within this of the cabinet out
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.60)     # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)       # cabinet yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)      # cabinet xy jitter (+/- m)
    randomize_bay: bool = tunable(True)   # sample WHICH bay is marked per episode
    spawn_x: tuple = tunable((0.32, 0.44))   # drawer spawn band on the apron (cabinet local x)
    spawn_y: float = tunable(0.14)        # drawer spawn half-range (cabinet local y)

    # --- info: cabinet (local frame: origin at the front face centre on the ground) --------------
    bay_w_in: float = info(0.176)
    panel_t: float = info(0.012)
    bay_d: float = info(0.210)
    back_t: float = info(0.012)
    bay_z0: float = info(0.100)           # lower bay floor top (= plinth top)
    bay_h_in: float = info(0.104)
    shelf_t: float = info(0.030)
    top_t: float = info(0.030)
    # --- info: drawer ----------------------------------------------------------------------------
    d_l: float = info(0.200)
    d_w: float = info(0.160)
    d_h: float = info(0.070)
    box_t: float = info(0.008)
    handle_post: float = info(0.028)
    handle_bar: float = info(0.012)
    handle_hw: float = info(0.030)
    drawer_mass: float = info(0.18)
    # --- info: beacon ----------------------------------------------------------------------------
    beacon_s: float = info(0.030)         # green lamp cube side
    # --- info: materials -------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    cab_mu_s: float = info(0.25)
    cab_mu_d: float = info(0.20)
    drawer_mu_s: float = info(0.30)
    drawer_mu_d: float = info(0.25)
    # --- info: score-latch gates (wider than the success tolerances, tighter than the decoy) -----
    hold_x: tuple = info((0.02, 0.32))    # approach window: drawer centre x (cabinet local)
    hold_y: float = info(0.12)
    hold_z: float = info(0.06)            # |centre z - bay rest z| for the hold latch
    hold_ground: float = info(0.05)       # drawer bottom must be above this (clear of the ground)
    ins_y: float = info(0.030)            # centred gate while insertion credit accrues
    ins_z: float = info(0.035)            # bay-band gate while insertion credit accrues
    ins_up_deg: float = info(15.0)
    ins_face_deg: float = info(20.0)
    # --- info: rubric weights (sum = 0.70 = the non-success cap) ---------------------------------
    w_hold: float = info(0.15)
    w_ins: float = info(0.55)

    # Derived (filled in __post_init__).
    handle_len: float = field(default=None, init=False)   # post + bar standoff
    bay_z: tuple = field(default=None, init=False)        # (lower, upper) bay floor tops
    bay_ceil: tuple = field(default=None, init=False)     # (lower, upper) bay ceilings
    seat_x: float = field(default=None, init=False)       # seated front-face x (vs the back wall)

    def __post_init__(self) -> None:
        self.handle_len = self.handle_post + self.handle_bar
        z_c0 = self.bay_z0 + self.bay_h_in
        z_b1 = z_c0 + self.shelf_t
        self.bay_z = (self.bay_z0, z_b1)
        self.bay_ceil = (z_c0, z_b1 + self.bay_h_in)
        self.seat_x = self.d_l - self.bay_d               # front face x when hard against the back
        # the drawer fits its bay with real clearance (arm-controllable, not a lottery)
        assert (self.bay_w_in - self.d_w) / 2 >= 0.006, "side clearance must be >= 6 mm"
        assert self.bay_h_in - self.d_h >= 0.030, "headroom must be >= 30 mm"
        # the back wall IS the flush pose: seating on the hard stop succeeds
        assert -self.front_tol < self.seat_x < 0.0, "hard stop must land inside the flush window"
        # a backwards drawer (handle in) is geometrically PROUD beyond the flush window
        assert self.handle_len + self.d_l - self.bay_d > self.front_tol + 0.010, \
            "backwards insertion must stand proud of the flush window"
        # an upside-down drawer still fits, so the upright clause is load-bearing
        assert self.d_h + 0.020 < self.bay_h_in, "flipped drawer must fit (rubric rejects it)"
        # credit bands: bays are separated and off the ground (no leak between them)
        assert self.bay_z[1] - self.bay_z[0] > 2 * self.ins_z + 0.02, "bay bands must not overlap"
        assert self.bay_z[0] - self.hold_z > 0.02, "lower-bay hold band must sit above the ground"
        assert self.hold_ground > self.d_h / 2 + 0.005, \
            "a drawer resting on the ground must sit below the hold gate"
        # the drawer spawns clear of the cabinet (handle included)
        assert self.spawn_x[0] - self.d_l / 2 - self.handle_len > 0.05, \
            "drawer spawn band must be clear of the cabinet front"
        assert abs(self.w_hold + self.w_ins - 0.70) < 1e-9


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
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drawer_rerail")
class DrawerRerailScene(BaseScene):
    cfg: DrawerRerailSceneCfg

    def __init__(self, cfg: DrawerRerailSceneCfg | None = None) -> None:
        super().__init__(cfg or DrawerRerailSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cab_spawn = cls["cabinet"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bay_w_in=c.bay_w_in, panel_t=c.panel_t, bay_d=c.bay_d, back_t=c.back_t,
            bay_z0=c.bay_z0, bay_h_in=c.bay_h_in, shelf_t=c.shelf_t, top_t=c.top_t,
            contact_offset=c.contact_offset, cab_mu_s=c.cab_mu_s, cab_mu_d=c.cab_mu_d)
        drw_spawn = cls["drawer"](
            d_l=c.d_l, d_w=c.d_w, d_h=c.d_h, box_t=c.box_t,
            handle_post=c.handle_post, handle_bar=c.handle_bar, handle_hw=c.handle_hw,
            drawer_mass=c.drawer_mass, contact_offset=c.contact_offset,
            drawer_mu_s=c.drawer_mu_s, drawer_mu_d=c.drawer_mu_d)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet", spawn=cab_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drw_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9, 0.0, 0.084))),
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CuboidCfg(
                    size=(c.beacon_s, c.beacon_s, c.beacon_s),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.85, 0.10), emissive_color=(0.02, 0.35, 0.04)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6))),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.target_bay = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._hold = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ins = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the cabinet (free yaw + xy jitter), sample WHICH bay
        is marked and mount the beacon over its mouth, drop the drawer on its SIDE
        on the apron (random spawn spot, random side, free yaw), clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_cab = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_cab
        self.cabinet.write_root_state_to_sim(st, env_ids)

        # which bay is marked (torch.rand comparison: first-randint-degenerate trap)
        if c.randomize_bay:
            k = (torch.rand(m, device=dev) < 0.5).long()
        else:
            k = torch.zeros(m, dtype=torch.long, device=dev)
        self.target_bay[env_ids] = k

        # beacon: mounted on the front face band just above the marked bay's mouth
        ceil_z = torch.tensor(c.bay_ceil, device=dev)[k]
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.beacon_s / 2 + 0.001
        loc[:, 2] = ceil_z + c.shelf_t / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_cab, loc)
        st[:, 3:7] = q_cab
        self.beacon.write_root_state_to_sim(st, env_ids)

        # drawer: on its SIDE on the apron — random spot, random side, free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.spawn_x[0] + torch.rand(m, device=dev) * (c.spawn_x[1] - c.spawn_x[0])
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.spawn_y
        loc[:, 2] = c.d_w / 2 + 0.004
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.full((m,), math.pi / 2, device=dev),
                           torch.full((m,), -math.pi / 2, device=dev))
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        q_drw = _qmul(_qmul(q_cab, _qz(dyaw)), _qx(side))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_cab, loc)
        st[:, 3:7] = q_drw
        self.drawer.write_root_state_to_sim(st, env_ids)

        self._hold[env_ids] = False
        self._ins[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "beacon": self.beacon.data.root_state_w[env_ids].clone(),
            "target_bay": self.target_bay[env_ids].clone(),
            "hold": self._hold[env_ids].clone(),
            "ins": self._ins[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.beacon.write_root_state_to_sim(state["beacon"], env_ids)
        self.target_bay[env_ids] = state["target_bay"]
        self._hold[env_ids] = state["hold"]
        self._ins[env_ids] = state["ins"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-blue two-bay CABINET stands on the ground: two identical open "
            f"bays (mouths {c.bay_w_in * 1000:.0f} mm wide x {c.bay_h_in * 1000:.0f} mm "
            f"tall) stacked one above the other, both empty, both open at the front. "
            f"A small bright-GREEN BEACON lamp is mounted on the cabinet's front face "
            f"directly above the mouth of ONE bay — that marked bay is the drawer's "
            f"home (which bay is marked varies by episode; the other bay is a decoy "
            f"and does not count). The cabinet's DRAWER — a light open-top box "
            f"({c.d_l * 1000:.0f} x {c.d_w * 1000:.0f} x {c.d_h * 1000:.0f} mm, an "
            f"amber front wall and a yellow handle bar standing "
            f"{c.handle_len * 1000:.0f} mm off its front face) — has FALLEN OUT: it "
            f"lies tipped over on its SIDE on the ground in front of the cabinet, at "
            f"a random spot and heading.\n"
            f"Goal: put the drawer back. Pick it up off the ground, turn it upright "
            f"(open top facing up) with its handle facing out toward you, insert it "
            f"through the mouth of the bay marked by the green beacon, and slide it "
            f"all the way in until its front face sits flush with the cabinet front "
            f"(within about {c.front_tol * 1000:.0f} mm — the back wall is the stop; "
            f"push gently home) resting on that bay's floor, centred. Leave it "
            f"settled. A drawer seated in the unmarked bay, seated upside-down, "
            f"inserted handle-first, left short of flush, or left anywhere else "
            f"does not count. No other objects are involved; there is no required "
            f"order beyond the obvious dependency of the steps."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the fallen drawer box from the ground, turn it upright with "
            "its yellow handle facing out, and slide it fully into the cabinet bay "
            "marked by the green beacon until its front face is flush with the "
            "cabinet. Seating it in the unmarked bay, upside-down, or handle-first "
            "fails."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _cab_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.cabinet.data.root_quat_w,
                                  pos_w - self.cabinet.data.root_pos_w)

    def _axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(up_dot, face_dot): drawer open-top axis vs cabinet up, and drawer handle
        axis vs cabinet out (+x), both as cosines."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        dev = self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        dz = quat_apply(self.drawer.data.root_quat_w, ez)
        dx = quat_apply(self.drawer.data.root_quat_w, ex)
        cz = quat_apply(self.cabinet.data.root_quat_w, ez)
        cx = quat_apply(self.cabinet.data.root_quat_w, ex)
        return (dz * cz).sum(-1), (dx * cx).sum(-1)

    def front_x(self) -> torch.Tensor:
        """(N,) cabinet-local x of the drawer's FRONT FACE centre (the +x face of
        the box, handle excluded). 0 = the cabinet front plane; `seat_x` = hard
        against the back wall."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        off = torch.tensor([self.cfg.d_l / 2, 0.0, 0.0], device=self.env.device).expand(n, 3)
        face_w = self.drawer.data.root_pos_w + quat_apply(self.drawer.data.root_quat_w, off)
        return self._cab_local(face_w)[:, 0]

    def bay_rest_z(self) -> torch.Tensor:
        """(N,) cabinet-local z of the drawer CENTRE when resting on its marked
        bay's floor."""
        c = self.cfg
        floor = torch.tensor(c.bay_z, device=self.env.device)[self.target_bay]
        return floor + c.d_h / 2

    def seated(self) -> torch.Tensor:
        """(N,) bool, geometric: drawer resting on the MARKED bay's floor, upright
        (open top up), facing out (handle out), centred, front face flush."""
        c = self.cfg
        loc = self._cab_local(self.drawer.data.root_pos_w)
        up_dot, face_dot = self._axes()
        fx = self.front_x()
        flush = (fx < c.front_tol) & (fx > c.seat_x - 0.02)
        centred = loc[:, 1].abs() < c.y_tol
        bottom = loc[:, 2] - c.d_h / 2
        floor = torch.tensor(c.bay_z, device=self.env.device)[self.target_bay]
        on_floor = (bottom - floor).abs() < c.z_tol
        upright = up_dot >= math.cos(math.radians(c.upright_max_deg))
        facing = face_dot >= math.cos(math.radians(c.facing_max_deg))
        return flush & centred & on_floor & upright & facing

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer lin AND ang velocity below the gates."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        return torch.isfinite(self.drawer.data.root_state_w).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        loc = self._cab_local(self.drawer.data.root_pos_w)
        up_dot, face_dot = self._axes()
        rest_z = self.bay_rest_z()
        # hold: carried upright, clear of the ground, into the marked bay's approach window
        hold = (up_dot >= math.cos(math.radians(20.0))) \
            & (loc[:, 2] - c.d_h / 2 >= c.hold_ground) \
            & (loc[:, 0] > c.hold_x[0]) & (loc[:, 0] < c.hold_x[1]) \
            & (loc[:, 1].abs() < c.hold_y) \
            & ((loc[:, 2] - rest_z).abs() < c.hold_z)
        self._hold |= hold & fin
        # insertion fraction into the MARKED bay, gated to a genuinely-tracking pose
        gate = (loc[:, 1].abs() < c.ins_y) & ((loc[:, 2] - rest_z).abs() < c.ins_z) \
            & (up_dot >= math.cos(math.radians(c.ins_up_deg))) \
            & (face_dot >= math.cos(math.radians(c.ins_face_deg)))
        frac = ((c.d_l - self.front_x()) / (c.d_l - c.seat_x)).clamp(0.0, 1.0)
        self._ins = torch.where(gate & fin, torch.maximum(self._ins, frac), self._ins)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the drawer physically seated in the MARKED bay — resting on
        its floor, upright, handle out, flush at the front plane, centred — settled
        and finite. A live geometric judgement of the terminal state; latch-free."""
        self._update_latches()
        return self.seated() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 hold (latched: carried upright into the
        marked bay's approach window) + 0.55 * latched max insertion fraction into
        the marked bay; exactly 1.0 iff success() holds live. Doing nothing scores
        ~0 (the drawer lies where it fell); pushing it along the ground — the
        seed's whole strategy — jams it on the plinth and also scores ~0."""
        c = self.cfg
        self._update_latches()
        base = c.w_hold * self._hold.float() + c.w_ins * self._ins
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drawer_rerail", robot="null"))
