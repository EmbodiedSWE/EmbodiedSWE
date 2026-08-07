"""HatchShelfScene — close the cabinet's standing hatch lid to CREATE the top surface,
then set the BLACK bowl upright on it.

Derived from libero_90 kitchen_scene5 "put the black bowl on top of the cabinet" (a
Franka picks a black bowl off the table and places it on the cabinet's FIXED flat top),
but the destination surface is REMOVED from the initial scene: the cabinet's top face
is a hinged HATCH LID standing OPEN past vertical, resting against its hinge stop, and
under it the cabinet is an open-mouthed pit. The seed's whole plan — carry the bowl to
the cabinet top and set it down — is a trap here: a bowl carried "to the top of the
cabinet" falls straight through the open mouth into the cavity and counts for nothing.
The only winning plan is to first CONSTRUCT the destination: push the standing lid
forward past its balance point so gravity carries it down flat over the mouth (the lid
is bistable — gravity holds it open against the hinge stop or flat closed, nothing in
between), and only then place the BLACK bowl — not the identical-shaped WHITE decoy
bowl — upright and centered on the closed lid. The seed needs one pick-and-place onto
an existing shelf; this task needs a mechanism actuation that creates the shelf, an
ordering forced by physics, and object discrimination the seed never asks for.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - cabinet: KINEMATIC compound at a FIXED pose (it anchors the lid's hinge; jointed
    mechanisms are never teleported — the randomization lives in the bowls and the lid
    angle): floor plate + 4 walls forming an open-topped box, interior 26 x 26 cm,
    rim at 24 cm.
  - lid: DYNAMIC plate (30 x 30 x 0.8 cm, 0.25 kg), root origin ON the hinge line at
    the cabinet's front-top edge (robot side); bind-time revolute joint cabinet->lid
    about +Y, limits [-open_limit, 0] deg, joint-pair collision disabled. Closed
    (0 deg) it lies flat over the mouth resting on the upper joint stop; open
    (-open_limit, i.e. leaning ~32 deg past vertical toward the robot) gravity pins it
    against the lower stop. Sleep thresholds zeroed.
  - bowls: two identical octagonal cups (base of 4 crossed boxes + 8 wall boxes,
    ~11 cm across, 4.5 cm tall) — one BLACK (the target), one WHITE (the decoy).

Per-episode randomization (readback-verifiable): Bernoulli left/right slot swap of the
two bowls + per-bowl xy jitter + free yaw, and the lid's initial open angle
(uniform in [lid_init_lo, lid_init_hi] deg).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 * close-progress — running max of the lid's travel from its OWN initial open
                          angle toward 0 (~0 for doing nothing: at rest the lid falls
                          AWAY from closed, onto its open stop)
  0.25 * lid-closed     — lid ever settled closed (latched bool)
  0.15 * carry          — running max of the black bowl's progress from its spawn
                          distance toward the goal point above the closed lid's centre
  0.20 * placed         — black bowl ever upright, centered on the CLOSED lid (latched)
  1.0 iff success()     — lid closed and still, black bowl upright within the centered
                          tolerance on the lid's top, at rest. Non-success cap 0.85.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             rot_z_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_z_deg:
        xf.AddRotateZOp().Set(float(rot_z_deg))
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


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet: KINEMATIC open-topped box. Local origin at the centre of the
    footprint at ground level; the mouth (interior cross-section) is fully open upward
    — the top surface does not exist until the lid is closed over it."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    t = c.t
    w_out = c.in_w + 2 * t
    d_out = c.in_d + 2 * t
    _add_box(stage, f"{prim_path}/floor",
             center=(0.0, 0.0, c.floor_t / 2),
             size=(c.in_d, c.in_w, c.floor_t), color=c.color, collide=collide)
    for sgn, nm in ((-1.0, "wall_front"), (1.0, "wall_back")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(sgn * (c.in_d + t) / 2, 0.0, c.h / 2),
                 size=(t, w_out, c.h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (c.in_w + t) / 2, c.h / 2),
                 size=(d_out, t, c.h), color=c.color, collide=collide)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hatch lid: DYNAMIC plate whose root origin lies ON the hinge line;
    the plate extends +x from the origin (over the mouth when closed). Sleep and
    stabilization thresholds zeroed (it must respond the instant it is pushed)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)  # damp the slam when gravity closes it
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/plate",
             center=(cfg.lid_l / 2, 0.0, 0.0),
             size=(cfg.lid_l, cfg.lid_w, cfg.lid_t),
             color=cfg.color, collide=collide)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a bowl: DYNAMIC octagonal cup. Base = 4 crossed flat boxes (a near-disc),
    walls = 8 boxes on the octagon flats. Root origin at the centre of the base's
    BOTTOM face (root z == support height when the bowl stands upright)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    for k in range(4):
        _add_box(stage, f"{prim_path}/base_{k}",
                 center=(0.0, 0.0, c.base_t / 2),
                 size=(c.base_len, c.base_w, c.base_t), color=c.color,
                 collide=collide, rot_z_deg=45.0 * k)
    for k in range(8):
        phi = math.radians(45.0 * k)
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(c.wall_r * math.cos(phi), c.wall_r * math.sin(phi),
                         c.base_t + c.wall_h / 2),
                 size=(c.wall_t, c.wall_w, c.wall_h), color=c.color,
                 collide=collide, rot_z_deg=45.0 * k)
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
            in_w: float = 0.26
            in_d: float = 0.26
            h: float = 0.24
            t: float = 0.012
            floor_t: float = 0.012
            color: tuple = (0.45, 0.30, 0.15)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            lid_l: float = 0.30
            lid_w: float = 0.30
            lid_t: float = 0.008
            color: tuple = (0.30, 0.42, 0.60)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            base_len: float = 0.10
            base_w: float = 0.0415
            base_t: float = 0.010
            wall_r: float = 0.0455
            wall_t: float = 0.012
            wall_w: float = 0.044
            wall_h: float = 0.035
            color: tuple = (0.04, 0.04, 0.04)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, lid=LidSpawnerCfg,
                              bowl=BowlSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HatchShelfSceneCfg(BaseCfg):
    """Config for `HatchShelfScene`. The interlock is architectural AND gravitational:
    the cabinet's top surface IS the lid, and the lid is bistable — gravity pins it
    fully open (leaning past vertical against its hinge stop) or lies it flat closed;
    a bowl brought to "the top of the cabinet" while the lid stands open falls through
    the 26 x 26 cm mouth into the cavity. Placement on top is only possible after the
    lid has been driven over-centre and has fallen flat."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    closed_tol_deg: float = tunable(8.0)  # lid counts as closed within this of flat 0 deg
    settle_speed: float = tunable(0.05)  # max bowl |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.50)  # max lid |ang vel| when judging (rad/s)
    on_xy_tol: float = tunable(0.09)  # bowl centre within this of the lid centre (each axis)
    up_min: float = tunable(0.85)  # min body-z . world-z for "upright"

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_jitter: float = tunable(0.03)  # per-bowl spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli black/white spawn-slot swap (demo sets False)
    bowl_yaw_deg: float = tunable(180.0)  # free yaw on both bowls (+/- deg)
    lid_init_lo: float = tunable(104.0)  # initial lid open angle, uniform in [lo, hi] deg
    lid_init_hi: float = tunable(118.0)

    # --- info: layout (single Franka base at the origin; lid face and goal within reach) --------
    cab_pos: tuple = info((0.56, 0.0))  # cabinet footprint centre (fixture is FIXED: it
    # anchors the lid's hinge, and jointed mechanisms are never teleported — the
    # randomization lives in the bowls and the lid's initial angle)
    slot_a: tuple = info((0.40, 0.22))  # bowl spawn slot A (on the floor, robot side)
    slot_b: tuple = info((0.40, -0.22))  # bowl spawn slot B

    # --- info: cabinet structure -----------------------------------------------------------------
    wall_t: float = info(0.012)
    in_w: float = info(0.26)  # interior width (y) — the open mouth; bowls are ~11 cm wide
    in_d: float = info(0.26)  # interior depth (x)
    cab_h: float = info(0.24)  # rim height (walls' top face)
    floor_t: float = info(0.012)
    cab_color: tuple = info((0.45, 0.30, 0.15))  # brown wooden cabinet

    # --- info: lid ---------------------------------------------------------------------------------
    lid_l: float = info(0.30)  # hinge-to-front-edge length (x when closed)
    lid_w: float = info(0.30)
    lid_t: float = info(0.008)
    lid_mass: float = info(0.25)
    open_limit_deg: float = info(122.0)  # hinge stop: fully open leans 32 deg past vertical
    lid_color: tuple = info((0.30, 0.42, 0.60))  # blue-gray lid

    # --- info: bowls -------------------------------------------------------------------------------
    bowl_r_out: float = info(0.055)  # octagon outer apothem ~5.2-6.1 cm (call it 11 cm wide)
    bowl_h: float = info(0.045)
    bowl_mass: float = info(0.15)
    black_color: tuple = info((0.04, 0.04, 0.04))
    white_color: tuple = info((0.92, 0.92, 0.88))

    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.25 + 0.15 + 0.20 = 0.70 <= the 0.85 non-success cap)
    w_prog: float = info(0.10)
    w_closed: float = info(0.25)
    w_carry: float = info(0.15)
    w_on: float = info(0.20)

    # Derived (filled in __post_init__).
    hinge_x: float = field(default=None, init=False)  # hinge line x (env-local)
    hinge_z: float = field(default=None, init=False)
    lid_center_x: float = field(default=None, init=False)  # closed lid centre x
    lid_top_z: float = field(default=None, init=False)  # closed lid top face height
    goal_pt: tuple = field(default=None, init=False)  # carry-progress target above the lid
    on_z_lo: float = field(default=None, init=False)  # bowl-root z band for "on the lid"
    on_z_hi: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        cx, cy = self.cab_pos
        self.hinge_x = cx - (self.in_d / 2 + self.wall_t)
        self.hinge_z = self.cab_h + self.lid_t / 2
        self.lid_center_x = self.hinge_x + self.lid_l / 2
        self.lid_top_z = self.cab_h + self.lid_t
        self.goal_pt = (self.lid_center_x, cy, self.lid_top_z + 0.030)
        # z band: tolerate ~1 deg of joint-limit sag below, and only ~2.5 cm above the
        # lid top — a bowl RESTING on the lid sits at lid_top + ~2 mm, while anything
        # hovering, carried, or stacked higher is NOT "on the lid" (a 3 cm hover must
        # not satisfy the gate: measured on the forge)
        self.on_z_lo = self.cab_h - 0.015
        self.on_z_hi = self.lid_top_z + 0.025


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hatch_shelf")
class HatchShelfScene(BaseScene):
    cfg: HatchShelfSceneCfg

    def __init__(self, cfg: HatchShelfSceneCfg | None = None) -> None:
        super().__init__(cfg or HatchShelfSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        cx, cy = c.cab_pos
        cabinet_spawn = spawners["cabinet"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_w=c.in_w, in_d=c.in_d, h=c.cab_h, t=c.wall_t, floor_t=c.floor_t,
            color=c.cab_color, contact_offset=c.contact_offset,
        )
        lid_spawn = spawners["lid"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            lid_l=c.lid_l, lid_w=c.lid_w, lid_t=c.lid_t,
            color=c.lid_color, contact_offset=c.contact_offset,
        )
        bowl_black_spawn = spawners["bowl"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            color=c.black_color, contact_offset=c.contact_offset,
        )
        bowl_white_spawn = spawners["bowl"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            color=c.white_color, contact_offset=c.contact_offset,
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=cabinet_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.hinge_x, cy, c.hinge_z)),
            ),
            "bowl_black": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BowlBlack",
                spawn=bowl_black_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], 0.003)),
            ),
            "bowl_white": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BowlWhite",
                spawn=bowl_white_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], 0.003)),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.lid: RigidObject = env.iscene["lid"]
        self.bowl_black: RigidObject = env.iscene["bowl_black"]
        self.bowl_white: RigidObject = env.iscene["bowl_white"]
        self.env_origins = env.iscene.env_origins
        self._author_hinge()
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._prog_max = torch.zeros(n, device=dev)  # lid travel toward closed, running max
        self._closed = torch.zeros(n, dtype=torch.bool, device=dev)  # lid ever settled closed
        self._carry_max = torch.zeros(n, device=dev)  # black bowl approach, running max
        self._on = torch.zeros(n, dtype=torch.bool, device=dev)  # ever placed on the closed lid
        self._lid0 = torch.full((n,), 110.0, device=dev)  # initial open angle (deg)
        self._d0 = torch.full((n,), 0.30, device=dev)  # black bowl spawn distance to goal

    def _author_hinge(self) -> None:
        """Per env: a +Y revolute joint cabinet->lid on the front-top edge, limits
        [-open_limit, 0] deg (0 = flat closed resting on the upper stop; -open_limit =
        leaning past vertical on the lower stop), joint-pair collision disabled."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        cx, cy = c.cab_pos
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/lid_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Lid"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x - cx), 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.open_limit_deg))
            j.CreateUpperLimitAttr(0.0)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: bowls randomly ASSIGNED to the two floor slots (+ xy jitter,
        free yaw), lid re-posed OPEN at a random angle past vertical (pure joint-
        coordinate re-pose of the follower about the unchanged hinge — the proven safe
        articulated re-pose; gravity then rests it on the open stop), fixture
        re-asserted, latches cleared, spawn distances recorded."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        cx, cy = c.cab_pos

        # --- fixture (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = cx, cy
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)

        # --- lid: open at a random angle past vertical (rotation about its own origin,
        # which IS the hinge line, so the hinge stays coincident) ---
        lid0 = c.lid_init_lo + torch.rand(m, device=dev) * (c.lid_init_hi - c.lid_init_lo)
        half = torch.deg2rad(-lid0) / 2  # joint angle is NEGATIVE when open
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = c.hinge_x, cy, c.hinge_z
        st[:, 3] = torch.cos(half)
        st[:, 5] = torch.sin(half)
        st[:, 0:3] += origin
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- bowls: Bernoulli slot swap + xy jitter + free yaw, standing on the floor ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        blk_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        wht_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.bowl_black, blk_xy), (self.bowl_white, wht_xy)):
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bowl_yaw_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = 0.003
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- latches + references (close/carry progress are measured from these) ---
        goal = torch.tensor(c.goal_pt, device=dev)
        p = torch.cat([blk_xy, torch.full((m, 1), 0.003, device=dev)], dim=1)
        self._d0[env_ids] = (p - goal).norm(dim=-1).clamp(min=0.05)
        self._lid0[env_ids] = lid0
        self._prog_max[env_ids] = 0.0
        self._closed[env_ids] = False
        self._carry_max[env_ids] = 0.0
        self._on[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "bowl_black": self.bowl_black.data.root_state_w[env_ids].clone(),
            "bowl_white": self.bowl_white.data.root_state_w[env_ids].clone(),
            "prog_max": self._prog_max[env_ids].clone(),
            "closed": self._closed[env_ids].clone(),
            "carry_max": self._carry_max[env_ids].clone(),
            "on": self._on[env_ids].clone(),
            "lid0": self._lid0[env_ids].clone(),
            "d0": self._d0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.bowl_black.write_root_state_to_sim(state["bowl_black"], env_ids)
        self.bowl_white.write_root_state_to_sim(state["bowl_white"], env_ids)
        self._prog_max[env_ids] = state["prog_max"]
        self._closed[env_ids] = state["closed"]
        self._carry_max[env_ids] = state["carry_max"]
        self._on[env_ids] = state["on"]
        self._lid0[env_ids] = state["lid0"]
        self._d0[env_ids] = state["d0"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        out_w = (c.in_w + 2 * c.wall_t) * 100
        return (
            f"A BROWN wooden cabinet ({out_w:.0f} x {out_w:.0f} cm footprint, rim "
            f"{c.cab_h * 100:.0f} cm high) stands on the floor. It has NO top surface at "
            f"the start: its top face is a BLUE-GRAY hatch lid ({c.lid_l * 100:.0f} x "
            f"{c.lid_w * 100:.0f} cm plate) hinged along the cabinet's top edge on the "
            f"side facing you, and the lid currently stands OPEN, leaning toward you "
            f"about {c.lid_init_lo:.0f}-{c.lid_init_hi:.0f} deg past flat (i.e. tilted "
            f"backward past vertical against its hinge stop, its free edge up at "
            f"~{(c.hinge_z + c.lid_l * 0.85) * 100:.0f} cm). Under it the cabinet is an "
            f"open-mouthed pit {c.in_w * 100:.0f} cm square and {c.cab_h * 100:.0f} cm "
            f"deep: anything released over the mouth falls INSIDE the cabinet, which "
            f"counts for nothing. The lid is bistable — push its raised face forward "
            f"(away from you) past vertical and gravity drops it flat, closing the "
            f"cabinet and CREATING the top surface; nothing holds it anywhere in "
            f"between. On the floor in front of the cabinet stand two identical "
            f"octagonal bowls (~{2 * c.bowl_r_out * 100:.0f} cm wide, "
            f"{c.bowl_h * 100:.1f} cm tall), one BLACK and one WHITE; which stands left "
            f"and which stands right changes per episode. The WHITE bowl is a decoy.\n"
            f"Goal: first close the hatch lid (push it over-centre so it falls flat "
            f"over the mouth, within {c.closed_tol_deg:.0f} deg of flat), then place "
            f"the BLACK bowl UPRIGHT on top of the closed lid, its centre within "
            f"{c.on_xy_tol * 100:.0f} cm of the lid's centre in both directions, and "
            f"leave everything at rest. The order is forced by physics: with the lid "
            f"open there is no top to place onto. A bowl dropped into the cabinet's "
            f"interior, the white bowl on the lid, a bowl upside-down or balanced at "
            f"the lid's edge, or a bowl anywhere else does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the cabinet's standing blue-gray hatch lid forward past vertical so "
            "it falls flat and closes the cabinet's open top, then set the BLACK bowl "
            "upright on the middle of the closed lid. Do not drop the bowl into the "
            "cabinet and do not use the white bowl."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def lid_open_deg(self) -> torch.Tensor:
        """(N,) lid opening in DEG (0 = flat closed, +122 = fully open). The lid only
        ever rotates about the hinge +y axis, so the root quat is (cos t/2, 0,
        sin t/2, 0) with t negative when open."""
        q = self.lid.data.root_quat_w
        return -torch.rad2deg(2.0 * torch.atan2(q[:, 2], q[:, 0]))

    def lid_closed(self) -> torch.Tensor:
        """(N,) bool: lid within `closed_tol_deg` of flat."""
        return self.lid_open_deg() <= self.cfg.closed_tol_deg

    def _upright(self, body: RigidObject) -> torch.Tensor:
        q = body.data.root_quat_w
        r33 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        return r33 >= self.cfg.up_min

    def _on_lid(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body upright on TOP of the CLOSED lid, centered within tolerance.
        The z band rejects the cavity (root ~0.02 there) and the xy band rejects the
        rim tops and the lid's overhanging edges."""
        c = self.cfg
        p = body.data.root_pos_w - self.env_origins
        ok_x = (p[:, 0] - c.lid_center_x).abs() <= c.on_xy_tol
        ok_y = (p[:, 1] - c.cab_pos[1]).abs() <= c.on_xy_tol
        ok_z = (p[:, 2] >= c.on_z_lo) & (p[:, 2] <= c.on_z_hi)
        return ok_x & ok_y & ok_z & self._upright(body) & self.lid_closed()

    def _bowl_still(self, body: RigidObject) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _lid_still(self) -> torch.Tensor:
        return self.lid.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_omega

    def _update_latches(self) -> None:
        c = self.cfg
        # lid travel toward closed, measured from THIS episode's initial open angle
        # (at rest the lid falls AWAY from closed, so the null policy latches ~0)
        prog = ((self._lid0 - self.lid_open_deg()) / self._lid0).clamp(0.0, 1.0)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self._prog_max = torch.maximum(self._prog_max, prog)
        self._closed |= self.lid_closed() & self._lid_still()
        goal = torch.tensor(c.goal_pt, device=self.env.device)
        d = (self.bowl_black.data.root_pos_w - self.env_origins - goal).norm(dim=-1)
        carry = (1.0 - d / self._d0).clamp(0.0, 1.0)
        carry = torch.nan_to_num(carry, nan=0.0, posinf=0.0, neginf=0.0)
        self._carry_max = torch.maximum(self._carry_max, carry)
        self._on |= self._on_lid(self.bowl_black)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No driven mechanics: gravity owns both of the lid's rest states. Latch."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: lid closed and still, BLACK bowl upright and centered on the
        lid's top face, at rest. Physical outcomes only (settled poses on a surface
        that exists only because the lid was actually closed)."""
        self._update_latches()
        return (self.lid_closed() & self._lid_still()
                & self._on_lid(self.bowl_black) & self._bowl_still(self.bowl_black))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 * close-progress + 0.25 * lid-closed + 0.15 *
        carry + 0.20 * placed-on-lid — all latched, ~0 for doing nothing, capped
        0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_prog * self._prog_max
                + c.w_closed * self._closed.float()
                + c.w_carry * self._carry_max
                + c.w_on * self._on.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hatch_shelf", robot="null"))
