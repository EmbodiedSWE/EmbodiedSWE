"""SteleWalkScene — WALK the left black stele down the kerbed causeway into its socket
(sim_gen task `living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i419`).

Derived from libero_90/living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray,
but STRATEGICALLY different: the seed is a free-space grasp-carry-release of a gram-weight
bowl (two identical black bowls disambiguated only by POSITION — "the left one" — dropped
into a stationary open tray). Here the seed's outcome shape is kept — two IDENTICAL black
objects, only the LEFT one belongs in the receptacle — but the object is a stout black
basalt STELE (120 x 120 x 240 mm, 2.2 kg) and every continuous transport mode is removed:

  - it CANNOT be grasped or lifted: 120 mm across > an ~80 mm parallel jaw, and carrying
    it (all four base corners clear of the deck while over the causeway) is a declared,
    readback-measured FOUL that permanently forfeits the episode;
  - it CANNOT be slid: two raised KERBS cross the causeway; a stele pushed flat jams its
    leading base edge against the 20 mm kerb face (smoke proves the pinned state);
  - it CANNOT be tumbled end-over-end: any tilt past 30 deg (short of the 35.3 deg
    diagonal topple, past the 26.6 deg edge topple — i.e. "it is falling over") is the
    same permanent foul.

What remains is the ancient furniture-mover's answer: WALK it. Tip the stele a few
degrees onto one base edge, pivot the raised side forward around the planted corner, set
it down, alternate sides — a rock-and-swivel gait that steps over each kerb (a 15 deg tip
raises the swung edge 31 mm > the 20 mm kerb), crosses the causeway, steps over the low
socket threshold, and squares up inside the socket. The RIGHT stele is an identical decoy:
walking the wrong one in scores nothing (the seed's positional disambiguation, kept).

success() iff, settled and never fouled: all four base corners of the LEFT stele inside
the socket interior, upright (tilt < 8 deg), and the decoy NOT in the socket. score()
latches monotone progress (leave the berth 0.10, cross kerb 1 +0.20, cross kerb 2 +0.20,
reach the socket apron +0.15; cap 0.65), evaluated every physics substep and gated on
no-foul; a foul zeroes the score permanently. Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - causeway (KINEMATIC): deck slab (0.98 x 0.42 m, deck top = fixture local z 0), two
    full-width kerbs (20 mm tall, 20 mm wide) at local x 0.14 and 0.32, and a SOCKET at
    local x 0.58: a 170 x 170 mm interior walled 25 mm high on three sides with a low
    12 mm threshold on the approach side (the arrival is itself a step, not a slide);
  - two identical black steles (DYNAMIC 120 x 120 x 240 mm boxes, 2.2 kg), spawned
    side by side in the berth at local x 0, y +/-0.09.

Per-episode randomization (verified by readback in smoke): causeway xy +/- 40 mm and yaw
+/- 25 deg, WHICH body spawns on the left (fair swap), per-stele xy jitter +/- 8 mm and
yaw jitter +/- 4 deg. Heavy imports (isaaclab, pxr) are deferred so importing this module
— and registering the scene — stays app-free.
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


# ----- custom compound spawner -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_causeway(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC causeway at `prim_path`. Origin = DECK TOP under the berth
    centre; travel runs along local +x: berth, kerb 1, bay, kerb 2, apron, socket."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)

    co = cfg.contact_offset
    # deck slab: top at local z = 0
    _box(stage, f"{prim_path}/slab", (cfg.slab_x, cfg.slab_y, cfg.slab_t),
         (cfg.slab_cx, 0.0, -cfg.slab_t / 2), cfg.deck_color, co)
    # two full-width kerbs across the causeway
    for i, kx in enumerate(cfg.kerb_x):
        _box(stage, f"{prim_path}/kerb_{i}", (cfg.kerb_w, cfg.slab_y, cfg.kerb_h),
             (kx, 0.0, cfg.kerb_h / 2), cfg.kerb_color, co)
    # socket: three 25 mm walls + one low 12 mm threshold on the -x (approach) side
    ix, iy = cfg.sock_in, cfg.sock_in
    wt, wh, th = cfg.wall_t, cfg.wall_h, cfg.thresh_h
    sx = cfg.sock_x
    _box(stage, f"{prim_path}/sock_back", (wt, iy + 2 * wt, wh),
         (sx + ix / 2 + wt / 2, 0.0, wh / 2), cfg.sock_color, co)
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/sock_{tag}", (ix + 2 * wt, wt, wh),
             (sx, sgn * (iy / 2 + wt / 2), wh / 2), cfg.sock_color, co)
    _box(stage, f"{prim_path}/sock_thresh", (wt, iy + 2 * wt, th),
         (sx - ix / 2 - wt / 2, 0.0, th / 2), cfg.thresh_color, co)
    return root


def _causeway_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "causeway" not in _SPAWNER_CACHE:

        @configclass
        class CausewaySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_causeway)
            slab_x: float = 0.98
            slab_y: float = 0.42
            slab_t: float = 0.02
            slab_cx: float = 0.29
            kerb_x: tuple = (0.14, 0.32)
            kerb_w: float = 0.02
            kerb_h: float = 0.02
            sock_x: float = 0.58
            sock_in: float = 0.17
            wall_t: float = 0.012
            wall_h: float = 0.025
            thresh_h: float = 0.012
            deck_color: tuple = (0.55, 0.52, 0.46)
            kerb_color: tuple = (0.80, 0.55, 0.15)
            sock_color: tuple = (0.20, 0.45, 0.20)
            thresh_color: tuple = (0.30, 0.55, 0.30)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["causeway"] = CausewaySpawnerCfg

    return _SPAWNER_CACHE["causeway"](
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        slab_x=c.slab_x, slab_y=c.slab_y, slab_t=c.slab_t, slab_cx=c.slab_cx,
        kerb_x=c.kerb_x, kerb_w=c.kerb_w, kerb_h=c.kerb_h, sock_x=c.sock_x,
        sock_in=c.sock_in, wall_t=c.wall_t, wall_h=c.wall_h, thresh_h=c.thresh_h,
        deck_color=c.deck_color, kerb_color=c.kerb_color, sock_color=c.sock_color,
        thresh_color=c.thresh_color, contact_offset=c.contact_offset,
    )


def _stele_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=(c.stele_w, c.stele_w, c.stele_h),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.10,
            disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.stele_m),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.stele_color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SteleWalkSceneCfg(BaseCfg):
    """Config for `SteleWalkScene`. Honesty knobs asserted in `__post_init__`: the stele
    is wider than the jaw (no grasp), the gait tip clears kerb and threshold while staying
    well inside the edge-topple angle, the topple foul sits between the edge-topple and
    the diagonal-topple angles (walking survives, tumbling cannot), the airborne-foul
    threshold sits above every legitimate rest (deck, kerb top, threshold), a flat stele
    fits every bay and the socket with margin, and one Franka base pose reaches it all."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    upright_deg: float = tunable(8.0)     # max tilt for "standing" at the goal
    foul_tilt_deg: float = tunable(30.0)  # tilt past this = toppling = permanent foul
    air_h: float = tunable(0.032)         # min-base-corner clearance that means "carried"
    air_persist: int = tunable(12)        # substeps the carry must persist (0.1 s @ 120 Hz)
    air_band: tuple = tunable((0.05, 0.50))  # fixture-x band where the carry foul is armed
    dep_x: float = tunable(0.06)          # fixture-x that counts as "left the berth"
    apron_x: float = tunable(0.44)        # fixture-x that counts as "reached the apron"
    cross_margin: float = tunable(0.004)  # corners must clear a kerb face by this
    goal_margin: float = tunable(0.004)   # corners must sit inside the socket by this
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.40)     # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    cway_jitter: float = tunable(0.04)    # uniform +/- xy jitter of the causeway (m)
    cway_yaw_deg: float = tunable(25.0)   # uniform +/- causeway yaw
    stele_jxy: float = tunable(0.008)     # per-stele xy jitter in the berth (m)
    stele_yaw_deg: float = tunable(4.0)   # per-stele yaw jitter

    # --- tunable: placement ------------------------------------------------------------------
    cway_pos: tuple = tunable((-0.10, 0.0))  # causeway origin (deck top @ berth), nominal
    berth_dy: float = tunable(0.09)          # steles spawn at fixture y = +/- berth_dy

    # --- info: causeway (fixture local frame: origin = deck top at the berth) ----------------
    slab_x: float = info(0.98)
    slab_y: float = info(0.42)
    slab_t: float = info(0.02)
    slab_cx: float = info(0.29)           # slab spans local x in [-0.20, 0.78]
    kerb_x: tuple = info((0.14, 0.32))    # kerb centres along the causeway
    kerb_w: float = info(0.02)
    kerb_h: float = info(0.02)
    sock_x: float = info(0.58)            # socket interior centre (apron kerb2->threshold
                                          # is 153 mm: a 120 mm stele can yaw ~15 deg there)
    sock_in: float = info(0.17)           # socket interior width (square)
    wall_t: float = info(0.012)
    wall_h: float = info(0.025)
    thresh_h: float = info(0.012)         # low approach threshold (stepped over)
    # steles
    stele_w: float = info(0.12)           # square base — wider than an 80 mm jaw
    stele_h: float = info(0.24)
    stele_m: float = info(2.2)
    # embodiment / gait reference numbers (documented + asserted, not enforced at runtime)
    gait_tip_deg: float = info(15.0)      # nominal walking tip
    jaw_w: float = info(0.08)             # Franka parallel jaw span
    base_pose: tuple = info((0.24, -0.50))  # one plausible Franka base, fixture frame
    reach: float = info(0.76)
    # colors
    deck_color: tuple = info((0.55, 0.52, 0.46))
    kerb_color: tuple = info((0.80, 0.55, 0.15))
    sock_color: tuple = info((0.20, 0.45, 0.20))
    thresh_color: tuple = info((0.30, 0.55, 0.30))
    stele_color: tuple = info((0.08, 0.08, 0.09))  # black basalt (the seed's black bowls)
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    edge_topple_deg: float = field(default=None, init=False)
    diag_topple_deg: float = field(default=None, init=False)
    cos_foul: float = field(default=None, init=False)
    cos_upright: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        hx, hz = self.stele_w / 2, self.stele_h / 2
        self.edge_topple_deg = math.degrees(math.atan2(hx, hz))
        self.diag_topple_deg = math.degrees(math.atan2(hx * math.sqrt(2.0), hz))
        self.cos_foul = math.cos(math.radians(self.foul_tilt_deg))
        self.cos_upright = math.cos(math.radians(self.upright_deg))

        # ---- no-grasp honesty: the stele does not fit the jaw ----
        assert self.stele_w > self.jaw_w + 0.03, "stele must be far too wide for the jaw"

        # ---- topple-foul window: walking survives, tumbling cannot ----
        # A 90-deg roll over any base edge passes through tilt ~90 deg, so it MUST cross
        # the foul line; the gait tip must stay well below the edge-topple angle; the
        # foul line itself sits above the edge-topple (a stele that fouls really is
        # falling) and below the diagonal topple (nothing recoverable is fouled late).
        assert self.gait_tip_deg + 8.0 < self.edge_topple_deg, "gait tip too close to topple"
        assert self.foul_tilt_deg > self.edge_topple_deg + 3.0, "foul must mean 'falling'"
        assert self.foul_tilt_deg < self.diag_topple_deg - 4.0, "foul must precede flat"

        # ---- step clearance: a gait tip lifts the swung edge over kerb + threshold ----
        lift = self.stele_w * math.sin(math.radians(self.gait_tip_deg))
        assert lift > self.kerb_h + 0.006, "gait tip must clear the kerb"
        assert lift > self.thresh_h + 0.006, "gait tip must clear the socket threshold"

        # ---- slide jam: a flat stele's leading base edge catches the kerb face ----
        assert self.kerb_h >= 0.016, "kerb must be tall enough to jam a flat slide"
        assert self.kerb_h > 8 * self.contact_offset, "kerb must dwarf the contact skin"

        # ---- carried-foul threshold sits above every legitimate rest ----
        assert self.air_h > self.kerb_h + 0.008, "kerb-top rests must never read as carried"
        assert self.air_h > self.thresh_h + 0.008, "threshold rests must never read as carried"
        assert self.air_band[0] < self.dep_x < self.kerb_x[0] - self.kerb_w / 2 - self.stele_w / 2
        assert self.air_band[1] > self.sock_x - self.sock_in / 2 - self.wall_t

        # ---- flat stele fits every bay / the socket with walking room ----
        bay = (self.kerb_x[1] - self.kerb_x[0]) - self.kerb_w
        assert bay >= self.stele_w + 0.03, "bay must admit a flat stele with margin"
        assert self.sock_in >= self.stele_w + 0.03, "socket must admit the stele with margin"
        assert self.sock_in - 2 * self.goal_margin > self.stele_w + 0.02, "goal tolerance real"
        assert self.kerb_x[0] - self.kerb_w / 2 > self.stele_w / 2 + 0.06, "berth run-up room"
        # berth spacing: the two steles never touch, and the swing stays on the deck
        gap = 2 * self.berth_dy - self.stele_w
        assert gap >= 0.05, "the two steles must spawn clear of each other"
        diag = self.stele_w * math.sqrt(2.0) / 2
        assert self.berth_dy + diag < self.slab_y / 2 - 0.02, "swing must stay on the deck"

        # ---- one Franka base pose reaches every required contact ----
        bx, by = self.base_pose
        pts = [(-self.stele_w / 2, self.berth_dy + diag),           # berth far corner
               (self.sock_x + self.sock_in / 2 + self.wall_t,
                self.sock_in / 2 + self.wall_t),                    # socket back corner
               (self.kerb_x[1], -self.slab_y / 2)]                  # near kerb end
        for px, py in pts:
            assert math.hypot(px - bx, py - by) < self.reach, "base pose must reach the work"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stele_walk")
class SteleWalkScene(BaseScene):
    cfg: SteleWalkSceneCfg

    def __init__(self, cfg: SteleWalkSceneCfg | None = None) -> None:
        super().__init__(cfg or SteleWalkSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            "causeway": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Causeway",
                spawn=_causeway_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cway_pos[0], c.cway_pos[1], c.slab_t)),
            ),
            "stele_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SteleA",
                spawn=_stele_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cway_pos[0], c.cway_pos[1] + c.berth_dy,
                         c.slab_t + c.stele_h / 2 + 0.002)),
            ),
            "stele_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SteleB",
                spawn=_stele_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cway_pos[0], c.cway_pos[1] - c.berth_dy,
                         c.slab_t + c.stele_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cway: RigidObject = env.iscene["causeway"]
        self.stele_a: RigidObject = env.iscene["stele_a"]
        self.stele_b: RigidObject = env.iscene["stele_b"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.tgt_is_a = torch.ones(n, dtype=torch.bool, device=dev)  # which body is LEFT
        self.fouled = torch.zeros(n, dtype=torch.bool, device=dev)
        self.air_ctr = torch.zeros(n, dtype=torch.long, device=dev)
        self.dep_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.k1_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.k2_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.apron_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        # body-frame bottom corners (4, 3)
        c = self.cfg
        a, hz = c.stele_w / 2, c.stele_h / 2
        self.corners_b = torch.tensor(
            [[sx * a, sy * a, -hz] for sx in (1.0, -1.0) for sy in (1.0, -1.0)], device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: causeway with xy jitter + yaw, a fair draw of WHICH stele body
        spawns on the LEFT (fixture +y), per-stele xy + yaw jitter in the berth; foul and
        progress latches zeroed."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        cxy = torch.tensor(c.cway_pos, device=dev).expand(m, 2).clone()
        cxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cway_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cway_yaw_deg)
        half = yaw / 2
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = cxy
        st[:, 2] = c.slab_t
        st[:, 3:7] = q_yaw
        st[:, 0:3] += origin
        self.cway.write_root_state_to_sim(st, env_ids)

        # fair draw: which body is the LEFT (fixture +y) stele
        swap = torch.rand(m, device=dev) < 0.5
        self.tgt_is_a[env_ids] = ~swap  # tgt_is_a True -> body A spawned left

        for body, is_left in ((self.stele_a, ~swap), (self.stele_b, swap)):
            side = torch.where(is_left, torch.tensor(1.0, device=dev),
                               torch.tensor(-1.0, device=dev))
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.stele_jxy
            loc[:, 1] = side * c.berth_dy + (torch.rand(m, device=dev) * 2 - 1) * c.stele_jxy
            loc[:, 2] = c.stele_h / 2 + 0.003
            syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stele_yaw_deg) / 2
            q_s = torch.zeros(m, 4, device=dev)
            q_s[:, 0] = torch.cos(syaw)
            q_s[:, 3] = torch.sin(syaw)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = cxy
            st[:, 2] = c.slab_t
            st[:, 0:3] += quat_apply(q_yaw, loc)
            st[:, 3:7] = quat_mul(q_yaw, q_s)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self.fouled[env_ids] = False
        self.air_ctr[env_ids] = 0
        self.dep_latch[env_ids] = False
        self.k1_latch[env_ids] = False
        self.k2_latch[env_ids] = False
        self.apron_latch[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "causeway": self.cway.data.root_state_w[env_ids].clone(),
            "stele_a": self.stele_a.data.root_state_w[env_ids].clone(),
            "stele_b": self.stele_b.data.root_state_w[env_ids].clone(),
            "tgt_is_a": self.tgt_is_a[env_ids].clone(),
            "fouled": self.fouled[env_ids].clone(),
            "air_ctr": self.air_ctr[env_ids].clone(),
            "dep": self.dep_latch[env_ids].clone(),
            "k1": self.k1_latch[env_ids].clone(),
            "k2": self.k2_latch[env_ids].clone(),
            "apron": self.apron_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cway.write_root_state_to_sim(state["causeway"], env_ids)
        self.stele_a.write_root_state_to_sim(state["stele_a"], env_ids)
        self.stele_b.write_root_state_to_sim(state["stele_b"], env_ids)
        self.tgt_is_a[env_ids] = state["tgt_is_a"]
        self.fouled[env_ids] = state["fouled"]
        self.air_ctr[env_ids] = state["air_ctr"]
        self.dep_latch[env_ids] = state["dep"]
        self.k1_latch[env_ids] = state["k1"]
        self.k2_latch[env_ids] = state["k2"]
        self.apron_latch[env_ids] = state["apron"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        lift = c.stele_w * math.sin(math.radians(c.gait_tip_deg))
        return (
            f"A stone CAUSEWAY runs across the floor. At its near end (the BERTH) stand "
            f"two IDENTICAL black basalt steles — square pillars "
            f"{c.stele_w * 1000:.0f} x {c.stele_w * 1000:.0f} x {c.stele_h * 1000:.0f} mm, "
            f"{c.stele_m:.1f} kg — side by side. Stand at the berth looking down the "
            f"causeway toward its far end: the LEFT-hand stele is the target; the "
            f"right-hand one is a decoy and must be left out of the goal. Down the "
            f"causeway, two orange KERBS ({c.kerb_h * 1000:.0f} mm tall) cross its full "
            f"width, and past them a green SOCKET is set into the deck: a walled square "
            f"court ({c.sock_in * 1000:.0f} mm interior) with a low "
            f"{c.thresh_h * 1000:.0f} mm threshold facing the berth.\n"
            f"Goal: get the LEFT stele standing upright inside the socket. No continuous "
            f"transport works: the stele is {c.stele_w * 1000:.0f} mm across — too wide "
            f"for a parallel gripper to grasp — and the episode is FORFEIT (score zero, "
            f"forever) if it is ever CARRIED (all four base corners more than "
            f"{c.air_h * 1000:.0f} mm clear of the deck over the causeway for more than "
            f"a tenth of a second) or ever TIPPED past {c.foul_tilt_deg:.0f} deg from "
            f"vertical (past that it is toppling — it balances on an edge only up to "
            f"{c.edge_topple_deg:.0f} deg). Sliding it flat jams: the base edge catches "
            f"the {c.kerb_h * 1000:.0f} mm kerb face. The stele must therefore be "
            f"WALKED, the way movers walk furniture: tip it a few degrees onto one base "
            f"edge (a {c.gait_tip_deg:.0f} deg tip raises the free edge "
            f"{lift * 1000:.0f} mm — enough to swing it over a kerb), pivot the raised "
            f"side forward about the planted corner, set it down, and alternate sides, "
            f"stepping it over both kerbs and over the socket threshold. Judged only "
            f"when settled: the left stele's whole base inside the socket, standing "
            f"within {c.upright_deg:.0f} deg of vertical, the decoy NOT in the socket, "
            f"and no forfeit ever incurred. Walking the wrong (right-hand) stele in "
            f"scores nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Walk the LEFT black stele down the causeway and stand it in the green "
            "socket: rock it onto one base edge, pivot the raised side forward, set it "
            "down, and alternate — stepping it over both kerbs and the socket threshold. "
            "Never tip it past 30 degrees, never lift it clear of the deck, and leave "
            "the right-hand stele behind."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fix_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> causeway fixture frame (origin = deck top @ berth)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.cway.data.root_quat_w, p_w - self.cway.data.root_pos_w)

    def target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3) pos + (N,4) quat of the LEFT (target) stele."""
        m3 = self.tgt_is_a.unsqueeze(-1)
        pos = torch.where(m3, self.stele_a.data.root_pos_w, self.stele_b.data.root_pos_w)
        quat = torch.where(m3, self.stele_a.data.root_quat_w, self.stele_b.data.root_quat_w)
        return pos, quat

    def decoy_pos(self) -> torch.Tensor:
        m3 = self.tgt_is_a.unsqueeze(-1)
        return torch.where(m3, self.stele_b.data.root_pos_w, self.stele_a.data.root_pos_w)

    def _corners_w(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """(N,4,3) world positions of the 4 base corners."""
        from isaaclab.utils.math import quat_apply

        n = pos.shape[0]
        q = quat.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4)
        cb = self.corners_b.unsqueeze(0).expand(n, 4, 3).reshape(-1, 3)
        return (pos.unsqueeze(1) + quat_apply(q, cb).view(n, 4, 3))

    def target_corners_fix(self) -> torch.Tensor:
        """(N,4,3) target base corners in the fixture frame."""
        pos, quat = self.target()
        return self._fix_local_batch(self._corners_w(pos, quat))

    def _fix_local_batch(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,K,3) world -> fixture frame."""
        from isaaclab.utils.math import quat_apply_inverse

        n, k, _ = p_w.shape
        q = self.cway.data.root_quat_w.unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
        d = (p_w - self.cway.data.root_pos_w.unsqueeze(1)).reshape(-1, 3)
        return quat_apply_inverse(q, d).view(n, k, 3)

    def tilt_cos(self, quat: torch.Tensor) -> torch.Tensor:
        """(N,) cosine of the body's tilt from vertical."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(quat.shape[0], 3)
        return quat_apply(quat, ez)[:, 2]

    def target_fix(self) -> torch.Tensor:
        pos, _q = self.target()
        return self._fix_local(pos)

    def target_tilt_cos(self) -> torch.Tensor:
        _p, quat = self.target()
        return self.tilt_cos(quat)

    def target_in_socket(self) -> torch.Tensor:
        """(N,) bool: all four base corners inside the socket interior, standing."""
        c = self.cfg
        cf = self.target_corners_fix()
        m = c.sock_in / 2 - c.goal_margin
        inside = ((cf[:, :, 0] - c.sock_x).abs() < m) & (cf[:, :, 1].abs() < m) \
            & (cf[:, :, 2] < 0.06)
        return inside.all(dim=1) & (self.target_tilt_cos() > c.cos_upright)

    def decoy_in_socket(self) -> torch.Tensor:
        """(N,) bool: the decoy's centre inside the socket court (blocks success)."""
        c = self.cfg
        df = self._fix_local(self.decoy_pos())
        return ((df[:, 0] - c.sock_x).abs() < c.sock_in / 2 + c.wall_t) \
            & (df[:, 1].abs() < c.sock_in / 2 + c.wall_t) & (df[:, 2] < c.stele_h)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok_a = (self.stele_a.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.stele_a.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        ok_b = (self.stele_b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.stele_b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok_a & ok_b

    def min_corner_clear(self) -> torch.Tensor:
        """(N,) lowest base-corner height above the deck top (fixture z)."""
        return self.target_corners_fix()[:, :, 2].min(dim=1).values

    # ----- fouls + progress latches (step-coupled) --------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: update the CARRIED and TOPPLED fouls (permanent), then
        latch monotone progress gated on not-currently-fouled."""
        c = self.cfg
        tf = self.target_fix()
        cf = self.target_corners_fix()
        tilt = self.target_tilt_cos()

        # topple foul: tilted past the point of no return
        self.fouled |= tilt < c.cos_foul

        # carried foul: every base corner clear of the deck, sustained, over the causeway
        clear = cf[:, :, 2].min(dim=1).values > c.air_h
        in_band = (tf[:, 0] > c.air_band[0]) & (tf[:, 0] < c.air_band[1])
        airborne = clear & in_band
        self.air_ctr = torch.where(airborne, self.air_ctr + 1,
                                   torch.zeros_like(self.air_ctr))
        self.fouled |= self.air_ctr >= c.air_persist

        # progress latches (positions in the fixture frame, gated on no active foul)
        ok = ~self.fouled
        min_x = cf[:, :, 0].min(dim=1).values
        upright_ish = tilt > math.cos(math.radians(20.0))
        self.dep_latch |= ok & (tf[:, 0] > c.dep_x)
        self.k1_latch |= ok & upright_ish \
            & (min_x > c.kerb_x[0] + c.kerb_w / 2 + c.cross_margin)
        self.k2_latch |= ok & upright_ish \
            & (min_x > c.kerb_x[1] + c.kerb_w / 2 + c.cross_margin)
        self.apron_latch |= ok & upright_ish & (tf[:, 0] > c.apron_x) \
            & (tf[:, 1].abs() < 0.12)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: never fouled, LEFT stele standing in the socket, decoy out,
        everything settled."""
        return (~self.fouled) & self.target_in_socket() & (~self.decoy_in_socket()) \
            & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched walk progress 0.10 (left the berth) + 0.20
        (crossed kerb 1) + 0.20 (crossed kerb 2) + 0.15 (reached the apron), cap 0.65;
        exactly 1.0 iff success(); a foul zeroes everything permanently."""
        base = (0.10 * self.dep_latch.float() + 0.20 * self.k1_latch.float()
                + 0.20 * self.k2_latch.float() + 0.15 * self.apron_latch.float())
        base = torch.where(self.fouled, torch.zeros_like(base), base.clamp(0.0, 0.65))
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="stele_walk", robot="null"))
