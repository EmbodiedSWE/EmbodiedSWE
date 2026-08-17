"""HutchFeedScene — open the hutch door, push the feed block in through the floor-level
doorway, close the door behind it.

Derived from embodiedgen/put_banana ("pick the banana off the cluttered table and drop it
into the mug" — a free aerial pick-and-place into an open-topped container, judged by a
bounding box). Here the container is a ROOFED hutch: its top is sealed, so the seed's
entire plan — carry the object over the container and drop it in — physically produces a
FAILING state (the object lands on the roof; smoke-checked). The only way in is a
floor-level doorway in one wall, and at reset that doorway is CLOSED by a removable door
slab standing in a C-channel. The solver must (1) OPEN: grasp the door slab by its knob
and lift it out of the channel, parking it clear of the doorway; (2) FEED: push the
yellow feed block along the ground through the doorway until it is fully inside the
hutch (a ground-level slide through an aperture — the block is never carried over
anything); (3) CLOSE: put the slab back into its channel so the loaded hutch ends shut.
The order open -> feed is enforced by physics (a closed door blocks the block); success
additionally requires the door re-seated, so the terminal state is only reachable by
open -> feed -> close.

Assets are fully procedural, authored by custom compound spawners (child colliders of
one body never self-collide):
  - hutch: KINEMATIC compound — 4 walls (front wall pierced by a 90 x 85 mm doorway with
    a lintel), a full roof, and the door channel outside the front wall: 2 front rail
    posts (block the slab falling outward) + 2 lateral stop posts (block it sliding
    sideways). Interior 220 x 220 mm, inner height 120 mm. Origin at the interior ground
    centre; local +x points OUT through the doorway. There is NO floor lip: the ground
    runs flat through the doorway.
  - door: DYNAMIC compound — a 130 x 8 x 110 mm blue slab + a grasp knob on its top
    edge. At reset it stands in the channel, covering the doorway, held by wall/rails/
    stops/ground; the channel is open at the top, so the door leaves by lifting.
  - block: a 45 mm yellow dynamic cube (the feed) on the ground outside, on the doorway
    side.

Per-episode randomization (readback-verifiable): hutch xy + yaw (the doorway heading
swings +/-28 deg, so the approach lane must be found, not memorized), block spawn
distance + lateral offset + yaw. The door pose follows the hutch (it spawns seated).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * opened            — the doorway was ever cleared of the door (latched bool)
  0.20 * approach          — block progress toward the doorway mouth, gated on opened
                             (latched running max; ~0 for doing nothing)
  0.25 * inside            — block ever fully inside the hutch interior (latched bool)
  0.25 * closed_after      — door seated back in its channel WHILE the block is inside
                             (latched bool; a door that never re-closes, or closes on an
                             empty hutch, earns nothing here)
  1.0 iff success()        — LIVE: block settled fully inside the interior at ground
                             height AND the door seated upright in its channel, both
                             still. Non-success capped at 0.85.

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
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, size, center, color, collide: Callable) -> None:
    """Author one axis-aligned box child collider (Cube prim + scale op)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _root(stage, prim_path: str, translation, orientation):
    """Define the root Xform and author its (single) translate/orient ops."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _collider(cfg) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _spawn_hutch(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hutch at `prim_path`: KINEMATIC root, origin at the interior ground
    centre, local +x OUT through the doorway. Children: back/side walls, front wall
    segments + lintel around the doorway, full roof, and the door channel (2 front
    rails + 2 lateral stops) outside the front wall. No floor — the ground runs flat
    through the doorway."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    collide = _collider(cfg)

    hx, hy, t, H = cfg.hx, cfg.hy, cfg.wall_t, cfg.wall_h
    dw, dh = cfg.ap_w, cfg.ap_h
    wall, roof_c, post_c = cfg.wall_color, cfg.roof_color, cfg.wall_color
    # back wall (local -x) and the two side walls
    _add_box(stage, f"{prim_path}/wall_back", (t, 2 * (hy + t), H),
             (-hx - t / 2, 0.0, H / 2), wall, collide)
    _add_box(stage, f"{prim_path}/wall_left", (2 * hx, t, H),
             (0.0, hy + t / 2, H / 2), wall, collide)
    _add_box(stage, f"{prim_path}/wall_right", (2 * hx, t, H),
             (0.0, -hy - t / 2, H / 2), wall, collide)
    # front wall (local +x): two full-height segments flanking the doorway + a lintel
    seg_w = (hy + t) - dw / 2
    seg_cy = dw / 2 + seg_w / 2
    _add_box(stage, f"{prim_path}/wall_front_l", (t, seg_w, H),
             (hx + t / 2, seg_cy, H / 2), wall, collide)
    _add_box(stage, f"{prim_path}/wall_front_r", (t, seg_w, H),
             (hx + t / 2, -seg_cy, H / 2), wall, collide)
    _add_box(stage, f"{prim_path}/lintel", (t, dw, H - dh),
             (hx + t / 2, 0.0, (H + dh) / 2), wall, collide)
    # roof: seals the top completely
    _add_box(stage, f"{prim_path}/roof", (2 * (hx + t), 2 * (hy + t), cfg.roof_t),
             (0.0, 0.0, H + cfg.roof_t / 2), roof_c, collide)
    # door channel, outside the front wall: front rails hold the slab against +x,
    # lateral stops hold it against +/-y; the wall itself holds -x; open at the top.
    rail_cx = hx + t + cfg.chan_w + cfg.post_t / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rail_{'l' if sgn > 0 else 'r'}",
                 (cfg.post_t, cfg.post_t, cfg.rail_h),
                 (rail_cx, sgn * cfg.rail_y, cfg.rail_h / 2), post_c, collide)
        _add_box(stage, f"{prim_path}/stop_{'l' if sgn > 0 else 'r'}",
                 (cfg.chan_w + cfg.post_t + 0.004, cfg.post_t, cfg.rail_h),
                 (hx + t + (cfg.chan_w + cfg.post_t) / 2, sgn * cfg.stop_y,
                  cfg.rail_h / 2), post_c, collide)
    return root


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the door at `prim_path`: DYNAMIC root with explicit MassAPI (custom
    spawners apply no cfg schemas — author everything here), a thin slab (height along
    +z, thickness along +x) + a grasp knob on the top edge. Sleep thresholds zeroed —
    the solve pose-holds it and a sleeping body would freeze mid-episode."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.10)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _collider(cfg)
    _add_box(stage, f"{prim_path}/slab", (cfg.slab_t, cfg.slab_w, cfg.slab_h),
             (0.0, 0.0, 0.0), cfg.color, collide)
    _add_box(stage, f"{prim_path}/knob", (cfg.knob[0], cfg.knob[1], cfg.knob[2]),
             (0.0, 0.0, cfg.slab_h / 2 + cfg.knob[2] / 2), cfg.color, collide)
    return root


def _hutch_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hutch" not in _SPAWNER_CACHE:

        @configclass
        class HutchSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hutch)
            hx: float = 0.11
            hy: float = 0.11
            wall_t: float = 0.012
            wall_h: float = 0.12
            roof_t: float = 0.010
            ap_w: float = 0.09
            ap_h: float = 0.085
            chan_w: float = 0.020
            post_t: float = 0.012
            rail_h: float = 0.06
            rail_y: float = 0.056
            stop_y: float = 0.073
            wall_color: tuple = (0.45, 0.30, 0.15)
            roof_color: tuple = (0.30, 0.16, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["hutch"] = HutchSpawnerCfg

    return _SPAWNER_CACHE["hutch"](
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        hx=c.hx, hy=c.hy, wall_t=c.wall_t, wall_h=c.wall_h, roof_t=c.roof_t,
        ap_w=c.ap_w, ap_h=c.ap_h, chan_w=c.chan_w, post_t=c.post_t, rail_h=c.rail_h,
        rail_y=c.rail_y, stop_y=c.stop_y, wall_color=c.wall_color,
        roof_color=c.roof_color, contact_offset=c.contact_offset,
    )


def _door_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "door" not in _SPAWNER_CACHE:

        @configclass
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            slab_t: float = 0.008
            slab_w: float = 0.13
            slab_h: float = 0.11
            knob: tuple = (0.014, 0.036, 0.022)
            mass: float = 0.06
            color: tuple = (0.15, 0.35, 0.80)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["door"] = DoorSpawnerCfg

    return _SPAWNER_CACHE["door"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.door_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        slab_t=c.door_t, slab_w=c.door_w, slab_h=c.door_h, knob=c.door_knob,
        mass=c.door_mass, color=c.door_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HutchFeedSceneCfg(BaseCfg):
    """Config for `HutchFeedScene`. The roof denies the seed's aerial drop; the doorway
    (90 mm) vs block (45 mm) leaves 22.5 mm of lateral slack per side for the ground
    push; the channel (20 mm) vs slab (8 mm) leaves ~6 mm per side for the re-seat."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    inside_margin: float = tunable(0.035)  # block centre this far inside the inner walls (m)
    block_z_tol: float = tunable(0.015)  # block centre at ground rest height within this (m)
    door_xy_tol: float = tunable(0.014)  # door centre within this of the channel seat, x (m)
    door_y_tol: float = tunable(0.025)  # door centre within this of the doorway axis, y (m)
    door_z_tol: float = tunable(0.012)  # door centre at seat height within this (m)
    door_tilt_max_deg: float = tunable(15.0)  # door up-axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    approach_d0: float = tunable(0.50)  # approach ramp: p = 1 - d/approach_d0

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    hutch_jitter: float = tunable(0.03)  # hutch xy jitter (+/- m)
    hutch_yaw_deg: float = tunable(28.0)  # doorway heading swing (+/- deg about facing base)
    block_dist: tuple = tunable((0.30, 0.38))  # block spawn distance out of the doorway (m)
    block_lat: float = tunable(0.07)  # block spawn lateral offset (+/- m)
    block_yaw_deg: float = tunable(180.0)  # block spawn yaw (+/- deg)

    # --- info: layout (single Franka base at the origin) ----------------------------------------
    hutch_pos: tuple = info((0.55, 0.0))  # hutch centre; doorway faces the origin

    # --- info: structure -------------------------------------------------------------------------
    hx: float = info(0.11)  # interior half-extent, x (doorway axis)
    hy: float = info(0.11)  # interior half-extent, y
    wall_t: float = info(0.012)
    wall_h: float = info(0.12)  # inner height (roof underside)
    roof_t: float = info(0.010)
    ap_w: float = info(0.09)  # doorway aperture width
    ap_h: float = info(0.085)  # doorway aperture height
    chan_w: float = info(0.020)  # door channel gap (x, outside the front wall)
    post_t: float = info(0.012)
    rail_h: float = info(0.06)  # channel post height — the lift-out depth
    rail_y: float = info(0.056)  # front rail centres at +/- this
    stop_y: float = info(0.073)  # lateral stop centres at +/- this
    door_t: float = info(0.008)
    door_w: float = info(0.13)
    door_h: float = info(0.11)
    door_knob: tuple = info((0.014, 0.036, 0.022))  # grasp knob on the slab top edge
    door_mass: float = info(0.06)
    door_color: tuple = info((0.15, 0.35, 0.80))  # blue
    block_s: float = info(0.045)  # feed block edge length
    block_mass: float = info(0.06)
    block_color: tuple = info((0.90, 0.80, 0.10))  # yellow
    wall_color: tuple = info((0.45, 0.30, 0.15))  # wood brown
    roof_color: tuple = info((0.30, 0.16, 0.10))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.20 + 0.25 + 0.25 = 0.85 = the non-success cap)
    w_open: float = info(0.15)
    w_app: float = info(0.20)
    w_in: float = info(0.25)
    w_close: float = info(0.25)

    # Derived (filled in __post_init__).
    door_seat_lx: float = field(default=None, init=False)  # door centre x, hutch frame, seated
    door_seat_lz: float = field(default=None, init=False)  # door centre z, seated
    mouth_lx: float = field(default=None, init=False)  # doorway mouth point, hutch frame x
    block_rest_z: float = field(default=None, init=False)  # block centre height at ground rest

    def __post_init__(self) -> None:
        self.door_seat_lx = round(self.hx + self.wall_t + self.chan_w / 2, 4)
        self.door_seat_lz = round(self.door_h / 2, 4)
        self.mouth_lx = round(self.hx + self.wall_t + self.chan_w + self.post_t + 0.06, 4)
        self.block_rest_z = round(self.block_s / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hutch_feed")
class HutchFeedScene(BaseScene):
    cfg: HutchFeedSceneCfg

    def __init__(self, cfg: HutchFeedSceneCfg | None = None) -> None:
        super().__init__(cfg or HutchFeedSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
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
            "hutch": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hutch",
                spawn=_hutch_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hutch_pos[0], c.hutch_pos[1], 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0)),  # yaw pi: doorway faces the origin
            ),
            "door": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=_door_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hutch_pos[0] - c.door_seat_lx, c.hutch_pos[1],
                         c.door_seat_lz + 0.002),
                    rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            "block": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block",
                spawn=sim_utils.CuboidCfg(
                    size=(c.block_s, c.block_s, c.block_s),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.10,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.block_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hutch_pos[0] - 0.34, c.hutch_pos[1], c.block_rest_z + 0.002)),
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
        self.hutch: RigidObject = env.iscene["hutch"]
        self.door: RigidObject = env.iscene["door"]
        self.block: RigidObject = env.iscene["block"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._opened = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._app_max = torch.zeros(n, device=env.device)
        self._inside = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._closed_after = torch.zeros(n, dtype=torch.bool, device=env.device)

    def _yaw_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        return q

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hutch placed with xy jitter + doorway-heading yaw, the door
        seated CLOSED in its channel (pose derived from the hutch pose), the block on
        the ground out on the doorway side; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- hutch: base yaw pi (doorway toward the origin) + swing ---
        yaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.hutch_yaw_deg)
        hxy = torch.tensor(c.hutch_pos, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.hutch_jitter
        q = self._yaw_quat(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = hxy
        st[:, 3:7] = q
        st[:, 0:3] += origin
        self.hutch.write_root_state_to_sim(st, env_ids)

        # doorway direction in world (hutch local +x)
        dvec = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

        # --- door: seated in the channel, aligned with the hutch ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = hxy + dvec * c.door_seat_lx
        st[:, 2] = c.door_seat_lz + 0.002
        st[:, 3:7] = q
        st[:, 0:3] += origin
        self.door.write_root_state_to_sim(st, env_ids)

        # --- block: out on the doorway side ---
        d = c.block_dist[0] + torch.rand(m, device=dev) * (c.block_dist[1] - c.block_dist[0])
        lat = (torch.rand(m, device=dev) * 2 - 1) * c.block_lat
        lvec = torch.stack([-dvec[:, 1], dvec[:, 0]], dim=-1)
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = hxy + dvec * d.unsqueeze(-1) + lvec * lat.unsqueeze(-1)
        st[:, 2] = c.block_rest_z + 0.002
        st[:, 3:7] = self._yaw_quat(byaw)
        st[:, 0:3] += origin
        self.block.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._opened[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._inside[env_ids] = False
        self._closed_after[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hutch": self.hutch.data.root_state_w[env_ids].clone(),
            "door": self.door.data.root_state_w[env_ids].clone(),
            "block": self.block.data.root_state_w[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "inside": self._inside[env_ids].clone(),
            "closed_after": self._closed_after[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hutch.write_root_state_to_sim(state["hutch"], env_ids)
        self.door.write_root_state_to_sim(state["door"], env_ids)
        self.block.write_root_state_to_sim(state["block"], env_ids)
        self._opened[env_ids] = state["opened"]
        self._app_max[env_ids] = state["app_max"]
        self._inside[env_ids] = state["inside"]
        self._closed_after[env_ids] = state["closed_after"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the ground stands a wooden HUTCH (brown box, interior "
            f"{2 * c.hx * 100:.0f} x {2 * c.hy * 100:.0f} cm, walls {c.wall_h * 100:.0f} cm "
            f"tall) whose top is completely sealed by a dark ROOF — nothing can be dropped "
            f"in from above. Its only opening is a doorway at ground level in one wall, "
            f"{c.ap_w * 100:.0f} cm wide and {c.ap_h * 100:.1f} cm tall; the ground runs "
            f"flat through it (no sill). At the start the doorway is CLOSED by a BLUE DOOR: "
            f"a thin {c.door_w * 100:.0f} cm-wide slab with a grasp knob on its top edge, "
            f"standing in a vertical channel in front of the doorway (wall behind it, two "
            f"short rail posts in front, stop posts at its sides). The channel is open at "
            f"the top: the door comes out by LIFTING it straight up about "
            f"{c.rail_h * 100:.0f} cm by the knob, and goes back in by lowering it into the "
            f"same channel. A YELLOW feed block (a {c.block_s * 100:.1f} cm cube) rests on "
            f"the ground outside, some {c.block_dist[0] * 100:.0f}-"
            f"{c.block_dist[1] * 100:.0f} cm out on the doorway side. The hutch position "
            f"and its doorway heading change between episodes — locate the doorway "
            f"visually.\n"
            f"Goal, in the only physically possible order: (1) lift the blue door out of "
            f"its channel and set it aside clear of the doorway; (2) move the yellow block "
            f"ALONG THE GROUND through the doorway — push it, it slides — until it is fully "
            f"inside the hutch (its centre at least {c.inside_margin * 100:.1f} cm past the "
            f"inner wall faces, resting on the ground); (3) put the blue door back into its "
            f"channel so it stands upright covering the doorway again. Success requires the "
            f"block settled inside AND the door re-seated (upright within "
            f"{c.door_tilt_max_deg:.0f} deg, centred on the doorway, at channel height), "
            f"everything at rest. A block left in the doorway, outside, or on the roof "
            f"fails; a hutch left open fails; closing the door on an empty hutch earns "
            f"nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the blue door out of its channel to open the hutch, push the yellow "
            "block along the ground through the doorway until it is fully inside, then "
            "seat the blue door back in its channel so the hutch ends closed with the "
            "block inside."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _hutch_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3): `pos_w` in the hutch body frame (origin at interior ground centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = pos_w - self.hutch.data.root_pos_w
        return quat_apply_inverse(self.hutch.data.root_quat_w, rel)

    def _mouth_w(self) -> torch.Tensor:
        """(N, 2): doorway mouth point (just outside the channel) in world xy."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        loc = torch.tensor([self.cfg.mouth_lx, 0.0, 0.0], device=self.env.device)
        p = self.hutch.data.root_pos_w + quat_apply(
            self.hutch.data.root_quat_w, loc.expand(n, 3))
        return p[:, :2]

    def door_up_z(self) -> torch.Tensor:
        """(N,) the door slab's up-axis world z component."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(self.door.data.root_quat_w, ez)[:, 2]

    def door_blocking(self) -> torch.Tensor:
        """(N,) bool: the door is in/over the doorway region (channel band, near the
        doorway axis, below lift-out height) — i.e. the doorway is NOT passable."""
        c = self.cfg
        loc = self._hutch_local(self.door.data.root_pos_w)
        return ((loc[:, 0] > c.hx - 0.02) & (loc[:, 0] < c.door_seat_lx + 0.03)
                & (loc[:, 1].abs() < 0.06) & (loc[:, 2] < c.door_h / 2 + 0.05))

    def door_closed_now(self) -> torch.Tensor:
        """(N,) bool: the door seated in its channel — centred on the doorway, at seat
        height, upright, still."""
        c = self.cfg
        loc = self._hutch_local(self.door.data.root_pos_w)
        seat_x = (loc[:, 0] - c.door_seat_lx).abs() < c.door_xy_tol
        seat_y = loc[:, 1].abs() < c.door_y_tol
        seat_z = (loc[:, 2] - c.door_seat_lz).abs() < c.door_z_tol
        upright = self.door_up_z() >= math.cos(math.radians(c.door_tilt_max_deg))
        still = self.door.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return seat_x & seat_y & seat_z & upright & still

    def block_inside_now(self) -> torch.Tensor:
        """(N,) bool: block centre fully inside the interior, resting at ground height."""
        c = self.cfg
        loc = self._hutch_local(self.block.data.root_pos_w)
        in_x = loc[:, 0].abs() < c.hx - c.inside_margin
        in_y = loc[:, 1].abs() < c.hy - c.inside_margin
        on_ground = (loc[:, 2] - c.block_rest_z).abs() < c.block_z_tol
        return in_x & in_y & on_ground

    def _update_latches(self) -> None:
        c = self.cfg
        clear = ~self.door_blocking()
        self._opened |= clear
        d = (self.block.data.root_pos_w[:, :2] - self._mouth_w()).norm(dim=-1)
        app = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * self._opened.float()
        self._app_max = torch.maximum(self._app_max, app)
        inside = self.block_inside_now()
        self._inside |= inside
        self._closed_after |= inside & self.door_closed_now()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool, LIVE physical outcome: block settled fully inside the hutch at
        ground height AND the door seated upright in its channel, everything still."""
        self._update_latches()
        block_still = self.block.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed
        return self.block_inside_now() & block_still & self.door_closed_now()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*opened + 0.20*approach (gated on opened) +
        0.25*inside + 0.25*closed-with-block-inside — all latched, ~0 for doing nothing
        (the door starts closed and every other term is gated behind acting), capped
        0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened.float() + c.w_app * self._app_max
                + c.w_in * self._inside.float()
                + c.w_close * self._closed_after.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hutch_feed", robot="null"))
