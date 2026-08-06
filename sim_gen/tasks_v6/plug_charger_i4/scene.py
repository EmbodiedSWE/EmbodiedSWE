"""BayonetLockScene — drop the lugged plug into the socket well and TWIST it locked
(sim_gen task `plug_charger_i4`).

Derived from maniskill/plug_charger, but STRATEGICALLY different: the seed is a
precision pick-align-and-push insertion — grasp a charger, align its prongs with two
tight holes in a fixed base, and translate it straight in; the checker is pure bbox
containment of the charger in the base frame, and the whole difficulty is the
translational alignment tolerance. Here insertion is deliberately EASY (a 48 mm plug in
a 90 x 84 mm well, entered through a wide slot) and translation alone can NEVER succeed:
the plug carries two radial lugs, the well mouth is covered by two overhang plates that
leave only a slot for the lugs to pass, and success requires ROTATING the seated plug
about its axis (>= `lock_deg` away from the entry-slot direction) so the lugs ride
UNDER the plates — a bayonet lock, like a twist-lock connector. A solver needs a
different plan (align lugs to the slot -> seat -> twist to lock, with a wrist rotation
as the load-bearing final action) and different code structure (a yaw-latch predicate
and a regulated twist, not a translational insertion servo). The seed's own end state —
plug pushed straight in to full depth, never twisted — is expressible here and is
smoke control #6: it is NOT success and its credit is capped well below the lock stage.

Judged in the SOCKET'S body frame (its yaw is randomized, so the entry-slot direction
must be read from the scene). success() iff the plug is SEATED (inside the well
footprint, bottom at the well floor, upright) AND LOCKED (relative yaw at least
`lock_deg` away from the slot direction, mod 180 deg — the lug pair is symmetric) AND
settled. Geometry makes the lock real: with the lugs more than ~39 deg from the slot,
the plug physically cannot be lifted out (lugs hit the plate undersides); the 55 deg
success threshold sits 16 deg beyond that with margin, and the twist stroke can run to
90 deg, so the target band is wide. score() is graded and latched: 0.15 * best approach
+ 0.35 * best seating depth (gated inside the well, roughly upright) + 0.35 * best lock
rotation (gated seated), capped at 0.85; 0.9 once seated + locked; 1.0 iff success().
Doing nothing scores ~0 (approach is normalized by the episode's own spawn distance).

Assets are fully procedural, one rigid body each, authored by custom compound spawners
(the pen_holder pattern — child colliders of one body never self-collide):
  - socket: KINEMATIC block — floor slab, four walls forming a rectangular well
    (interior 90 x 84 mm, x-axis = slot axis), and two ORANGE overhang plates covering
    the mouth on both +-y sides, leaving an open slot strip (|y| <= 27 mm) along local
    x for the lugs (tip radius 38 mm) to pass. Plate underside 28 mm above the table;
    a seated plug's lug top is at 22 mm, so the lugs rotate freely under the plates.
  - plug: BLUE cylinder (r 24 mm, h 60 mm) + two radial lug boxes at the bottom
    (reaching r 38 mm) + an elongated cap handle on top (34 x 20 x 30 mm, long axis
    parallel to the lugs, so the lug direction is visible from above and the cap fits
    a parallel jaw). Sleep/stabilization thresholds are zeroed at spawn so applied
    wrenches always act.
  - distractor: a smooth RED cylinder of similar size, no lugs — it drops into the well
    but can never lock, and putting it there scores nothing (object identity matters).
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat the
3-7 mm clearances that make entry and rotation possible.

Per-episode randomization: socket xy jitter + yaw (moves the slot direction), plug
spawn xy jitter + free yaw, distractor xy jitter. Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


def _spawn_socket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC bayonet socket at `prim_path`. Origin = well-floor centre at
    TABLE level (z=0): floor slab, four walls, and two overhang plates on +-local-y that
    cover the mouth except an open slot strip (|y| <= open_half) along local x."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.0)

    xh, yh, wt, ft = cfg.well_x_half, cfg.well_y_half, cfg.wall_t, cfg.floor_t
    pu, pt, oh = cfg.plate_under, cfg.plate_t, cfg.open_half
    co, color, pcolor = cfg.contact_offset, cfg.color, cfg.plate_color
    wall_h = pu + pt - ft  # walls run from the floor top flush to the plate top

    # floor slab (full outer footprint)
    _box(stage, f"{prim_path}/floor", (2 * (xh + wt), 2 * (yh + wt), ft),
         (0.0, 0.0, ft / 2), color, co)
    # x walls (at +-x, full y extent)
    _box(stage, f"{prim_path}/wall_xn", (wt, 2 * (yh + wt), wall_h),
         (-(xh + wt / 2), 0.0, ft + wall_h / 2), color, co)
    _box(stage, f"{prim_path}/wall_xp", (wt, 2 * (yh + wt), wall_h),
         (xh + wt / 2, 0.0, ft + wall_h / 2), color, co)
    # y walls (between the x walls)
    _box(stage, f"{prim_path}/wall_yn", (2 * xh, wt, wall_h),
         (0.0, -(yh + wt / 2), ft + wall_h / 2), color, co)
    _box(stage, f"{prim_path}/wall_yp", (2 * xh, wt, wall_h),
         (0.0, yh + wt / 2, ft + wall_h / 2), color, co)
    # overhang plates: cover open_half..yh on both sides, full x extent, at the mouth
    pw = yh - oh
    _box(stage, f"{prim_path}/plate_yn", (2 * xh, pw, pt),
         (0.0, -(oh + pw / 2), pu + pt / 2), pcolor, co)
    _box(stage, f"{prim_path}/plate_yp", (2 * xh, pw, pt),
         (0.0, oh + pw / 2, pu + pt / 2), pcolor, co)
    return root


def _spawn_plug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bayonet plug at `prim_path`: one rigid body = cylinder (axis +z) + two
    radial lug boxes along local +-x near the bottom + an elongated cap handle on top
    (long axis parallel to the lugs — the visible yaw cue and the jaw grasp feature).
    Depenetration capped + light damping; sleep/stabilization thresholds zeroed (a
    sleeping body silently ignores applied external wrenches)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    r, h = cfg.plug_r, cfg.plug_h
    body = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    body.CreateRadiusAttr(r)
    body.CreateHeightAttr(h)
    body.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    body.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(body.GetPrim(), cfg.contact_offset)

    # lugs: radial boxes along +-x, spanning plug_r .. lug_tip_r, near the bottom
    lug_len = cfg.lug_tip_r - r
    lug_zc = -h / 2 + cfg.lug_z_lo + cfg.lug_h / 2
    for tag, sgn in (("lug_xp", 1.0), ("lug_xn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (lug_len, cfg.lug_w, cfg.lug_h),
             (sgn * (r + lug_len / 2), 0.0, lug_zc), cfg.lug_color, cfg.contact_offset)
    # cap handle on top, elongated along the lug axis (local x)
    _box(stage, f"{prim_path}/cap", (cfg.cap_l, cfg.cap_w, cfg.cap_h),
         (0.0, 0.0, h / 2 + cfg.cap_h / 2), cfg.cap_color, cfg.contact_offset)
    return root


def _socket_spawner_cfg(*, well_x_half: float, well_y_half: float, wall_t: float,
                        floor_t: float, plate_under: float, plate_t: float,
                        open_half: float, color: tuple, plate_color: tuple,
                        contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "socket" not in _SPAWNER_CACHE:

        @configclass
        class BayonetSocketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_socket)
            well_x_half: float = 0.045
            well_y_half: float = 0.042
            wall_t: float = 0.012
            floor_t: float = 0.008
            plate_under: float = 0.028
            plate_t: float = 0.008
            open_half: float = 0.027
            color: tuple = (0.30, 0.30, 0.33)
            plate_color: tuple = (0.90, 0.45, 0.10)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["socket"] = BayonetSocketSpawnerCfg

    return _SPAWNER_CACHE["socket"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        well_x_half=well_x_half, well_y_half=well_y_half, wall_t=wall_t, floor_t=floor_t,
        plate_under=plate_under, plate_t=plate_t, open_half=open_half, color=color,
        plate_color=plate_color, contact_offset=contact_offset,
    )


def _plug_spawner_cfg(*, plug_r: float, plug_h: float, lug_tip_r: float, lug_w: float,
                      lug_h: float, lug_z_lo: float, cap_l: float, cap_w: float,
                      cap_h: float, mass: float, color: tuple, lug_color: tuple,
                      cap_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "plug" not in _SPAWNER_CACHE:

        @configclass
        class BayonetPlugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_plug)
            plug_r: float = 0.024
            plug_h: float = 0.060
            lug_tip_r: float = 0.038
            lug_w: float = 0.008
            lug_h: float = 0.008
            lug_z_lo: float = 0.006
            cap_l: float = 0.034
            cap_w: float = 0.020
            cap_h: float = 0.030
            color: tuple = (0.15, 0.30, 0.85)
            lug_color: tuple = (0.10, 0.20, 0.60)
            cap_color: tuple = (0.20, 0.42, 0.95)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["plug"] = BayonetPlugSpawnerCfg

    return _SPAWNER_CACHE["plug"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        plug_r=plug_r, plug_h=plug_h, lug_tip_r=lug_tip_r, lug_w=lug_w, lug_h=lug_h,
        lug_z_lo=lug_z_lo, cap_l=cap_l, cap_w=cap_w, cap_h=cap_h, color=color,
        lug_color=lug_color, cap_color=cap_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BayonetLockSceneCfg(BaseCfg):
    """Config for `BayonetLockScene`. `lock_deg` is the strategic honesty knob: the
    success rotation must sit well beyond the geometric extraction limit (the yaw range
    in which the lugs still fit back out through the slot), so a locked plug is REALLY
    held under the plates (enforced in `__post_init__`)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    lock_deg: float = tunable(55.0)  # slot-distance (deg) at/above which the plug is locked
    seat_z_tol: float = tunable(0.006)  # plug-centre height slack over seated height (m)
    upright_deg: float = tunable(10.0)  # plug axis within this of world-up when judged
    gate_upright_deg: float = tunable(25.0)  # rough-upright gate for depth-progress credit
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(0.5)  # max |ang vel| when judging success (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    socket_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the socket at reset (m)
    socket_yaw_deg: float = tunable(20.0)  # uniform +/- socket yaw at reset (deg)
    plug_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the plug spawn (m)
    plug_yaw_deg: float = tunable(180.0)  # uniform +/- plug yaw at reset (free)
    distractor_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the red plug (m)

    # --- tunable: placement ------------------------------------------------------------------
    socket_pos: tuple = tunable((0.0, 0.10))  # well-floor centre, nominal
    plug_spawn: tuple = tunable((0.0, -0.16))  # plug spawn centre, in front of the socket
    distractor_pos: tuple = tunable((0.28, -0.02))  # red distractor plug, off to the side

    # --- info: structure ---------------------------------------------------------------------
    well_x_half: float = info(0.045)  # well interior half-extent along the slot axis
    well_y_half: float = info(0.042)  # well interior half-extent across the slot
    wall_t: float = info(0.012)
    floor_t: float = info(0.008)  # well floor sits this high above the table
    plate_under: float = info(0.028)  # overhang-plate underside above the table
    plate_t: float = info(0.008)
    open_half: float = info(0.027)  # slot half-width: mouth open where |y_local| <= this
    plug_r: float = info(0.024)
    plug_h: float = info(0.060)
    lug_tip_r: float = info(0.038)  # lug tips reach this radius (blocked by the plates)
    lug_w: float = info(0.008)
    lug_h: float = info(0.008)
    lug_z_lo: float = info(0.006)  # lug underside above the plug bottom
    cap_l: float = info(0.034)  # cap long axis (parallel to the lugs — the yaw cue)
    cap_w: float = info(0.020)  # cap grasp width for a parallel jaw
    cap_h: float = info(0.030)
    plug_mass: float = info(0.25)
    plug_color: tuple = info((0.15, 0.30, 0.85))
    lug_color: tuple = info((0.10, 0.20, 0.60))
    cap_color: tuple = info((0.20, 0.42, 0.95))
    socket_color: tuple = info((0.30, 0.30, 0.33))
    plate_color: tuple = info((0.90, 0.45, 0.10))
    distractor_r: float = info(0.022)  # fits the well and the slot — identity rejects it
    distractor_h: float = info(0.055)
    distractor_mass: float = info(0.20)
    distractor_color: tuple = info((0.85, 0.15, 0.12))
    # Explicit small offsets: the ~2 cm default would eat the 3-7 mm working clearances.
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    seated_z: float = field(default=None, init=False)  # plug-centre height when seated (m)
    entry_z: float = field(default=None, init=False)  # plug-centre height, bottom at plate top
    extract_deg: float = field(default=None, init=False)  # geometric un-lock yaw limit (deg)

    def __post_init__(self) -> None:
        self.seated_z = self.floor_t + self.plug_h / 2
        self.entry_z = self.plate_under + self.plate_t + self.plug_h / 2
        # Geometric extraction limit: largest slot-distance at which a lug corner still
        # clears the slot strip |y| <= open_half (solve lug_tip_r*sin(t) + lug_w/2*cos(t)
        # = open_half by fixed-point iteration).
        t = 0.0
        for _ in range(20):
            t = math.asin(max(0.0, (self.open_half - self.lug_w / 2 * math.cos(t))) / self.lug_tip_r)
        self.extract_deg = math.degrees(t)
        assert self.lock_deg >= self.extract_deg + 10.0, (
            f"lock_deg {self.lock_deg:.0f} must sit >= 10 deg beyond the geometric extraction "
            f"limit {self.extract_deg:.1f} deg, or 'locked' would not be physically held")
        assert self.lock_deg <= 80.0, "lock zone must stay reachable well before the 90 deg apex"
        assert self.open_half - self.plug_r >= 0.002, "plug body must clear the slot strip"
        assert self.well_y_half - self.lug_tip_r >= 0.003, "lugs must clear the y walls at 90 deg"
        assert self.well_x_half - self.lug_tip_r >= 0.005, "lugs must clear the x walls at entry"
        under_clear = self.plate_under - self.floor_t - self.lug_z_lo - self.lug_h
        assert under_clear >= 0.004, (
            f"seated lugs need >= 4 mm below the plate underside to rotate (got "
            f"{under_clear * 1000:.1f} mm)")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("plug_bayonet_lock")
class BayonetLockScene(BaseScene):
    cfg: BayonetLockSceneCfg

    def __init__(self, cfg: BayonetLockSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetLockSceneCfg())

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
            "socket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Socket",
                spawn=_socket_spawner_cfg(
                    well_x_half=c.well_x_half, well_y_half=c.well_y_half, wall_t=c.wall_t,
                    floor_t=c.floor_t, plate_under=c.plate_under, plate_t=c.plate_t,
                    open_half=c.open_half, color=c.socket_color, plate_color=c.plate_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.socket_pos[0], c.socket_pos[1], 0.0)),
            ),
            "plug": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plug",
                spawn=_plug_spawner_cfg(
                    plug_r=c.plug_r, plug_h=c.plug_h, lug_tip_r=c.lug_tip_r, lug_w=c.lug_w,
                    lug_h=c.lug_h, lug_z_lo=c.lug_z_lo, cap_l=c.cap_l, cap_w=c.cap_w,
                    cap_h=c.cap_h, mass=c.plug_mass, color=c.plug_color,
                    lug_color=c.lug_color, cap_color=c.cap_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.plug_spawn[0], c.plug_spawn[1], c.plug_h / 2 + 0.002)),
            ),
            "distractor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Distractor",
                spawn=sim_utils.CylinderCfg(
                    radius=c.distractor_r, height=c.distractor_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.distractor_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.distractor_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.distractor_pos[0], c.distractor_pos[1], c.distractor_h / 2 + 0.002)),
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
        self.socket: RigidObject = env.iscene["socket"]
        self.plug: RigidObject = env.iscene["plug"]
        self.distractor: RigidObject = env.iscene["distractor"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 1.0, device=dev)  # spawn->socket distance, per episode
        self.approach_latch = torch.zeros(n, device=dev)
        self.depth_latch = torch.zeros(n, device=dev)
        self.rot_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: socket with xy jitter + yaw (the slot direction moves), plug
        upright in front with xy jitter + free yaw, distractor off to the side; latches
        zeroed and the approach baseline `d0` captured from the sampled poses."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        sock_xy = torch.tensor(c.socket_pos, device=dev).expand(m, 2).clone()
        sock_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.socket_jitter
        sock_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.socket_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = sock_xy
        st[:, 3] = torch.cos(sock_yaw / 2)
        st[:, 6] = torch.sin(sock_yaw / 2)
        st[:, 0:3] += origin
        self.socket.write_root_state_to_sim(st, env_ids)

        plug_xy = torch.tensor(c.plug_spawn, device=dev).expand(m, 2).clone()
        plug_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plug_jitter
        plug_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.plug_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = plug_xy
        st[:, 2] = c.plug_h / 2 + 0.002
        st[:, 3] = torch.cos(plug_yaw / 2)
        st[:, 6] = torch.sin(plug_yaw / 2)
        st[:, 0:3] += origin
        self.plug.write_root_state_to_sim(st, env_ids)

        dis_xy = torch.tensor(c.distractor_pos, device=dev).expand(m, 2).clone()
        dis_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.distractor_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dis_xy
        st[:, 2] = c.distractor_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.distractor.write_root_state_to_sim(st, env_ids)

        self.d0[env_ids] = (plug_xy - sock_xy).norm(dim=-1).clamp(min=0.05)
        self.approach_latch[env_ids] = 0.0
        self.depth_latch[env_ids] = 0.0
        self.rot_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "socket": self.socket.data.root_state_w[env_ids].clone(),
            "plug": self.plug.data.root_state_w[env_ids].clone(),
            "distractor": self.distractor.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
            "depth_latch": self.depth_latch[env_ids].clone(),
            "rot_latch": self.rot_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.socket.write_root_state_to_sim(state["socket"], env_ids)
        self.plug.write_root_state_to_sim(state["plug"], env_ids)
        self.distractor.write_root_state_to_sim(state["distractor"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.approach_latch[env_ids] = state["approach_latch"]
        self.depth_latch[env_ids] = state["depth_latch"]
        self.rot_latch[env_ids] = state["rot_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A dark-gray socket block stands on the table. Its rectangular well "
            f"({2 * c.well_x_half * 1000:.0f} x {2 * c.well_y_half * 1000:.0f} mm inside) is "
            f"covered by two ORANGE overhang plates that leave one open slot strip, "
            f"{2 * c.open_half * 1000:.0f} mm wide, running straight across the mouth — the "
            f"slot's long direction is the socket's entry axis. A BLUE cylindrical plug "
            f"({2 * c.plug_r * 1000:.0f} mm across) stands upright nearby: it has two side "
            f"lugs at its base (tip to tip {2 * c.lug_tip_r * 1000:.0f} mm, wider than the "
            f"slot is narrow) and an elongated cap on top whose long axis is PARALLEL to the "
            f"lugs, so the lug direction is visible from above. A smooth RED cylinder of "
            f"similar size (no lugs) sits off to the side — it is a decoy.\n"
            f"Goal: bayonet-lock the blue plug in the socket. Rotate the plug so its lugs "
            f"line up with the open slot, lower it through the slot until it rests on the "
            f"well floor, then TWIST it about its vertical axis by roughly 55-90 degrees "
            f"(either direction) so the lugs ride under the orange plates and the plug can "
            f"no longer be lifted out. Leave it seated, upright and at rest. Pushing the "
            f"plug in without twisting it does NOT count — it must end locked. Putting the "
            f"red decoy in the well counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lower the blue lugged plug through the open slot of the socket well until it "
            "sits on the well floor, then twist it about 60 degrees so its lugs lock under "
            "the orange plates, and leave it seated at rest. Inserting without twisting "
            "fails; the red decoy counts for nothing."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _plug_local(self) -> torch.Tensor:
        """(N,3) plug centre in the SOCKET body frame (origin = well-floor centre at
        table level)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = self.plug.data.root_pos_w - self.socket.data.root_pos_w
        return quat_apply_inverse(self.socket.data.root_quat_w, rel)

    def _plug_up_z(self) -> torch.Tensor:
        """(N,) world-z of the plug's +z axis (1 = perfectly upright)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.plug.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    @staticmethod
    def _yaw(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def slot_distance_deg(self) -> torch.Tensor:
        """(N,) in [0, 90]: angular distance (deg) of the lug axis from the entry-slot
        axis, mod 180 (the lug pair is symmetric). 0 = lugs aligned with the slot
        (extractable), 90 = quarter turn (deep under the plates)."""
        rel = self._yaw(self.plug.data.root_quat_w) - self._yaw(self.socket.data.root_quat_w)
        a = torch.rad2deg(rel) % 180.0
        return torch.minimum(a, 180.0 - a)

    def inside_well(self) -> torch.Tensor:
        """(N,) bool: plug axis within the well footprint (with the body radius margin)."""
        c = self.cfg
        loc = self._plug_local()
        return (loc[:, 0].abs() <= c.well_x_half - c.plug_r + 0.004) & \
               (loc[:, 1].abs() <= c.well_y_half - c.plug_r + 0.004)

    def seated(self) -> torch.Tensor:
        """(N,) bool: inside the well, bottom at the well floor, upright."""
        c = self.cfg
        loc = self._plug_local()
        z_ok = (loc[:, 2] <= c.seated_z + c.seat_z_tol) & (loc[:, 2] >= c.seated_z - 0.01)
        upright = self._plug_up_z() >= math.cos(math.radians(c.upright_deg))
        return self.inside_well() & z_ok & upright

    def locked(self) -> torch.Tensor:
        """(N,) bool: seated AND rotated into the lock zone (lugs held under the plates)."""
        return self.seated() & (self.slot_distance_deg() >= self.cfg.lock_deg)

    def depth_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: seating progress from mouth level down to the well floor, gated
        on being inside the well footprint and roughly upright (a plug resting on the
        plates or lying on the table earns ~0)."""
        c = self.cfg
        loc = self._plug_local()
        raw = ((c.entry_z - loc[:, 2]) / (c.entry_z - c.seated_z)).clamp(0.0, 1.0)
        rough_up = self._plug_up_z() >= math.cos(math.radians(c.gate_upright_deg))
        return raw * (self.inside_well() & rough_up).float()

    def rot_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: lock-rotation progress (slot distance / lock_deg), gated on
        SEATED — twisting in the air or above the plates earns nothing."""
        raw = (self.slot_distance_deg() / self.cfg.lock_deg).clamp(0.0, 1.0)
        return raw * self.seated().float()

    def settled(self) -> torch.Tensor:
        return (self.plug.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) & \
               (self.plug.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best approach, best seating depth and best lock rotation each physics
        substep, so transient progress keeps its credit."""
        d = (self.plug.data.root_pos_w - self.socket.data.root_pos_w)[:, :2].norm(dim=-1)
        approach = (1.0 - d / self.d0).clamp(0.0, 1.0)
        self.approach_latch = torch.maximum(self.approach_latch, approach)
        self.depth_latch = torch.maximum(self.depth_latch, self.depth_frac())
        self.rot_latch = torch.maximum(self.rot_latch, self.rot_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: plug seated in the well, twisted into the lock zone, settled."""
        return self.locked() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * latched approach + 0.35 * latched seating depth +
        0.35 * latched lock rotation (max 0.85), 0.9 once seated + locked, 1.0 iff
        success. Doing nothing scores ~0; the seed's untwisted full-depth insertion caps
        at 0.5."""
        base = (0.15 * self.approach_latch + 0.35 * self.depth_latch
                + 0.35 * self.rot_latch).clamp(0.0, 0.85)
        s = torch.where(self.locked(), torch.maximum(base, base.new_tensor(0.9)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="plug_bayonet_lock", robot="null"))
