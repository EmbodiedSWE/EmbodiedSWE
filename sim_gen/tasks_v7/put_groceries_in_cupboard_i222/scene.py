"""GravityFeedRackScene — stock three colored cans into a store-style GRAVITY-FEED
FIFO rack so they face RED-GREEN-BLUE at the front window. Derived from
rlbench/put_groceries_in_cupboard, but the cupboard is a first-in-first-out
dispenser: the only thing the solver truly chooses is the LOADING ORDER, and that
order is irreversibly frozen into the goal arrangement by gravity.

Seed (rlbench/put_groceries_in_cupboard): identify a named grocery among
distractors, grasp it, carry it up, and SET IT DOWN on a passive open cupboard
shelf — any resting pose on the shelf, reached by lowering from above, is
terminal. Here that plan is dead on arrival, and a new one is forced:

- The rack is FULLY ENCLOSED: a 14-degree inclined ramp behind walls, a front
  stop wall, and a ramp-parallel roof. There is no open shelf anywhere. The sole
  opening at storage height is a 24 mm VIEWING SLOT in the front wall — less than
  half a can diameter, nothing passes (metric interlock, asserted).
- The only entry is a LOADING PORT in the roof line at the TOP-REAR, ringed by
  45-degree funnel flares. Its downhill extent (82.5 mm) is shorter than a can
  (90 mm), so a can cannot be posted lying along the slope (asserted); the
  intended feed is a can dropped CROSS-LANE through the port.
- A can released in the port falls onto the ramp and GRAVITY takes over: it
  rolls downhill under the roof and queues against the front stop (or against
  the can already there). The roof gap (78 mm) admits exactly one rolling can —
  no stacking, no leapfrogging, no standing under the roof (asserted). Once a
  can is in the queue there is NO way to reorder: the port is behind the queue,
  the slot is too small, the roof pins everything down. FIRST IN = FRONTMOST.
- Success reads back the facing arrangement: RED at the window, GREEN behind
  it, BLUE at the rear — three cans resting cross-lane in a contact queue.
  The ONLY policy that produces it is load red, then green, then blue.

So a solver needs a different PLAN (deliver each can to a loading port and let
a passive gravity mechanism do the placement, sequencing the colors correctly)
and different CODE STRUCTURE (an ordered loop of pick-drop-wait-verify episodes
where the load-bearing decision is WHICH can goes next, not where to set it
down). Strict execution order: red before green before blue — enforced by the
physics of the queue itself, not by bookkeeping.

Assets are fully procedural (pen_holder-pattern compound spawner for the rack;
cans are plain cylinders). The rack is one KINEMATIC compound with no joints,
re-posed each reset (yaw + xy); per-episode randomization: rack pose, can slot
PERMUTATION over three floor slots + xy jitter — all readback-verifiable.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.10 lifted  — red can ever raised above 0.15 m (latched)
  0.10 entered — red can ever inside the lane under the roof line (latched)
  0.20 q1      — red can ever seated at the FRONT window slot (latched, streak)
  0.20 q2      — red AND green ever seated at slots 1+2 together (latched, streak)
  1.0 iff success() — red/green/blue seated at slots 1/2/3 cross-lane, all
                 settled, all finite — judged LIVE. Non-success cap 0.60.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawner -----------------------------------------------------------------
# One rigid body, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             material=None, quat=None) -> None:
    """Author one box collider (optionally rotated: quat = (w, x, y, z))."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if material is not None:
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, materialPurpose="physics")


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _make_material(stage, path: str, mu: float):
    """DEFINED-friction physics material (the rolling/no-slide budget must be known,
    not backend-default: cans must ROLL down the ramp, and a standing can must NOT
    slide). Restitution 0 (the queue must thud, not bounce)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(mu))
    pm.CreateDynamicFrictionAttr(float(mu))
    pm.CreateRestitutionAttr(0.0)
    return mat


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


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gravity-feed rack: KINEMATIC compound. Origin at the base centre on
    the ground; local +x = DOWNHILL (toward the front window), +y across the lane.
    Interior: inclined ramp (surface z = z_front + (ramp_x1 - x) * tan(incline)),
    side walls, rear wall, front stop wall with a viewing slot, a ramp-parallel roof
    from roof_x0 to the front, and 45-degree slick funnel FLARES around the top-rear
    loading port (the open roof span ramp_x0..roof_x0)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    mat = _make_material(stage, f"{prim_path}/rack_mat", cfg.mu)
    slick = _make_material(stage, f"{prim_path}/flare_mat", cfg.flare_mu)
    c = cfg
    inc = math.radians(c.incline_deg)
    s, co, ta = math.sin(inc), math.cos(inc), math.tan(inc)
    x0, x1, th = c.ramp_x0, c.ramp_x1, c.slab_t
    half_w = c.lane_w / 2
    out_x0, out_x1 = x0 - c.wall_t, x1 + c.wall_t

    def zsurf(x: float) -> float:
        return c.z_front + (x1 - x) * ta

    qy = (math.cos(inc / 2), 0.0, math.sin(inc / 2), 0.0)  # ramp/roof tilt about y
    # ramp slab: top surface = zsurf, span embedded into both end walls
    ramp_len = (x1 - x0) / co + 0.02
    _add_box(stage, f"{prim_path}/ramp",
             center=(-th / 2 * s, 0.0, zsurf(0.0) - th / 2 * co),
             size=(ramp_len, c.lane_w, th),
             color=c.rack_color, collide=collide, material=mat, quat=qy)
    # roof slab: underside = zsurf + roof_gap, from the port's front edge to the front
    rx_m = (c.roof_x0 + out_x1) / 2
    _add_box(stage, f"{prim_path}/roof",
             center=(rx_m + th / 2 * s, 0.0, zsurf(rx_m) + c.roof_gap + th / 2 * co),
             size=((out_x1 - c.roof_x0) / co, c.lane_w + 2 * c.wall_t, th),
             color=c.roof_color, collide=collide, material=mat, quat=qy)
    # side walls (full length, ground to wall_h)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (half_w + c.wall_t / 2), c.wall_h / 2),
                 size=(out_x1 - out_x0, c.wall_t, c.wall_h),
                 color=c.rack_color, collide=collide, material=mat)
    # rear wall
    _add_box(stage, f"{prim_path}/wall_rear",
             center=(x0 - c.wall_t / 2, 0.0, c.wall_h / 2),
             size=(c.wall_t, c.lane_w, c.wall_h),
             color=c.rack_color, collide=collide, material=mat)
    # front stop wall, split around the viewing slot (slot_z0..slot_z1)
    _add_box(stage, f"{prim_path}/wall_front_lo",
             center=(x1 + c.wall_t / 2, 0.0, c.slot_z0 / 2),
             size=(c.wall_t, c.lane_w, c.slot_z0),
             color=c.front_color, collide=collide, material=mat)
    _add_box(stage, f"{prim_path}/wall_front_hi",
             center=(x1 + c.wall_t / 2, 0.0, (c.slot_z1 + c.wall_h) / 2),
             size=(c.wall_t, c.lane_w, c.wall_h - c.slot_z1),
             color=c.front_color, collide=collide, material=mat)
    # funnel flares over the loading port (slick, 45 deg out)
    fl, ft = c.flare_len, c.flare_t
    d = fl / 2 * math.cos(math.radians(45.0))
    q45 = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    # rear flare: hinged at the rear wall inner top, tilting up-and-back
    _add_box(stage, f"{prim_path}/flare_rear",
             center=(x0 - d, 0.0, c.wall_h + d),
             size=(fl, c.lane_w + 2 * c.wall_t, ft),
             color=c.flare_color, collide=collide, material=slick,
             quat=(q45[0], 0.0, q45[1], 0.0))
    # front flare: hinged just above the roof's leading edge, tilting up-and-front
    _add_box(stage, f"{prim_path}/flare_front",
             center=(c.roof_x0 + d, 0.0, c.wall_h - 0.01 + d),
             size=(fl, c.lane_w + 2 * c.wall_t, ft),
             color=c.flare_color, collide=collide, material=slick,
             quat=(q45[0], 0.0, -q45[1], 0.0))
    # side flares over the port span, tilting up-and-out
    fx0, fx1 = x0 - c.wall_t, c.roof_x0 + 0.02
    for sgn, nm in ((1.0, "flare_l"), (-1.0, "flare_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((fx0 + fx1) / 2, sgn * (half_w + d), c.wall_h + d),
                 size=(fx1 - fx0, fl, ft),
                 color=c.flare_color, collide=collide, material=slick,
                 quat=(q45[0], sgn * q45[1], 0.0, 0.0))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclass (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            incline_deg: float = 14.0
            ramp_x0: float = -0.135
            ramp_x1: float = 0.135
            z_front: float = 0.05
            lane_w: float = 0.098
            wall_t: float = 0.015
            wall_h: float = 0.20
            slab_t: float = 0.02
            roof_gap: float = 0.078
            roof_x0: float = -0.0525
            slot_z0: float = 0.098
            slot_z1: float = 0.122
            flare_len: float = 0.06
            flare_t: float = 0.008
            mu: float = 0.60
            flare_mu: float = 0.10
            rack_color: tuple = (0.42, 0.36, 0.28)
            roof_color: tuple = (0.32, 0.27, 0.20)
            front_color: tuple = (0.14, 0.14, 0.16)
            flare_color: tuple = (0.55, 0.50, 0.40)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GravityFeedRackSceneCfg(BaseCfg):
    """Config for `GravityFeedRackScene`. The interlocks are metric and asserted:
    the viewing slot passes no can, the port passes no can lying along the slope,
    the roof gap admits exactly one rolling can (no stacking / standing under it),
    the rearmost queue slot still sits fully under the roof, and adjacent queue
    slots are farther apart than twice the x tolerance (windows cannot alias)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.12)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(3.0)  # max |ang vel| when judging (rad/s)
    axis_cos: float = tunable(0.95)  # |can axis . lane cross axis| at a slot
    x_tol: float = tunable(0.016)  # queue slot half-window, downhill (m)
    y_tol: float = tunable(0.020)  # queue slot half-window, across (m)
    z_tol: float = tunable(0.012)  # queue slot half-window, height (m)
    lift_z: float = tunable(0.15)  # 'lifted' latch height (m)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    cab_xy_jitter: float = tunable(0.03)  # rack base xy jitter (+/- m)
    cab_yaw_jitter_deg: float = tunable(15.0)  # rack yaw jitter (+/- deg)
    slot_jitter: float = tunable(0.025)  # per-can floor-slot xy jitter (+/- m)
    shuffle_slots: bool = tunable(True)  # random permutation of cans over slots

    # --- info: layout (single Franka base at the origin) ----------------------------------------
    cab_pos: tuple = info((0.48, 0.0))  # rack base centre (nominal, on the ground)
    cab_yaw_deg: float = info(180.0)  # nominal yaw: local +x (downhill) faces the robot
    slots: tuple = info(((0.28, -0.26), (0.20, 0.00), (0.28, 0.26)))  # can floor slots

    # --- info: rack structure (rack-local: origin base centre, +x downhill) ---------------------
    incline_deg: float = info(14.0)  # ramp incline (rolls a faceted cylinder reliably)
    ramp_x0: float = info(-0.135)  # ramp rear edge (interior)
    ramp_x1: float = info(0.135)  # ramp front edge (interior, at the stop wall)
    z_front: float = info(0.05)  # ramp surface height at the front edge
    lane_w: float = info(0.098)  # interior lane width: can + 4 mm play per side
    wall_t: float = info(0.015)
    wall_h: float = info(0.20)
    slab_t: float = info(0.02)  # ramp/roof slab thickness
    roof_gap: float = info(0.078)  # roof underside height above the ramp surface
    roof_x0: float = info(-0.0525)  # roof leading edge = the port's front edge
    slot_z0: float = info(0.098)  # viewing slot lower edge (front wall)
    slot_z1: float = info(0.122)  # viewing slot upper edge
    flare_len: float = info(0.06)
    flare_t: float = info(0.008)
    rack_mu: float = info(0.60)  # defined friction of ramp/walls (cans roll, not slide)
    flare_mu: float = info(0.10)  # slick funnel flares

    # --- info: cans (identical geometry, distinguishable ONLY by color) -------------------------
    can_r: float = info(0.030)
    can_l: float = info(0.090)
    can_mass: float = info(0.25)
    can_mu: float = info(0.60)
    red_color: tuple = info((0.85, 0.10, 0.08))  # goal: FRONT (window)
    green_color: tuple = info((0.10, 0.62, 0.18))  # goal: MIDDLE
    blue_color: tuple = info((0.12, 0.28, 0.85))  # goal: REAR

    contact_offset: float = info(0.002)
    streak_len: int = info(6)  # substeps a queue slot must hold before its latch
    # rubric weights (0.10 + 0.10 + 0.20 + 0.20 = 0.60 = the non-success cap)
    w_lift: float = info(0.10)
    w_enter: float = info(0.10)
    w_q1: float = info(0.20)
    w_q2: float = info(0.20)

    # ----- derived geometry helpers (rack-local) -----
    def zsurf(self, x: float) -> float:
        """Ramp surface height at downhill coordinate x."""
        return self.z_front + (self.ramp_x1 - x) * math.tan(math.radians(self.incline_deg))

    def rest_x(self, i: int) -> float:
        """Queue slot i (0 = front/window) centre x: can 0 against the stop wall,
        each next can one contact diameter up-slope."""
        co = math.cos(math.radians(self.incline_deg))
        return (self.ramp_x1 - self.can_r) - i * 2.0 * self.can_r * co

    def rest_z(self, i: int) -> float:
        """Queue slot i centre height: axis a radius above the ramp surface."""
        co = math.cos(math.radians(self.incline_deg))
        return self.zsurf(self.rest_x(i)) + self.can_r / co

    def __post_init__(self) -> None:
        r, ll = self.can_r, self.can_l
        inc = math.radians(self.incline_deg)
        co, ta = math.cos(inc), math.tan(inc)
        port_x = self.roof_x0 - self.ramp_x0  # port downhill extent
        # the port passes no can lying along the slope, but clears a dropping can
        assert 2 * r + 0.02 <= port_x / co <= ll - 0.004, "port slope-length interlock"
        assert ll >= port_x + 0.004, "axis-downhill horizontal footprint must not pass"
        # cross-lane feed: can + play fits the port across, and the lane guides it
        assert self.lane_w >= ll + 0.006, "port/lane must pass a cross-lane can"
        assert 0.003 <= (self.lane_w - ll) / 2 <= 0.006, "lane play keeps cans square"
        # roof admits exactly one rolling can: no stacking, no standing beneath
        assert 2 * r + 0.014 <= self.roof_gap <= 3 * r, "roof gap band"
        assert ll > self.roof_gap + 0.008, "a can cannot stand under the roof"
        assert 4 * r > self.roof_gap, "a can cannot ride over another under the roof"
        # the rearmost queue slot is fully under the roof (port is behind the queue)
        assert self.rest_x(2) - r >= self.roof_x0 + 0.008, "queue must clear the port"
        # the viewing slot passes nothing and sits above the front can's centre
        assert self.slot_z1 - self.slot_z0 <= 2 * r - 0.02, "slot passes no can"
        assert self.slot_z0 >= self.rest_z(0) + 0.008, "front can rests on solid wall"
        assert self.slot_z1 <= self.rest_z(0) + 2 * r, "the front can is visible"
        # queue windows cannot alias and physics is in the intended regimes
        assert 2 * r * co >= 2 * self.x_tol + 0.004, "adjacent windows must not overlap"
        assert self.incline_deg >= 8.0, "incline must roll a faceted cylinder"
        assert self.can_mu > ta + 0.15, "no sliding: cans roll, standing cans stay"
        assert ta < r / (ll / 2), "a standing can in the port shaft is stable (honest near-miss)"
        assert self.wall_h >= self.zsurf(self.roof_x0) + self.roof_gap + self.slab_t / co + 0.004, \
            "walls must reach above the roof"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gravity_feed_rack")
class GravityFeedRackScene(BaseScene):
    cfg: GravityFeedRackSceneCfg

    def __init__(self, cfg: GravityFeedRackSceneCfg | None = None) -> None:
        super().__init__(cfg or GravityFeedRackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        rack_spawn = spawners["rack"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            incline_deg=c.incline_deg, ramp_x0=c.ramp_x0, ramp_x1=c.ramp_x1,
            z_front=c.z_front, lane_w=c.lane_w, wall_t=c.wall_t, wall_h=c.wall_h,
            slab_t=c.slab_t, roof_gap=c.roof_gap, roof_x0=c.roof_x0,
            slot_z0=c.slot_z0, slot_z1=c.slot_z1, flare_len=c.flare_len,
            flare_t=c.flare_t, mu=c.rack_mu, flare_mu=c.flare_mu,
            contact_offset=c.contact_offset,
        )
        yaw = math.radians(c.cab_yaw_deg)

        def can_cfg(name: str, color: tuple, slot: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_l,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5, angular_damping=0.05,
                        solver_velocity_iteration_count=4,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.can_mu, dynamic_friction=c.can_mu,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(slot[0], slot[1], c.can_l / 2 + 0.002)),
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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cab_pos[0], c.cab_pos[1], 0.0),
                    rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
            ),
            "red": can_cfg("RedCan", c.red_color, c.slots[0]),
            "green": can_cfg("GreenCan", c.green_color, c.slots[1]),
            "blue": can_cfg("BlueCan", c.blue_color, c.slots[2]),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.red: RigidObject = env.iscene["red"]
        self.green: RigidObject = env.iscene["green"]
        self.blue: RigidObject = env.iscene["blue"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._lift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._enter_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._q1_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._q2_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._q1_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._q2_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack re-posed (kinematic, jointless — safe to teleport) with
        xy + yaw jitter; cans random-permuted over the three floor slots with xy
        jitter, standing upright; latches cleared; stale external-force buffers
        zeroed (set_external_force_and_torque persists across resets)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rack: pose jitter about the nominal base pose ---
        yaw = math.radians(c.cab_yaw_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cab_yaw_jitter_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.cab_pos[0]
        st[:, 1] = c.cab_pos[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.cab_xy_jitter
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- cans: permutation over floor slots + jitter, standing upright ---
        slots = torch.tensor(c.slots, device=dev)  # (3, 2)
        if c.shuffle_slots:
            perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)
        else:
            perm = torch.arange(3, device=dev).expand(m, 3)
        for k, body in enumerate((self.red, self.green, self.blue)):
            xy = slots[perm[:, k]] \
                + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.can_l / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches + stale force buffers ---
        self._lift_ever[env_ids] = False
        self._enter_ever[env_ids] = False
        self._q1_streak[env_ids] = 0
        self._q2_streak[env_ids] = 0
        self._q1_ever[env_ids] = False
        self._q2_ever[env_ids] = False
        n = self.env.num_envs
        zero = torch.zeros(n, 1, 3, device=dev)
        for body in (self.red, self.green, self.blue):
            body.set_external_force_and_torque(zero, zero)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "green": self.green.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "lift_ever": self._lift_ever[env_ids].clone(),
            "enter_ever": self._enter_ever[env_ids].clone(),
            "q1_streak": self._q1_streak[env_ids].clone(),
            "q2_streak": self._q2_streak[env_ids].clone(),
            "q1_ever": self._q1_ever[env_ids].clone(),
            "q2_ever": self._q2_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.green.write_root_state_to_sim(state["green"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self._lift_ever[env_ids] = state["lift_ever"]
        self._enter_ever[env_ids] = state["enter_ever"]
        self._q1_streak[env_ids] = state["q1_streak"]
        self._q2_streak[env_ids] = state["q2_streak"]
        self._q1_ever[env_ids] = state["q1_ever"]
        self._q2_ever[env_ids] = state["q2_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A brown GRAVITY-FEED CAN RACK stands on the floor, its dark front wall "
            f"facing the robot: a fully enclosed cabinet with a "
            f"{c.incline_deg:.0f}-degree ramp inside that slopes DOWN toward the "
            f"front. There is no open shelf anywhere: side walls, a rear wall, a "
            f"roof over the ramp, and a front stop wall enclose the lane completely. "
            f"The front wall has only a narrow horizontal VIEWING SLOT "
            f"({(c.slot_z1 - c.slot_z0) * 1000:.0f} mm tall — less than half a can), "
            f"through which the frontmost stored can shows its color. The ONLY way "
            f"in is the LOADING PORT: an opening in the top of the cabinet at the "
            f"HIGH, REAR end of the ramp, ringed by funnel plates. A can dropped "
            f"through the port lying CROSSWAYS lands on the ramp and rolls downhill "
            f"under the roof by itself, queuing up against the front stop wall — or "
            f"against the cans already stored. The port is too short for a can "
            f"lying along the slope, the roof is too low to stand a can under or "
            f"to let one roll over another, so the queue order can NEVER be "
            f"changed after loading: FIRST IN ends up FRONTMOST, at the window.\n"
            f"On the floor in front of the rack stand THREE identical cans "
            f"(diameter {2 * c.can_r * 100:.0f} cm, length {c.can_l * 100:.0f} cm), "
            f"distinguishable ONLY by color: RED, GREEN and BLUE, their floor "
            f"positions shuffled every episode.\n"
            f"Goal: stock the rack so it faces RED at the window, GREEN in the "
            f"middle, BLUE at the rear. Because the rack is first-in-first-out, "
            f"that means loading the cans in exactly that order: pick up the RED "
            f"can, drop it crossways through the port, let it roll down to the "
            f"window; then the GREEN can; then the BLUE can. SUCCESS is the "
            f"read-back arrangement: red, green, blue resting crossways in a "
            f"contact queue on the ramp (front to rear), everything settled. "
            f"Loading in any other order, leaving a can standing in the port, "
            f"resting a can on the roof, or posting it at the viewing slot — all "
            f"failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stock the gravity-feed rack in first-in-first-out order: drop the red "
            "can crossways through the top-rear loading port and let it roll to the "
            "front window, then load the green can, then the blue can, so the rack "
            "faces red-green-blue from the front."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _rack_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the rack frame (+x downhill, +y across)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.rack.data.root_pos_w
        return quat_apply_inverse(self.rack.data.root_quat_w, rel)

    def _axis_y(self, body: RigidObject) -> torch.Tensor:
        """(N,) |can axis . rack cross-lane axis| (cylinder axis = can local +z)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        a_w = quat_apply(body.data.root_quat_w, ez)
        return quat_apply_inverse(self.rack.data.root_quat_w, a_w)[:, 1].abs()

    def _zsurf_t(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return c.z_front + (c.ramp_x1 - x) * math.tan(math.radians(c.incline_deg))

    def _in_lane(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the lane volume under the roof line (a can on
        the roof, in the port funnel or hovering above does NOT count)."""
        c = self.cfg
        loc = self._rack_local(body)
        dz = loc[:, 2] - self._zsurf_t(loc[:, 0])
        return ((loc[:, 0] >= c.ramp_x0) & (loc[:, 0] <= c.ramp_x1)
                & (loc[:, 1].abs() <= c.lane_w / 2)
                & (dz >= 0.005) & (dz <= c.roof_gap - 0.003))

    def _seated_slot(self, body: RigidObject, i: int) -> torch.Tensor:
        """(N,) bool: body resting cross-lane at queue slot i (0 = front/window)."""
        c = self.cfg
        loc = self._rack_local(body)
        return ((loc[:, 0] - c.rest_x(i)).abs() <= c.x_tol) \
            & (loc[:, 1].abs() <= c.y_tol) \
            & ((loc[:, 2] - c.rest_z(i)).abs() <= c.z_tol) \
            & (self._axis_y(body) >= c.axis_cos)

    def _settled(self, body: RigidObject) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for body in (self.red, self.green, self.blue):
            ok &= torch.isfinite(body.data.root_state_w).all(dim=-1)
        return ok

    # ----- step-coupled bookkeeping (every substep; latches update ONLY here so the
    # streak counters advance once per physics step, never per rubric query) ----------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        z_env = self.red.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self._lift_ever |= z_env > c.lift_z
        self._enter_ever |= self._in_lane(self.red)
        q1_now = self._seated_slot(self.red, 0)
        q2_now = q1_now & self._seated_slot(self.green, 1)
        self._q1_streak = torch.where(q1_now, self._q1_streak + 1,
                                      torch.zeros_like(self._q1_streak))
        self._q2_streak = torch.where(q2_now, self._q2_streak + 1,
                                      torch.zeros_like(self._q2_streak))
        self._q1_ever |= self._q1_streak >= c.streak_len
        self._q2_ever |= self._q2_streak >= c.streak_len

    def success(self) -> torch.Tensor:
        """(N,) bool, judged LIVE: red at the window slot, green behind it, blue at
        the rear — all cross-lane in the queue, all settled, all finite."""
        seated = (self._seated_slot(self.red, 0)
                  & self._seated_slot(self.green, 1)
                  & self._seated_slot(self.blue, 2))
        still = (self._settled(self.red) & self._settled(self.green)
                 & self._settled(self.blue))
        return seated & still & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*lifted + 0.10*entered + 0.20*q1 + 0.20*q2 —
        all latched, ~0 for doing nothing, capped 0.60 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        base = (c.w_lift * self._lift_ever.float()
                + c.w_enter * self._enter_ever.float()
                + c.w_q1 * self._q1_ever.float()
                + c.w_q2 * self._q2_ever.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="gravity_feed_rack", robot="null"))
