"""SauceChuteScene — dock the tray under the covered chute, roll the captive sauce can
down into it, and deliver the loaded tray to the pad.

Derived from libero_90/living_room_scene3 "pick up the tomato sauce and put it in the
tray", but the MANIPULATION IS INVERTED AND MECHANIZED: the seed's plan is grasp the
freestanding sauce can among distractors, carry it through free space, and lower it
into a static tray. Here the RED SAUCE CAN CANNOT BE GRASPED AT ALL — it rests in a
detent cradle inside an elevated, roofed delivery chute (roof clearance 14 mm above
the can; the only access is a 55 x 50 mm push port in the rear wall, smaller than the
can in both length and diameter, sized for a poked fingertip and nothing else). The
can can only leave the chute by ROLLING: a nudge through the port pops it over the
cradle ridge and the 2-degree ramp carries it to the open spout, off the edge, into
free fall. The TRAY is the object the solver actually transports — twice: first EMPTY
to the catch dock under the spout (so the falling can lands inside it — the red can
must NEVER land outside the tray; a canned good that hits the floor is spoiled and the
episode is permanently failed), then LOADED onto the green delivery pad. A white
decoy can of identical shape lies loose in the open — the one object the seed's
strategy could act on — and must be left alone. The seed's plan (pick the loose can,
place it in the tray) is executable here only on the WRONG can, and the end state it
would produce — a loose can placed by hand into the tray on the pad — is an explicit
smoke-tested failure both for the decoy (wrong object) and for the red can (it can
only become loose by hitting the floor, which latches permanent failure).

Assets are fully procedural (compound spawners with raw pxr authoring for chute and
tray; built-in CapsuleCfg / CuboidCfg for cans and pad):
  - chute: KINEMATIC compound at a fixed pose — ground pedestal, a 2-deg declined
    channel floor (127 mm inner width), side walls, a parallel roof 74 mm above the
    floor, a rear wall pierced by the push port, a two-ridge detent cradle holding
    the can near the port, and an open spout overhanging the pedestal face by 50 mm
    so even a dead-slow exit drops into the docked tray's airspace.
  - can (red) / decoy (white): capsules r=30 mm, 115 mm long (capsules roll smoothly
    — analytic collision, no convex facets to stall the 2-deg ramp), 0.35 kg.
  - tray: DYNAMIC compound open box, outer 216 x 210 mm, floor 16 mm, walls 54 mm
    (rim 70 mm — 24 mm above a resting can's centre, retaining the drop), 0.45 kg,
    origin at the bottom centre. Grasp feature: any 8 mm wall top edge.
  - pad: KINEMATIC green plate 280 x 280 x 14 mm, re-posed per episode.

Per-episode randomization (readback-verifiable): tray start pose (xy jitter + free
yaw), delivery pad position (xy jitter), decoy pose (xy jitter + free yaw), can seat
x-jitter in the cradle. The chute never moves.

Rubric (0..1; stage credit latched so correct transients keep their credit):
  0.15 * docked    — tray ever settled upright at the catch dock (latched)
  0.15 * roll      — chute progress: running MIN of the can's y mapped over
                     cradle -> spout tip (exactly 0 for doing nothing)
  0.15 * exited    — can ever left the chute (dropped below the channel floor)
  0.25 * caught    — can ever settled INSIDE the tray without ever having landed
                     outside it (latched; unreachable once `floored` is latched)
  0.15 * delivered — caught AND the tray on the pad, settled (latched)
  1.0 iff success() — can settled inside the upright tray, tray resting on the pad,
  the can never floored, the white decoy not in the tray, everything at rest.
  Non-success cap 0.85.

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
             rot_x_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_x_deg:
        h = math.radians(rot_x_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(math.sin(h), 0.0, 0.0)))
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


def _spawn_chute(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the delivery chute: KINEMATIC compound, origin at the world origin of
    the layout (child centres carry the full placement). Channel runs along -y from
    the rear wall (push port) to the open spout tip; the floor and roof are parallel
    slabs declined `ramp_deg` toward the spout (rear end HIGHER)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    cx = c.chute_x
    s = math.sin(math.radians(c.ramp_deg))

    def floor_top(y: float) -> float:
        return c.floor_top_exit + (y - c.chan_y0) * s

    yc = (c.chan_y0 + c.chan_y1) / 2
    chan_len = c.chan_y1 - c.chan_y0
    ftc = floor_top(yc)
    slab_w = c.inner_w + 2 * c.wall_t * 2  # generous: floor/roof span past the walls
    # pedestal (support column under the channel, front face = the drop face)
    _add_box(stage, f"{prim_path}/pedestal",
             center=(cx, (c.ped_y0 + c.chan_y1) / 2, c.ped_h / 2),
             size=(0.18, c.chan_y1 - c.ped_y0, c.ped_h), color=c.ped_color, collide=collide)
    # declined channel floor (spout tip overhangs the pedestal face)
    _add_box(stage, f"{prim_path}/floor", center=(cx, yc, ftc - c.floor_t / 2),
             size=(slab_w, chan_len, c.floor_t), color=c.color, collide=collide,
             rot_x_deg=c.ramp_deg)
    # parallel roof, underside `roof_gap` above the floor top
    _add_box(stage, f"{prim_path}/roof", center=(cx, yc, ftc + c.roof_gap + c.floor_t / 2),
             size=(slab_w, chan_len, c.floor_t), color=c.roof_color, collide=collide,
             rot_x_deg=c.ramp_deg)
    # side walls
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(cx + sgn * (c.inner_w / 2 + c.wall_t / 2), yc, ftc + c.roof_gap / 2),
                 size=(c.wall_t, chan_len, c.roof_gap + 2 * c.floor_t + 0.02),
                 color=c.color, collide=collide)
    # rear wall with the push port (hole: |x-cx| < port_w/2, sill..sill+port_h)
    wy = c.chan_y1 - c.back_t / 2
    sill = floor_top(c.chan_y1 - c.back_t)
    full_w = c.inner_w + 2 * c.wall_t
    strip_w = (full_w - c.port_w) / 2
    for sgn, nm in ((1.0, "back_xp"), (-1.0, "back_xn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(cx + sgn * (c.port_w / 2 + strip_w / 2), wy,
                         (sill - 0.010 + sill + c.port_h) / 2),
                 size=(strip_w, c.back_t, c.port_h + 0.010), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/back_top",
             center=(cx, wy, sill + c.port_h + 0.018),
             size=(full_w, c.back_t, 0.036), color=c.color, collide=collide)
    # detent cradle: two low ridges across the floor, holding the can near the port
    for dy, nm in ((-c.ridge_dy, "ridge_front"), (c.ridge_dy, "ridge_back")):
        ry = c.cradle_y + dy
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(cx, ry, floor_top(ry) + c.ridge_h - 0.007),
                 size=(c.inner_w - 0.004, 0.006, 0.014), color=c.ridge_color,
                 collide=collide)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray: DYNAMIC open box, origin at the BOTTOM CENTRE (root z ~= 0
    when resting on the ground)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    ox, oy = c.outer_x, c.outer_y
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             size=(ox, oy, c.floor_t), color=c.color, collide=collide)
    wz = c.floor_t + c.wall_h / 2
    for sgn, nm in ((1.0, "wall_xp"), (-1.0, "wall_xn")):
        _add_box(stage, f"{prim_path}/{nm}", center=(sgn * (ox - c.wall_t) / 2, 0.0, wz),
                 size=(c.wall_t, oy, c.wall_h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_yp"), (-1.0, "wall_yn")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * (oy - c.wall_t) / 2, wz),
                 size=(ox - 2 * c.wall_t, c.wall_t, c.wall_h), color=c.color,
                 collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "chute" not in _SPAWNER_CACHE:

        @configclass
        class ChuteSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_chute)
            chute_x: float = 0.45
            chan_y0: float = 0.045
            chan_y1: float = 0.364
            ped_y0: float = 0.095
            ped_h: float = 0.150
            floor_top_exit: float = 0.160
            ramp_deg: float = 2.0
            inner_w: float = 0.127
            wall_t: float = 0.012
            floor_t: float = 0.012
            roof_gap: float = 0.074
            back_t: float = 0.008
            port_w: float = 0.055
            port_h: float = 0.050
            cradle_y: float = 0.315
            ridge_dy: float = 0.022
            ridge_h: float = 0.005
            color: tuple = (0.42, 0.45, 0.50)
            roof_color: tuple = (0.36, 0.39, 0.44)
            ped_color: tuple = (0.30, 0.32, 0.36)
            ridge_color: tuple = (0.25, 0.26, 0.28)
            contact_offset: float = 0.003

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            outer_x: float = 0.216
            outer_y: float = 0.210
            floor_t: float = 0.016
            wall_t: float = 0.008
            wall_h: float = 0.054
            color: tuple = (0.62, 0.44, 0.24)
            contact_offset: float = 0.003

        _SPAWNER_CACHE.update(chute=ChuteSpawnerCfg, tray=TraySpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SauceChuteSceneCfg(BaseCfg):
    """Config for `SauceChuteScene`. The captivity is metric: the roof leaves 14 mm
    over the seated can (no jaw fits), the port (55 x 50 mm) is smaller than the can
    (115 mm long, 60 mm dia) in both critical dimensions, and the only open face of
    the channel is the spout — rolling out is the can's only exit."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    in_x: float = tunable(0.095)  # can centre inside the tray: |x| bound (tray frame, m)
    in_y: float = tunable(0.092)  # ... |y| bound
    in_z_lo: float = tunable(0.028)  # ... z band above the tray bottom (resting can: 0.046)
    in_z_hi: float = tunable(0.062)  # ... rejects a can perched on the 70 mm rim (z 0.100)
    floored_z: float = tunable(0.048)  # can centre below this OUTSIDE the tray = floored
    out_x: float = tunable(0.113)  # tray OUTER footprint half-extents for the floored test
    out_y: float = tunable(0.110)
    dock_tol: float = tunable(0.035)  # tray centre within this of the dock target (each axis)
    pad_tol: float = tunable(0.030)  # tray centre within this of the pad centre (each axis)
    pad_z_tol: float = tunable(0.010)  # tray bottom within this of the pad top
    upright_deg: float = tunable(10.0)  # tray up-axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    exited_z: float = tunable(0.145)  # can centre below this = it left the chute floor

    # --- tunable: randomization (the task-family knobs) --------------------------------------------
    tray_jit: float = tunable(0.05)  # tray start xy jitter (+/- m)
    pad_jit: float = tunable(0.05)  # pad centre xy jitter (+/- m)
    decoy_jit: float = tunable(0.04)  # decoy xy jitter (+/- m)
    can_x_jit: float = tunable(0.004)  # can seat x jitter in the cradle (+/- m)
    yaw_free: bool = tunable(True)  # free yaw for tray and decoy (demo sets False)

    # --- info: layout (single Franka base at the origin; everything within 0.72 m) -----------------
    chute_x: float = info(0.45)  # channel centreline x
    chan_y0: float = info(0.045)  # spout tip y (channel runs -y; rear wall at chan_y1)
    chan_y1: float = info(0.364)
    ped_y0: float = info(0.095)  # pedestal (drop) face y
    ped_h: float = info(0.150)
    floor_top_exit: float = info(0.160)  # channel floor top at the spout tip
    ramp_deg: float = info(2.0)  # decline toward the spout
    inner_w: float = info(0.127)  # channel inner width (can is 115 long — no yaw room)
    wall_t: float = info(0.012)
    floor_t: float = info(0.012)
    roof_gap: float = info(0.074)  # floor top -> roof underside (can dia 60 + 14)
    back_t: float = info(0.008)
    port_w: float = info(0.055)  # push port: smaller than the can both ways
    port_h: float = info(0.050)
    cradle_y: float = info(0.315)  # can seat centre y (ridges at +/- ridge_dy)
    ridge_dy: float = info(0.022)
    ridge_h: float = info(0.005)
    dock_target: tuple = info((0.45, -0.02))  # tray centre for the catch dock
    tray_home: tuple = info((0.20, 0.38))  # tray spawn centre
    pad_home: tuple = info((0.16, -0.40))  # pad centre nominal
    decoy_home: tuple = info((0.66, -0.24))  # decoy spawn centre

    # --- info: bodies ------------------------------------------------------------------------------
    can_r: float = info(0.030)
    can_len: float = info(0.055)  # capsule cylinder section (total length 115)
    can_mass: float = info(0.35)
    can_color: tuple = info((0.78, 0.12, 0.10))  # RED — the sauce can (the payload)
    decoy_color: tuple = info((0.92, 0.92, 0.90))  # WHITE — identical shape, leave alone
    tray_outer: tuple = info((0.216, 0.210))
    tray_floor_t: float = info(0.016)
    tray_wall_t: float = info(0.008)
    tray_wall_h: float = info(0.054)
    tray_mass: float = info(0.45)
    tray_color: tuple = info((0.62, 0.44, 0.24))  # wood
    pad_size: tuple = info((0.28, 0.28, 0.014))
    pad_color: tuple = info((0.15, 0.60, 0.25))  # GREEN — the delivery pad
    contact_offset: float = info(0.003)

    # rubric weights (0.15+0.15+0.15+0.25+0.15 = 0.85 = the non-success cap)
    w_dock: float = info(0.15)
    w_roll: float = info(0.15)
    w_exit: float = info(0.15)
    w_catch: float = info(0.25)
    w_deliver: float = info(0.15)

    # Derived (filled in __post_init__).
    can_rest_z: float = field(default=None, init=False)  # can centre z seated in the cradle
    sill_z: float = field(default=None, init=False)  # push-port sill height

    def floor_top(self, y: float) -> float:
        return self.floor_top_exit + (y - self.chan_y0) * math.sin(math.radians(self.ramp_deg))

    def __post_init__(self) -> None:
        self.can_rest_z = self.floor_top(self.cradle_y) + self.can_r
        self.sill_z = self.floor_top(self.chan_y1 - self.back_t)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sauce_chute")
class SauceChuteScene(BaseScene):
    cfg: SauceChuteSceneCfg

    def __init__(self, cfg: SauceChuteSceneCfg | None = None) -> None:
        super().__init__(cfg or SauceChuteSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        chute_spawn = spawners["chute"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            chute_x=c.chute_x, chan_y0=c.chan_y0, chan_y1=c.chan_y1, ped_y0=c.ped_y0,
            ped_h=c.ped_h, floor_top_exit=c.floor_top_exit, ramp_deg=c.ramp_deg,
            inner_w=c.inner_w, wall_t=c.wall_t, floor_t=c.floor_t, roof_gap=c.roof_gap,
            back_t=c.back_t, port_w=c.port_w, port_h=c.port_h, cradle_y=c.cradle_y,
            ridge_dy=c.ridge_dy, ridge_h=c.ridge_h, contact_offset=c.contact_offset,
        )
        tray_spawn = spawners["tray"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.tray_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            outer_x=c.tray_outer[0], outer_y=c.tray_outer[1], floor_t=c.tray_floor_t,
            wall_t=c.tray_wall_t, wall_h=c.tray_wall_h, color=c.tray_color,
            contact_offset=c.contact_offset,
        )

        def can_spawn(color):
            # capsules roll smoothly (analytic collision — no convex facets to stall
            # the 2-deg ramp); vel_iters=4 kills the free-capsule phantom creep
            return sim_utils.CapsuleCfg(
                radius=c.can_r, height=c.can_len, axis="X",
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
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
            "chute": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Chute",
                spawn=chute_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_home[0], c.tray_home[1], 0.002)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_home[0], c.pad_home[1], c.pad_size[2] / 2)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=can_spawn(c.can_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.chute_x, c.cradle_y, c.can_rest_z + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=can_spawn(c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_home[0], c.decoy_home[1], c.can_r + 0.002)),
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
        self.chute: RigidObject = env.iscene["chute"]
        self.tray: RigidObject = env.iscene["tray"]
        self.pad: RigidObject = env.iscene["pad"]
        self.can: RigidObject = env.iscene["can"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: stage credit survives correct transients (rubric requirement)
        self._docked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._min_y = torch.full((n,), self.cfg.cradle_y, device=dev)
        self._exited = torch.zeros(n, dtype=torch.bool, device=dev)
        self._caught = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._floored = torch.zeros(n, dtype=torch.bool, device=dev)  # can hit open ground

    # ----- frames --------------------------------------------------------------------------------
    def _rel(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body root position relative to the env origin."""
        return body.data.root_pos_w - self.env_origins

    def _can_local(self) -> torch.Tensor:
        """(N, 3) can centre in the TRAY'S BODY FRAME (origin = tray bottom centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        d = self.can.data.root_pos_w - self.tray.data.root_pos_w
        return quat_apply_inverse(self.tray.data.root_quat_w, d)

    def tray_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.tray.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_deg))

    # ----- predicates ----------------------------------------------------------------------------
    def in_tray(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body centre inside the tray cavity (tray frame — holds while the
        tray is carried), below the rim (rejects rim-perch), above the floor."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        d = body.data.root_pos_w - self.tray.data.root_pos_w
        loc = quat_apply_inverse(self.tray.data.root_quat_w, d)
        return ((loc[:, 0].abs() < c.in_x) & (loc[:, 1].abs() < c.in_y)
                & (loc[:, 2] > c.in_z_lo) & (loc[:, 2] < c.in_z_hi))

    def _floored_now(self) -> torch.Tensor:
        """(N,) bool: the RED can low (below `floored_z`) and OUTSIDE the tray's outer
        footprint — i.e. it landed on the ground / pad / against the tray's outside.
        A can inside the tray (z_w 0.046+) sits inside the footprint and never trips."""
        c = self.cfg
        loc = self._can_local()
        low = self._rel(self.can)[:, 2] < c.floored_z
        outside = (loc[:, 0].abs() > c.out_x) | (loc[:, 1].abs() > c.out_y)
        return low & outside

    def tray_docked(self) -> torch.Tensor:
        """(N,) bool: tray settled upright at the catch dock under the spout."""
        c = self.cfg
        p = self._rel(self.tray)
        near = ((p[:, 0] - c.dock_target[0]).abs() < c.dock_tol) \
            & ((p[:, 1] - c.dock_target[1]).abs() < c.dock_tol)
        on_ground = p[:, 2] < 0.012
        still = self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return near & on_ground & self.tray_upright() & still

    def tray_on_pad(self) -> torch.Tensor:
        """(N,) bool: tray resting upright ON the pad (centre over the pad, bottom at
        the pad top)."""
        c = self.cfg
        p = self._rel(self.tray)
        q = self._rel(self.pad)
        near = ((p[:, 0] - q[:, 0]).abs() < c.pad_tol) & ((p[:, 1] - q[:, 1]).abs() < c.pad_tol)
        at_top = (p[:, 2] - c.pad_size[2]).abs() < c.pad_z_tol
        return near & at_top & self.tray_upright()

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.can.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: chute re-asserted at its fixed pose, can seated in the
        cradle (x jitter), tray at its spawn zone (xy jitter + free yaw), pad and
        decoy jittered, latches cleared. All draws via torch.rand."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, x, y, z, yaw=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            if yaw is None:
                st[:, 3] = 1.0
            else:
                st[:, 3] = torch.cos(yaw / 2)
                st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        write(self.chute, zero, zero, zero)
        # can: seated in the cradle, axis along x (capsule axis is baked — identity quat)
        can_x = c.chute_x + (torch.rand(m, device=dev) * 2 - 1) * c.can_x_jit
        write(self.can, can_x, torch.full((m,), c.cradle_y, device=dev),
              torch.full((m,), c.can_rest_z + 0.002, device=dev))
        # tray: spawn zone, free yaw
        yaw_t = ((torch.rand(m, device=dev) * 2 - 1) * math.pi if c.yaw_free
                 else torch.zeros(m, device=dev))
        tx = c.tray_home[0] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jit
        ty = c.tray_home[1] + (torch.rand(m, device=dev) * 2 - 1) * c.tray_jit
        write(self.tray, tx, ty, torch.full((m,), 0.002, device=dev), yaw_t)
        # pad: kinematic plate, jittered centre
        px = c.pad_home[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jit
        py = c.pad_home[1] + (torch.rand(m, device=dev) * 2 - 1) * c.pad_jit
        write(self.pad, px, py, torch.full((m,), c.pad_size[2] / 2, device=dev))
        # decoy: lying in the open, free yaw
        yaw_d = ((torch.rand(m, device=dev) * 2 - 1) * math.pi if c.yaw_free
                 else torch.zeros(m, device=dev))
        dx = c.decoy_home[0] + (torch.rand(m, device=dev) * 2 - 1) * c.decoy_jit
        dy = c.decoy_home[1] + (torch.rand(m, device=dev) * 2 - 1) * c.decoy_jit
        write(self.decoy, dx, dy, torch.full((m,), c.can_r + 0.002, device=dev), yaw_d)

        # --- clear latches ---
        self._docked[env_ids] = False
        self._min_y[env_ids] = c.cradle_y
        self._exited[env_ids] = False
        self._caught[env_ids] = False
        self._delivered[env_ids] = False
        self._floored[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "chute": self.chute.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "can": self.can.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "docked": self._docked[env_ids].clone(),
            "min_y": self._min_y[env_ids].clone(),
            "exited": self._exited[env_ids].clone(),
            "caught": self._caught[env_ids].clone(),
            "delivered": self._delivered[env_ids].clone(),
            "floored": self._floored[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.chute.write_root_state_to_sim(state["chute"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._docked[env_ids] = state["docked"]
        self._min_y[env_ids] = state["min_y"]
        self._exited[env_ids] = state["exited"]
        self._caught[env_ids] = state["caught"]
        self._delivered[env_ids] = state["delivered"]
        self._floored[env_ids] = state["floored"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An elevated slate-gray DELIVERY CHUTE stands on the floor on a pedestal: a "
            f"roofed channel about {(c.chan_y1 - c.chan_y0) * 100:.0f} cm long, its floor "
            f"{c.floor_top_exit * 100:.0f} cm up, sloping gently toward its open SPOUT "
            f"end (the mouth that overhangs the pedestal face). Seated in a shallow "
            f"cradle inside the channel, near the rear wall, lies a RED sauce can (a "
            f"{(c.can_len + 2 * c.can_r) * 1000:.0f} mm long, {2 * c.can_r * 1000:.0f} mm "
            f"thick cylinder, axis across the channel). The can CANNOT be grasped: the "
            f"roof leaves only {(c.roof_gap - 2 * c.can_r) * 1000:.0f} mm above it, and "
            f"the only other access is a small PUSH PORT in the rear wall "
            f"({c.port_w * 1000:.0f} x {c.port_h * 1000:.0f} mm — smaller than the can in "
            f"every direction, big enough for a poked fingertip). Elsewhere on the floor: "
            f"an open wooden TRAY ({c.tray_outer[0] * 100:.0f} x "
            f"{c.tray_outer[1] * 100:.0f} cm, {(c.tray_floor_t + c.tray_wall_h) * 100:.1f} "
            f"cm walls — grasp it by any wall top edge), a flat GREEN DELIVERY PAD whose "
            f"position varies between episodes, and a loose WHITE can identical in shape "
            f"to the red one — a decoy that must be LEFT ALONE.\n"
            f"Goal: deliver the RED can inside the tray onto the green pad, using the "
            f"chute — the only way to get the can out. First carry the EMPTY tray to the "
            f"catch dock on the ground directly under the spout (centre it on the spout "
            f"mouth, snug to the pedestal face). Then reach through the rear push port "
            f"and nudge the red can GENTLY forward: it pops over the cradle ridge, rolls "
            f"down the channel, drops off the spout and lands in the tray. Push gently — "
            f"a hard flick can throw the can past the tray. THE RED CAN MUST NEVER LAND "
            f"OUTSIDE THE TRAY: if it ever touches the open floor or the pad it is "
            f"spoiled and the episode has permanently failed (so the tray must be docked "
            f"BEFORE the can is pushed). Finally carry the LOADED tray (with the can "
            f"riding inside) onto the green pad and set it down flat, fully on the pad. "
            f"Success: the red can settled inside the upright tray, the tray resting on "
            f"the pad, the white decoy still outside the tray, everything at rest. "
            f"Putting the WHITE can in the tray counts for nothing, and a tipped tray or "
            f"a tray beside the pad does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the empty wooden tray on the floor under the chute's spout, then "
            "reach through the rear port and gently push the red sauce can so it rolls "
            "down the chute and drops into the tray. Carry the loaded tray onto the "
            "green pad and set it down flat. The red can must never land outside the "
            "tray, and the white decoy can must stay out of the tray."
        )

    # ----- rubric ----------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        c = self.cfg
        # floored FIRST: a can that lands outside can never earn `caught` afterwards
        self._floored |= self._floored_now()
        self._docked |= self.tray_docked()
        self._min_y = torch.minimum(self._min_y, self._rel(self.can)[:, 1])
        self._exited |= self._rel(self.can)[:, 2] < c.exited_z
        can_still = self.can.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._caught |= self.in_tray(self.can) & can_still & ~self._floored
        self._delivered |= (self._caught & self.in_tray(self.can) & self.tray_on_pad()
                            & self.settled())

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No mechanism plant — the chute is static and gravity does the rolling. The
        scene only latches rubric progress at sim rate (so the brief flight through
        the drop is never missed)."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the red can settled INSIDE the upright tray, the tray resting ON
        the pad, the can never floored (it can only have entered the tray through the
        chute drop or — never available to a real solver — a clean airborne path), the
        white decoy outside the tray, everything at rest. Physical outcomes: live
        settled poses plus the one declared trajectory rule (the no-floor clause)."""
        self._update_latches()
        return (self.in_tray(self.can) & self.tray_on_pad() & self.tray_upright()
                & ~self._floored & ~self.in_tray(self.decoy) & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*docked + 0.15*roll progress + 0.15*exited +
        0.25*caught + 0.15*delivered — all latched/rising-only, exactly 0 for doing
        nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        roll = ((c.cradle_y - self._min_y) / (c.cradle_y - c.chan_y0)).clamp(0.0, 1.0)
        base = (c.w_dock * self._docked.float() + c.w_roll * roll
                + c.w_exit * self._exited.float() + c.w_catch * self._caught.float()
                + c.w_deliver * self._delivered.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="sauce_chute", robot="null"))
