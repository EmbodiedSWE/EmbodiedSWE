"""RamChuteCabinetScene — close a cabinet drawer by DROPPING A HEAVY BALL down a chute
(sim_gen task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i276`).

Derived from libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet, but
STRATEGICALLY different: the seed's whole skill is a guided hand-push on the drawer
front along its prismatic travel until the joint reads closed — the robot actuates the
judged part directly and continuously. Here the robot NEVER touches the drawer and
never sustains any actuation: the drawer's front sticks out inside a GUARD TUNNEL
(steel hood + walls) that no hand can enter, and the only way to close it is to PICK
UP the heavy steel RAM BALL from its dock tray and DROP it into the mouth of the green
CHUTE. Gravity accelerates the ball down the chute, it shoots through the tunnel,
strikes the drawer front, and its momentum drives the drawer to its rear stop — the
ball ends parked at rest against the closed drawer face. The robot's contribution is
pure transport-and-release of a passive projectile; the closing translation is powered
by gravity and delivered through a real impulsive contact chain. Success demands BOTH
terminal states: the drawer seated (front within `q_closed_tol` of the cabinet face,
riding in its channel) AND the ram ball parked against it inside the tunnel, settled.

The machine (all station-local; face plane x=0, channel centre y=0, plinth top z=0,
+x runs OUT of the cabinet under the tunnel and up the chute):
  - the drawer is a free rigid body captured in a channel (floor slab, side walls,
    rear hard stop, roof) — no joint primitives anywhere; "closed" = front face at
    q_stop = drawer_l - chan_len = +3 mm, inside the 12 mm success tolerance;
  - the guard tunnel (x in [0.012, x_base]) roofs the drawer's whole protruding
    travel: interior height 80 mm and width 134 mm — the drawer (70 tall, 126 wide)
    slides through it, the 60 mm ball passes it, a parallel-jaw hand does not;
  - the chute is a 15 deg incline whose foot meets the tunnel floor 1 mm proud (no
    upward lip); its guide walls keep the ball on axis; a backstop crowns the top;
  - everything sliding is bound to a slick material (min combine, mu ~0.08): the
    energy audit in `__post_init__` proves the ball placed at the drop station
    arrives with >= 2.5x the work needed to seat the WIDEST-open drawer, at BOTH
    ends of the sampled q0 range — the ram cannot stall by construction. A ball laid
    at the drawer face WITHOUT the descent has no kinetic energy and cannot close it
    (smoke demonstrates exactly that).

Per-episode randomization (readback-verified by smoke): station yaw FREE (+/-180 deg)
+ xy jitter, the drawer's initial opening q0, and the ball's dock position.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.30  delivery — latched: the ball has entered the chute corridor / guard tunnel
  0.55  drawer progress — latched max closure fraction of the initial opening,
        counted ONLY while the drawer genuinely rides its channel
base capped at 0.85; exactly 1.0 iff success(): drawer seated AND ball parked against
it, settled and finite, judged LIVE. Null policy ~0 (ball rests in its dock). The
seed's end state — drawer closed with no ram delivered — earns the drawer credit only
(~0.55) and can never succeed: the parked-ram clause fails.

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


def _span(stage, path: str, *, x, y, z, color, collide: Callable, orient=None, center=None):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1) — or, when `center` is
    given, an ORIENTED box: `center` (3,) + spans interpreted as sizes + quat wxyz."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    if center is None:
        xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
        if orient is not None:
            w, qx, qy, qz = (float(v) for v in orient)
            xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(qx, qy, qz)))
        xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    else:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        w, qx, qy, qz = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(qx, qy, qz)))
        xf.AddScaleOp().Set(Gf.Vec3f(float(x), float(y), float(z)))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station at `prim_path`: KINEMATIC compound. Local frame: origin at
    the cabinet FACE plane (x=0) on the channel centre (y=0), z=0 at the plinth top;
    +x runs OUT of the cabinet, under the guard tunnel and up the chute."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    Y = c.chan_hw + c.wall_t                            # channel outer half-width
    th = math.radians(c.inc_deg)
    ux, uz = math.cos(th), math.sin(th)                 # up-slope unit
    nx, nz = -math.sin(th), math.cos(th)                # slope normal
    q_inc = (math.cos(th / 2), 0.0, -math.sin(th / 2), 0.0)  # rot_y(-th): +x -> up-slope

    def slope_box(tag, s, y, h, color):
        """Oriented box in slope frame: s along slope from the base, y lateral,
        h along the outward normal (h=0 is the sliding surface)."""
        sc, hc = (s[0] + s[1]) / 2, (h[0] + h[1]) / 2
        ctr = (c.x_base + ux * sc + nx * hc, (y[0] + y[1]) / 2, c.z_base + uz * sc + nz * hc)
        _span(stage, f"{prim_path}/{tag}", x=s[1] - s[0], y=y[1] - y[0], z=h[1] - h[0],
              color=color, collide=collide, orient=q_inc, center=ctr)

    # --- plinth (raises everything to Franka-friendly heights) ----------------------
    _span(stage, f"{prim_path}/plinth", x=(-0.20, 0.72), y=(-0.14, 0.25),
          z=(-c.plinth_h, 0.0), color=c.plinth_color, collide=collide)
    # --- floor slab: channel floor + tunnel runway (one continuous surface) ---------
    _span(stage, f"{prim_path}/slab", x=(-c.chan_len - c.back_t, c.x_base),
          y=(-0.079, 0.079), z=(0.0, c.slab_t), color=c.body_color, collide=collide)
    # --- channel: side walls, rear hard stop, roof ----------------------------------
    _span(stage, f"{prim_path}/wall_p", x=(-c.chan_len - c.back_t, 0.0),
          y=(c.chan_hw, Y), z=(c.slab_t, 0.135), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/wall_n", x=(-c.chan_len - c.back_t, 0.0),
          y=(-Y, -c.chan_hw), z=(c.slab_t, 0.135), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/rear", x=(-c.chan_len - c.back_t, -c.chan_len),
          y=(-Y, Y), z=(c.slab_t, 0.135), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof", x=(-c.chan_len - c.back_t, -0.02),
          y=(-Y, Y), z=(c.tun_z1, c.tun_z1 + 0.020), color=c.body_color, collide=collide)
    # --- cabinet face: side plates + header (aperture the drawer slides through) ----
    _span(stage, f"{prim_path}/face_p", x=(0.0, c.face_t), y=(c.ap_hw, 0.150),
          z=(c.slab_t, 0.160), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/face_n", x=(0.0, c.face_t), y=(-0.150, -c.ap_hw),
          z=(c.slab_t, 0.160), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/face_hdr", x=(0.0, c.face_t), y=(-c.ap_hw, c.ap_hw),
          z=(c.tun_z1, 0.160), color=c.body_color, collide=collide)
    # --- guard tunnel over the drawer's protruding travel (hand-denial hood) --------
    _span(stage, f"{prim_path}/tun_p", x=(c.face_t, c.x_base), y=(c.ap_hw, c.ap_hw + 0.012),
          z=(c.slab_t, c.tun_z1 + 0.020), color=c.guard_color, collide=collide)
    _span(stage, f"{prim_path}/tun_n", x=(c.face_t, c.x_base), y=(-c.ap_hw - 0.012, -c.ap_hw),
          z=(c.slab_t, c.tun_z1 + 0.020), color=c.guard_color, collide=collide)
    _span(stage, f"{prim_path}/hood", x=(c.face_t, c.x_base), y=(-c.ap_hw - 0.012, c.ap_hw + 0.012),
          z=(c.tun_z1, c.tun_z1 + 0.020), color=c.guard_color, collide=collide)
    # --- chute: 15 deg incline + guide walls + top backstop -------------------------
    slope_box("inc_slab", (0.0, c.inc_len), (-0.053, 0.053), (-0.020, 0.0), c.chute_color)
    slope_box("inc_wp", (0.0, c.inc_len), (c.cor_hw, 0.053), (0.0, 0.055), c.chute_color)
    slope_box("inc_wn", (0.0, c.inc_len), (-0.053, -c.cor_hw), (0.0, 0.055), c.chute_color)
    slope_box("inc_stop", (c.inc_len, c.inc_len + 0.012), (-0.053, 0.053),
              (-0.005, 0.060), c.chute_color)
    # --- ball dock: pedestal + tray rims (the ball's start point) -------------------
    dx, dy = c.dock_x, c.dock_y
    _span(stage, f"{prim_path}/dock_col", x=(dx - 0.050, dx + 0.050), y=(dy - 0.050, dy + 0.050),
          z=(0.0, c.dock_h), color=c.dock_color, collide=collide)
    ri = c.dock_inner / 2
    for tag, xs, ys in (
        ("rim_xp", (dx + ri, dx + ri + 0.008), (dy - ri - 0.008, dy + ri + 0.008)),
        ("rim_xn", (dx - ri - 0.008, dx - ri), (dy - ri - 0.008, dy + ri + 0.008)),
        ("rim_yp", (dx - ri, dx + ri), (dy + ri, dy + ri + 0.008)),
        ("rim_yn", (dx - ri, dx + ri), (dy - ri - 0.008, dy - ri)),
    ):
        _span(stage, f"{prim_path}/dock_{tag}", x=xs, y=ys,
              z=(c.dock_h, c.dock_h + 0.020), color=c.dock_color, collide=collide)

    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at `prim_path`: DYNAMIC compound open-top box. Local frame:
    origin at the xy CENTRE with z=0 at the BOTTOM face; +x is the front."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L2, B2 = c.drawer_l / 2, c.drawer_w / 2
    t, zt = c.drawer_wall_t, c.drawer_h
    _span(stage, f"{prim_path}/floor", x=(-L2, L2), y=(-B2, B2),
          z=(0.0, c.drawer_floor_t), color=c.drawer_color, collide=collide)
    zw = (c.drawer_floor_t, zt)
    _span(stage, f"{prim_path}/w_rear", x=(-L2, -L2 + t), y=(-B2, B2),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-L2, L2), y=(-B2, -B2 + t),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-L2, L2), y=(B2 - t, B2),
          z=zw, color=c.drawer_color, collide=collide)
    # the front face — full height, what the ram ball strikes
    _span(stage, f"{prim_path}/w_front", x=(L2 - t, L2), y=(-B2, B2),
          z=zw, color=c.drawer_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.drawer_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(c.drawer_damping))
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            chan_hw: float = 0.065
            wall_t: float = 0.012
            chan_len: float = 0.157
            back_t: float = 0.012
            slab_t: float = 0.020
            face_t: float = 0.012
            ap_hw: float = 0.067
            tun_z1: float = 0.100
            x_base: float = 0.205
            z_base: float = 0.021
            inc_deg: float = 15.0
            inc_len: float = 0.40
            cor_hw: float = 0.045
            plinth_h: float = 0.300
            dock_x: float = 0.630
            dock_y: float = 0.170
            dock_h: float = 0.055
            dock_inner: float = 0.084
            body_color: tuple = (0.45, 0.38, 0.30)
            plinth_color: tuple = (0.22, 0.22, 0.24)
            guard_color: tuple = (0.30, 0.34, 0.42)
            chute_color: tuple = (0.13, 0.50, 0.22)
            dock_color: tuple = (0.15, 0.35, 0.80)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            drawer_l: float = 0.160
            drawer_w: float = 0.126
            drawer_wall_t: float = 0.008
            drawer_floor_t: float = 0.010
            drawer_h: float = 0.070
            drawer_mass: float = 0.30
            drawer_damping: float = 1.0
            drawer_color: tuple = (0.66, 0.53, 0.35)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RamChuteCabinetSceneCfg(BaseCfg):
    """Config for `RamChuteCabinetScene`. The ram contract is asserted in
    `__post_init__`: the drawer's rear hard stop IS the seated pose (inside the
    success tolerance), the ball fits every corridor it must traverse with margin, the
    guard tunnel is too low for a hand but tall enough for drawer and ball, and the
    energy audit proves the ball dropped at the drop station arrives with >= 2.5x the
    work needed to seat the widest-open drawer (both ends of the q0 range)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    q_closed_tol: float = tunable(0.012)    # drawer front face within this of the face plane
    park_gap_tol: float = tunable(0.025)    # ball surface within this of the drawer front
    settle_ball: float = tunable(0.10)      # max ball |lin vel| when judging (m/s)
    settle_drawer: float = tunable(0.05)    # max drawer |lin vel| when judging (m/s)
    lane_y_tol: float = tunable(0.020)      # drawer centred in its channel when judged

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)         # station yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)        # station xy jitter (+/- m)
    q0_range: tuple = tunable((0.060, 0.130))   # initial drawer opening (m)
    dock_jitter: float = tunable(0.008)     # ball xy jitter inside its dock tray (m)

    # --- info: station (local frame: face plane x=0, channel centre y=0, plinth top z=0) ---------
    plinth_h: float = info(0.300)           # station stands on a plinth (Franka heights)
    chan_hw: float = info(0.065)            # channel interior half-width (2 mm/side clearance)
    wall_t: float = info(0.012)
    chan_len: float = info(0.157)           # rear hard stop inner face at x=-chan_len
    back_t: float = info(0.012)
    slab_t: float = info(0.020)             # sliding surface height (slab top)
    face_t: float = info(0.012)             # cabinet face plate thickness
    ap_hw: float = info(0.067)              # aperture/tunnel interior half-width
    tun_z1: float = info(0.100)             # tunnel/aperture interior ceiling height
    x_base: float = info(0.205)             # chute foot / tunnel outer end
    z_base: float = info(0.021)             # chute surface height at its foot (1 mm proud)
    inc_deg: float = info(15.0)             # chute incline angle
    inc_len: float = info(0.40)             # chute slope length
    cor_hw: float = info(0.045)             # chute corridor interior half-width
    drop_s: float = info(0.35)              # drop station: distance up the slope
    dock_x: float = info(0.630)             # ball dock tray centre
    dock_y: float = info(0.170)
    dock_h: float = info(0.055)             # dock tray floor height
    dock_inner: float = info(0.084)         # dock tray interior width
    # --- info: drawer ----------------------------------------------------------------------------
    drawer_l: float = info(0.160)
    drawer_w: float = info(0.126)
    drawer_wall_t: float = info(0.008)
    drawer_floor_t: float = info(0.010)
    drawer_h: float = info(0.070)
    drawer_mass: float = info(0.30)
    drawer_damping: float = info(1.0)
    # --- info: ball ------------------------------------------------------------------------------
    ball_r: float = info(0.030)
    ball_mass: float = info(1.40)           # a heavy steel ram ball (graspable: 60 mm dia)
    # --- info: materials -------------------------------------------------------------------------
    slide_mu_s: float = info(0.10)          # slick everywhere sliding (min combine)
    slide_mu_d: float = info(0.08)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.30 + 0.55 = 0.85 = the non-success cap) -------------------------
    w_deliver: float = info(0.30)
    w_drawer: float = info(0.55)

    # Derived (filled in __post_init__).
    q_stop: float = field(default=None, init=False)   # drawer front x pressed on the rear stop

    def chute_surface_z(self, x: float) -> float:
        """Chute sliding-surface height at local x (only meaningful x >= x_base)."""
        return self.z_base + (x - self.x_base) * math.tan(math.radians(self.inc_deg))

    def drop_point(self) -> tuple:
        """Station-local ball-centre release point above the drop station."""
        th = math.radians(self.inc_deg)
        s, h = self.drop_s, self.ball_r + 0.012
        return (self.x_base + s * math.cos(th) - h * math.sin(th), 0.0,
                self.z_base + s * math.sin(th) + h * math.cos(th))

    def __post_init__(self) -> None:
        c = self
        g = 9.81
        th = math.radians(c.inc_deg)
        # the drawer's rear hard stop IS the seated pose, inside the tolerance
        self.q_stop = c.drawer_l - c.chan_len
        assert 0.0 < self.q_stop < c.q_closed_tol - 0.005, \
            "rear stop must seat the drawer inside q_closed_tol with margin"
        # channel: 2 mm/side on the drawer; aperture/tunnel: 4 mm/side, 10 mm above
        assert abs(c.chan_hw * 2 - c.drawer_w - 0.004) < 1e-9
        assert c.ap_hw * 2 > c.drawer_w + 0.006, "drawer must pass the aperture/tunnel"
        assert c.tun_z1 > c.slab_t + c.drawer_h + 0.008, "drawer must pass under the hood"
        # ball fits the tunnel (>= 8 mm/side, >= 15 mm above) and the chute corridor
        assert c.ap_hw * 2 > 2 * c.ball_r + 0.016, "ball must fit the tunnel width"
        assert c.tun_z1 > c.slab_t + 2 * c.ball_r + 0.015, "ball must fit the tunnel height"
        assert c.cor_hw * 2 > 2 * c.ball_r + 0.020, "ball must fit the chute corridor"
        # ball strikes the drawer FACE: contact height inside the face band
        zc = c.slab_t + c.ball_r
        assert c.slab_t + 0.010 < zc < c.slab_t + c.drawer_h - 0.010, \
            "ball centre must meet the drawer front face"
        # the widest-open drawer leaves free flat runway before the chute foot
        assert c.q0_range[1] + c.ball_r + 0.030 < c.x_base, \
            "widest drawer must leave the chute foot clear"
        # a parked ball sits fully inside the tunnel
        assert self.q_stop + c.ball_r + c.park_gap_tol + c.ball_r < c.x_base
        # the dock is clear of the chute corridor and holds the jittered ball
        assert c.dock_y - c.dock_inner / 2 - 0.010 > 0.053, "dock must clear the chute"
        assert c.dock_inner / 2 - c.ball_r > c.dock_jitter + 0.004, \
            "jittered ball must land inside the dock tray"
        # chute foot is 1 mm proud of the runway (a drop-off, never an upward lip)
        assert 0.0 < c.z_base - c.slab_t <= 0.002
        # ENERGY AUDIT: ball released at (drop_s - 0.05) up the slope must deliver
        # >= 2.5x the work needed to seat the drawer, at BOTH q0 extremes.
        mu = c.slide_mu_d
        a_inc = g * (math.sin(th) - mu * math.cos(th))
        assert a_inc > 1.0, "chute must accelerate the ball briskly"
        v1sq = 2.0 * a_inc * (c.drop_s - 0.05)
        mb, md = c.ball_mass, c.drawer_mass
        for q0 in c.q0_range:
            flat = c.x_base - (q0 + c.ball_r)          # free runway before impact
            v2sq = v1sq - 2.0 * mu * g * flat
            assert v2sq > 0.2, "ball must reach the drawer briskly"
            v3 = math.sqrt(v2sq) * mb / (mb + md)      # inelastic momentum transfer
            energy = 0.5 * (mb + md) * v3 * v3
            travel = q0 - self.q_stop
            work = (mb + md) * mu * g * travel + md * c.drawer_damping * 0.6 * travel
            assert energy > 2.5 * work, \
                f"ram energy margin too thin at q0={q0} ({energy:.3f} vs {work:.3f} J)"
        assert abs(c.w_deliver + c.w_drawer - 0.85) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    """(N,) yaw angle of quats (wxyz)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ram_chute_cabinet")
class RamChuteCabinetScene(BaseScene):
    cfg: RamChuteCabinetSceneCfg

    def __init__(self, cfg: RamChuteCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or RamChuteCabinetSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            chan_hw=c.chan_hw, wall_t=c.wall_t, chan_len=c.chan_len, back_t=c.back_t,
            slab_t=c.slab_t, face_t=c.face_t, ap_hw=c.ap_hw, tun_z1=c.tun_z1,
            x_base=c.x_base, z_base=c.z_base, inc_deg=c.inc_deg, inc_len=c.inc_len,
            cor_hw=c.cor_hw, plinth_h=c.plinth_h, dock_x=c.dock_x, dock_y=c.dock_y,
            dock_h=c.dock_h, dock_inner=c.dock_inner, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        drawer_spawn = cls["drawer"](
            drawer_l=c.drawer_l, drawer_w=c.drawer_w, drawer_wall_t=c.drawer_wall_t,
            drawer_floor_t=c.drawer_floor_t, drawer_h=c.drawer_h,
            drawer_mass=c.drawer_mass, drawer_damping=c.drawer_damping,
            contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.plinth_h))),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.0, 0.05))),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,   # kills GPU phantom creep
                        max_depenetration_velocity=0.5,
                        linear_damping=0.02, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.30, dynamic_friction=0.25, restitution=0.0,
                        friction_combine_mode="average"),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.38, 0.38, 0.44), metallic=0.8)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.8, 0.05))),
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
        self.station: RigidObject = env.iscene["station"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.q0 = torch.zeros(n, device=dev)          # initial drawer opening
        # latches (partial credit survives transients; success is judged live)
        self._fdeliver = torch.zeros(n, device=dev)
        self._fdrawer = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (free yaw + xy jitter), sample the drawer
        opening q0, seat the drawer in its channel, rest the ball in its dock tray."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_st = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.plinth_h
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_st
        self.station.write_root_state_to_sim(st, env_ids)

        # sampled knob: drawer opening
        q0 = c.q0_range[0] + torch.rand(m, device=dev) * (c.q0_range[1] - c.q0_range[0])
        self.q0[env_ids] = q0

        # drawer: in its channel at opening q0 (origin = centre, so x = q0 - L/2)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = q0 - c.drawer_l / 2
        loc[:, 2] = c.slab_t + 0.002
        s = torch.zeros(m, 13, device=dev)
        s[:, 0:3] = dp + origin + quat_apply(q_st, loc)
        s[:, 3:7] = q_st
        self.drawer.write_root_state_to_sim(s, env_ids)

        # ball: resting in the dock tray with xy jitter
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.dock_x + (torch.rand(m, device=dev) * 2 - 1) * c.dock_jitter
        loc[:, 1] = c.dock_y + (torch.rand(m, device=dev) * 2 - 1) * c.dock_jitter
        loc[:, 2] = c.dock_h + c.ball_r + 0.003
        s = torch.zeros(m, 13, device=dev)
        s[:, 0:3] = dp + origin + quat_apply(q_st, loc)
        s[:, 3] = 1.0
        self.ball.write_root_state_to_sim(s, env_ids)

        self._fdeliver[env_ids] = 0.0
        self._fdrawer[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "q0": self.q0[env_ids].clone(),
            "fdeliver": self._fdeliver[env_ids].clone(),
            "fdrawer": self._fdrawer[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.q0[env_ids] = state["q0"]
        self._fdeliver[env_ids] = state["fdeliver"]
        self._fdrawer[env_ids] = state["fdrawer"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden CABINET stands on a dark plinth. Its single drawer (light wood, "
            f"open-topped, {c.drawer_w * 1000:.0f} mm wide) sticks OUT of the cabinet "
            f"face by roughly {c.q0_range[0] * 100:.0f}-{c.q0_range[1] * 100:.0f} cm — "
            f"the opening varies by episode, and the whole station's position and "
            f"heading also vary. The drawer's protruding front is covered by a "
            f"steel-blue GUARD TUNNEL (hood and side walls): no hand or tool fits "
            f"inside, so the drawer cannot be pushed directly. The tunnel's outer end "
            f"meets the foot of a GREEN CHUTE, an inclined channel that climbs away "
            f"from the cabinet. Beside the chute's upper mouth, a heavy STEEL BALL "
            f"({2 * c.ball_r * 1000:.0f} mm across, {c.ball_mass:.1f} kg) rests in a "
            f"small BLUE DOCK TRAY.\n"
            f"The machine is a GRAVITY RAM: a ball released into the chute's upper "
            f"mouth accelerates down the incline, shoots through the guard tunnel, "
            f"strikes the drawer's front face and drives the drawer shut with its "
            f"momentum, ending up resting against the closed drawer.\n"
            f"Goal: pick the steel ball out of its dock tray and drop it into the "
            f"green chute's upper mouth so the ram closes the drawer. Success is the "
            f"settled end state: the drawer front within "
            f"{c.q_closed_tol * 1000:.0f} mm of the cabinet face AND the ball at rest "
            f"against the closed drawer inside the tunnel. Both are required — a "
            f"closed drawer with no ram parked against it does NOT succeed, and a "
            f"ball merely laid at the drawer face without the descent has no energy "
            f"to close it. A ball placed too low on the chute may arrive too slowly: "
            f"release it at the upper mouth."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick the heavy steel ball out of the blue dock tray and drop it into "
            "the green chute's upper mouth; it must run down through the guard "
            "tunnel and ram the protruding drawer shut. Finish with the drawer "
            "front flush with the cabinet face and the ball resting against it."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def drawer_q(self) -> torch.Tensor:
        """(N,) drawer opening: the front face's x in the station frame
        (0 = flush with the cabinet face plane)."""
        return self._station_local(self.drawer.data.root_pos_w)[:, 0] \
            + self.cfg.drawer_l / 2

    def drawer_in_channel(self) -> torch.Tensor:
        """(N,) bool: drawer genuinely riding in its channel (guards every drawer
        clause and the drawer-progress latch against a drawer stolen out of the
        cabinet)."""
        c = self.cfg
        loc = self._station_local(self.drawer.data.root_pos_w)
        return (loc[:, 1].abs() < c.lane_y_tol) \
            & ((loc[:, 2] - c.slab_t).abs() < 0.030) \
            & (loc[:, 0] + c.drawer_l / 2 > -0.020) \
            & (loc[:, 0] + c.drawer_l / 2 < 0.180)

    def drawer_closed(self) -> torch.Tensor:
        """(N,) bool: drawer seated — front face within q_closed_tol of the cabinet
        face plane, riding in its channel."""
        return (self.drawer_q() < self.cfg.q_closed_tol) & self.drawer_in_channel()

    def ball_delivered_now(self) -> torch.Tensor:
        """(N,) bool: the ball is in the delivery corridor — on (or just above) the
        chute's sliding surface, or inside the guard tunnel."""
        c = self.cfg
        loc = self._station_local(self.ball.data.root_pos_w)
        x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
        surf = c.z_base + (x - c.x_base).clamp(min=0.0) * math.tan(math.radians(c.inc_deg))
        on_chute = (x > c.x_base - 0.005) & (x < c.x_base + c.inc_len + 0.01) \
            & (y.abs() < c.cor_hw + 0.011) & ((z - surf) > 0.015) & ((z - surf) < 0.120)
        in_tunnel = (x > -0.010) & (x < c.x_base) & (y.abs() < 0.060) \
            & (z > 0.030) & (z < 0.096)
        return on_chute | in_tunnel

    def ball_parked(self) -> torch.Tensor:
        """(N,) bool: the ram ball at rest against the drawer front inside the
        tunnel — its surface within park_gap_tol of the front face, on the runway
        floor, on the channel axis."""
        c = self.cfg
        loc = self._station_local(self.ball.data.root_pos_w)
        gap = loc[:, 0] - self.drawer_q()              # ball centre ahead of the face
        return (gap > -0.006) & (gap < c.ball_r + c.park_gap_tol) \
            & (loc[:, 1].abs() < 0.050) \
            & ((loc[:, 2] - c.slab_t - c.ball_r).abs() < 0.020)

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball and drawer translation at rest (a residual spin of the
        parked ball in place is harmless and not judged), drawer not tumbling."""
        c = self.cfg
        return (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_ball) \
            & (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_drawer) \
            & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < 0.5)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.ball.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        de = self.ball_delivered_now() & fin
        self._fdeliver = torch.maximum(self._fdeliver, de.float())
        q = self.drawer_q()
        span = (self.q0 - self.cfg.q_stop).clamp(min=1e-6)
        fd = ((self.q0 - q) / span).clamp(0.0, 1.0)
        fd = torch.where(self.drawer_in_channel() & fin, fd, torch.zeros_like(fd))
        self._fdrawer = torch.maximum(self._fdrawer, fd)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: drawer seated in its channel AND the ram ball parked at rest
        against it — live physical outcomes — settled and finite. The rubric neither
        knows nor cares what moved what: a seated drawer with the ram resting against
        its face is the terminal state the gravity ram produces."""
        self._update_latches()
        return self.drawer_closed() & self.ball_parked() & self.settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 * latched ball-delivery + 0.55 * latched
        drawer-closure fraction (channel-guarded), capped at 0.85; exactly 1.0 iff
        success() holds live. Doing nothing scores ~0. The seed's end state — a
        closed drawer with no ram delivered — earns the drawer credit only (~0.55):
        the parked-ram clause never fires."""
        c = self.cfg
        self._update_latches()
        base = (c.w_deliver * self._fdeliver + c.w_drawer * self._fdrawer).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ram_chute_cabinet", robot="null"))
