"""WedgeJackScene — jack an orange crate up to the delivery level with a wedge ram
(sim_gen task `two_robot_pick_cube_i315`).

Derived from maniskill/two_robot_pick_cube, but STRATEGICALLY different: the seed's
whole skill is a two-agent free-space relay — arm A picks a cube, lifts it to a
midpoint, arm B takes the handover and HOLDS it at a floating elevated goal pose.
Here there is ONE implied arm and NOTHING is ever held at the goal: the elevation is
produced and SUSTAINED by a force-amplifying machine the solver must operate. A green
elevator platform rides a vertical guide inside a gantry; a blue wedge ram slides on a
horizontal rail underneath it. Driving the ram inward (a long, regulated push on its
tall end plate) feeds a 15 deg inclined plane under the platform's matching inclined
foot and jacks the platform up; friction self-locks the wedge (mu = 0.6 >> tan 15 deg,
margin 2.2x), so the raised state persists with hands off. The goal: the crate seated
on the platform deck AND the deck jacked up to the yellow beam's delivery level AND
the platform actually SUPPORTED BY THE RAM (a held/hoisted platform never counts),
everything settled. No execution order is required — load-then-jack and
jack-then-load both work (the deck stays reachable from above at every height).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): station (heavy DYNAMIC compound — a kinematic root would orphan the
prismatic joint anchors when reset teleports it: slab, two gantry pillars, yellow
delivery beam), car (DYNAMIC compound on an authored prismatic-Z joint: deck plate,
low perimeter rim, centre column, 15-deg inclined foot), wedge ram (DYNAMIC compound
on an authored prismatic-X joint: base bar, 15-deg inclined top slab, tall end
plate), one orange crate cube. Wedge and foot inclines are the SAME angle, so the
load-bearing contact is flush face-on-face (no edge-contact limit cycles). The ram
carries a post_step force plant (viscous damping + external `ram_drive` clamp) on a
finite-difference insertion rate — root velocities are phantom under external
wrenches on this stack.

Per-episode randomization (readback-verified by smoke): station yaw FREE (+/-180 deg)
+ xy jitter, ram start withdrawal u in [0, 40 mm] (outward only — the reset gap stays
positive by construction), crate side (+y / -y), crate slot x and free yaw. All
discrete draws derive from torch.rand after a burn draw (the first post-seed draw is
degenerate on this stack).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.30  aboard   — the crate ever seated on the deck, inside the rim (latched)
  0.45  lift     — latched max lift fraction toward the delivery level, gated ABOARD
                   AND SUPPORTED (jacking the empty platform earns nothing, and so
                   does hoisting the loaded platform by hand)
capped at 0.75; exactly 1.0 iff success(): crate seated on the deck, deck at/above
the delivery level, platform height consistent with the ram insertion (SUPPORTED —
holding the car up scores nothing), everything settled and finite. Null policy ~0.
The seed's strategy — hold the cargo at an elevated pose — earns at most the aboard
latch: altitude without the jacked platform is refused by the supported clause, and
a crate resting anywhere off the deck is refused by the aboard clause.

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


def _pbox(stage, path: str, *, center, size, pitch: float, color, collide: Callable):
    """Pitch-rotated box child (rotation about +y by `pitch` rad; pitch < 0 raises +x)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    h = pitch / 2
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, math.sin(h), 0.0)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _author_mass(root, mass: float, com, inertia) -> None:
    """Explicit MassAPI mass + CoM + diagonal inertia. On this stack, MassAPI mass on a
    compound root leaves the CoM at the body ORIGIN and the shape-derived inertia is
    unknown — author all three so plant stability (K*dt/m) is auditable."""
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _rigid_armor(root, *, mass, com, inertia, lin_damp=0.0, ang_damp=0.0):
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, mass, com, inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    return prb


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the station at `prim_path`: heavy DYNAMIC compound. Local frame: origin
    at the elevator axis on the slab TOP (z=0); the ram rail runs along +x (the open
    approach side); the gantry pillars + yellow delivery beam stand on the -x side."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    _span(stage, f"{prim_path}/slab", x=c.slab_x, y=(-c.slab_y, c.slab_y),
          z=(-c.slab_t, 0.0), color=c.slab_color, collide=collide)
    for tag, sgn in (("p", 1.0), ("n", -1.0)):
        y0, y1 = sorted((sgn * c.pillar_y0, sgn * c.pillar_y1))
        _span(stage, f"{prim_path}/pillar_{tag}", x=(c.pillar_x0, c.pillar_x1),
              y=(y0, y1), z=(0.0, c.pillar_h), color=c.pillar_color, collide=collide)
    _span(stage, f"{prim_path}/beam", x=(c.pillar_x0, c.pillar_x1),
          y=(-c.pillar_y0, c.pillar_y0), z=(c.beam_z0, c.beam_z1),
          color=c.beam_color, collide=collide)

    _rigid_armor(root, mass=c.station_mass, com=(0.1, 0.0, -0.02),
                 inertia=c.station_inertia, lin_damp=0.5, ang_damp=2.0)
    base = _mk_material(prim_path, "base", c.base_mu_s, c.base_mu_d, "average")
    bind_physics_material(prim_path, base)
    return root


def _spawn_car(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the elevator car at `prim_path`: DYNAMIC compound on a prismatic-Z
    joint. Local frame: origin at the CENTRE OF THE DECK TOP FACE. Deck plate below
    z=0, low perimeter rim above it, centre column down to the 15-deg inclined foot
    (the flush counter-face the wedge slides under)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    d = c.deck_half

    _span(stage, f"{prim_path}/deck", x=(-d, d), y=(-d, d), z=(-c.deck_t, 0.0),
          color=c.deck_color, collide=collide)
    t, h = c.rim_t, c.rim_h
    _span(stage, f"{prim_path}/rim_xp", x=(d - t, d), y=(-d, d), z=(0.0, h),
          color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/rim_xn", x=(-d, -d + t), y=(-d, d), z=(0.0, h),
          color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/rim_yp", x=(-d + t, d - t), y=(d - t, d), z=(0.0, h),
          color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/rim_yn", x=(-d + t, d - t), y=(-d, -d + t), z=(0.0, h),
          color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/column", x=(-c.col_half, c.col_half),
          y=(-c.col_half, c.col_half), z=(c.col_z0, -c.deck_t),
          color=c.rim_color, collide=collide)
    _pbox(stage, f"{prim_path}/foot", center=(0.0, 0.0, c.foot_cz),
          size=(c.foot_len, c.foot_w, c.foot_t), pitch=-c.theta,
          color=c.rim_color, collide=collide)

    _rigid_armor(root, mass=c.car_mass, com=(0.0, 0.0, -0.04), inertia=c.car_inertia)
    mat = _mk_material(prim_path, "carmat", c.jack_mu_s, c.jack_mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawn_wedge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the wedge ram at `prim_path`: DYNAMIC compound on a prismatic-X joint.
    Local frame: origin at the BASE BAR BOTTOM CENTRE (hovers `wedge_hover` above the
    slab — the joint carries it; the only load-bearing contact is incline-on-foot).
    Base bar, 15-deg inclined top slab (rises toward +x = outward), tall end plate on
    the +x end (the push handle)."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    _span(stage, f"{prim_path}/base", x=(-c.wedge_base_half, c.wedge_base_half),
          y=(-c.wedge_w / 2, c.wedge_w / 2), z=(0.0, c.wedge_base_t),
          color=c.wedge_color, collide=collide)
    _pbox(stage, f"{prim_path}/incline", center=(c.incline_cx, 0.0, c.incline_cz),
          size=(c.incline_len, c.wedge_w, c.incline_t), pitch=-c.theta,
          color=c.wedge_color, collide=collide)
    _span(stage, f"{prim_path}/handle",
          x=(c.wedge_base_half, c.wedge_base_half + c.handle_t),
          y=(-c.wedge_w / 2, c.wedge_w / 2), z=(0.004, c.handle_h),
          color=c.handle_color, collide=collide)

    _rigid_armor(root, mass=c.wedge_mass, com=(0.0, 0.0, 0.03), inertia=c.wedge_inertia)
    mat = _mk_material(prim_path, "wedgemat", c.jack_mu_s, c.jack_mu_d, "average")
    bind_physics_material(prim_path, mat)
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
            slab_x: tuple = (-0.33, 0.62)
            slab_y: float = 0.21
            slab_t: float = 0.04
            pillar_x0: float = -0.125
            pillar_x1: float = -0.095
            pillar_y0: float = 0.095
            pillar_y1: float = 0.125
            pillar_h: float = 0.30
            beam_z0: float = 0.170
            beam_z1: float = 0.190
            station_mass: float = 40.0
            station_inertia: tuple = (2.0, 2.0, 3.0)
            slab_color: tuple = (0.45, 0.44, 0.42)
            pillar_color: tuple = (0.22, 0.24, 0.28)
            beam_color: tuple = (0.92, 0.80, 0.10)
            contact_offset: float = 0.0015
            base_mu_s: float = 0.50
            base_mu_d: float = 0.45

        @configclass
        class CarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_car)
            theta: float = math.radians(15.0)
            deck_half: float = 0.075
            deck_t: float = 0.02
            rim_t: float = 0.008
            rim_h: float = 0.012
            col_half: float = 0.02
            col_z0: float = -0.060
            foot_cz: float = -0.0647
            foot_len: float = 0.104
            foot_w: float = 0.10
            foot_t: float = 0.02
            car_mass: float = 0.6
            car_inertia: tuple = (0.003, 0.003, 0.003)
            deck_color: tuple = (0.15, 0.60, 0.25)
            rim_color: tuple = (0.10, 0.40, 0.17)
            contact_offset: float = 0.0015
            jack_mu_s: float = 0.60
            jack_mu_d: float = 0.55

        @configclass
        class WedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wedge)
            theta: float = math.radians(15.0)
            wedge_base_half: float = 0.21
            wedge_base_t: float = 0.024
            wedge_w: float = 0.10
            incline_cx: float = 0.01
            incline_cz: float = 0.0586
            incline_len: float = 0.352
            incline_t: float = 0.02
            handle_t: float = 0.02
            handle_h: float = 0.10
            wedge_mass: float = 1.2
            wedge_inertia: tuple = (0.002, 0.02, 0.02)
            wedge_color: tuple = (0.15, 0.35, 0.85)
            handle_color: tuple = (0.75, 0.82, 0.95)
            contact_offset: float = 0.0015
            jack_mu_s: float = 0.60
            jack_mu_d: float = 0.55

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["car"] = CarSpawnerCfg
        _SPAWNER_CACHE["wedge"] = WedgeSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WedgeJackSceneCfg(BaseCfg):
    """Config for `WedgeJackScene`. The jack contract is asserted in `__post_init__`:
    the reset gap is positive at every sampled ram start (randomization is outward
    only), the hard-stop lift clears the delivery requirement, the incline covers the
    foot over the whole engagement range (contact stays flush face-on-face), the
    incline never digs into the slab, friction self-locks the wedge with >= 2x margin
    over tan(theta), a crate shoved under the foot cannot prop the car anywhere near
    the delivery level, and the drive plant is discretely stable."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    q_req: float = tunable(0.050)          # required lift of the deck (m); beam_z0 = z_lo + q_req
    aboard_xy_tol: float = tunable(0.045)  # crate centre box in the car frame (m)
    aboard_z_tol: float = tunable(0.014)   # crate resting-height tolerance on the deck (m)
    support_tol: float = tunable(0.006)    # max q above what the ram insertion provides (m)
    settle_lin: float = tunable(0.05)      # max crate |lin vel| when judging (m/s)
    settle_rate: float = tunable(0.02)     # max |FD rate| of ram insertion AND car lift (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # station yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # station xy jitter (+/- m)
    wedge_pullout: float = tunable(0.040)  # ram start withdrawal u ~ U[0, this] (outward only)
    crate_x0: float = tunable(0.24)        # crate slot band on the apron (station x)
    crate_x1: float = tunable(0.38)
    crate_y: float = tunable(0.14)         # crate lane |y| (side sampled per episode)

    # --- tunable: ram drive plant -----------------------------------------------------------------
    # Discrete-stability audit (explicit external forces at 120 Hz, authored m=1.2):
    # b*dt/m = 6.0/(120*1.2) ~ 0.042 << 1; F_max/m*dt = 0.17 m/s per step worst case;
    # post_step wrenches act one substep late, so solver gains keep K*dt/m well under 1.
    f_max: float = tunable(25.0)           # |ram_drive| clamp (N)
    ram_b: float = tunable(6.0)            # viscous plant damping (N*s/m)
    rate_clamp: float = tunable(5.0)       # FD rate clamp (m/s) — teleport transients

    # --- info: station (local frame: origin at the elevator axis on the slab top) ----------------
    base_z: float = info(0.041)            # root height: slab bottom 1 mm above ground, settles
    slab_x: tuple = info((-0.33, 0.62))
    slab_y: float = info(0.21)
    slab_t: float = info(0.04)
    pillar_x0: float = info(-0.125)
    pillar_x1: float = info(-0.095)
    pillar_y0: float = info(0.095)
    pillar_y1: float = info(0.125)
    pillar_h: float = info(0.30)
    beam_z0: float = info(0.170)           # delivery level: deck top must reach this
    beam_z1: float = info(0.190)
    station_mass: float = info(40.0)       # heavy DYNAMIC fixture: joint anchors must
    station_inertia: tuple = info((2.0, 2.0, 3.0))  # follow reset teleports
    # --- info: elevator car (origin at deck top centre) -------------------------------------------
    theta_deg: float = info(15.0)          # jack angle (wedge incline == foot incline)
    deck_z_lo: float = info(0.12)          # deck top height at car-down (q = 0)
    deck_half: float = info(0.075)
    deck_t: float = info(0.02)
    rim_t: float = info(0.008)
    rim_h: float = info(0.012)
    col_half: float = info(0.02)
    col_z0: float = info(-0.060)
    foot_cz: float = info(-0.0647)
    foot_len: float = info(0.104)          # along the incline; horizontal half ~ 50 mm
    foot_w: float = info(0.10)
    foot_t: float = info(0.02)
    car_mass: float = info(0.6)
    car_inertia: tuple = info((0.003, 0.003, 0.003))
    q_lim: tuple = info((0.0, 0.070))      # car prismatic-Z joint limits (rests at 0)
    # --- info: wedge ram (origin at base bar bottom centre) ---------------------------------------
    wedge_hover: float = info(0.004)       # base bar hovers above the slab (joint-carried)
    wedge_base_half: float = info(0.21)
    wedge_base_t: float = info(0.024)
    wedge_w: float = info(0.10)
    incline_cx: float = info(0.01)
    incline_cz: float = info(0.0586)
    incline_len: float = info(0.352)
    incline_t: float = info(0.02)
    handle_t: float = info(0.02)
    handle_h: float = info(0.10)
    wedge_mass: float = info(1.2)
    wedge_inertia: tuple = info((0.002, 0.02, 0.02))
    xw_anchor: float = info(0.10)          # ram origin station-x at joint zero (spawn pose)
    xw_lim: tuple = info((-0.112, 0.145))  # ram origin station-x travel (joint limits)
    xw0_base: float = info(0.101)          # innermost reset position (withdrawal adds to it)
    # --- info: crate / materials -------------------------------------------------------------------
    crate_size: float = info(0.05)
    crate_mass: float = info(0.12)
    crate_color: tuple = info((0.90, 0.45, 0.10))
    contact_offset: float = info(0.0015)
    jack_mu_s: float = info(0.60)          # incline/foot/deck/crate faces (pair avg 0.60)
    jack_mu_d: float = info(0.55)
    base_mu_s: float = info(0.50)
    base_mu_d: float = info(0.45)
    # --- info: rubric weights (0.30 + 0.45 = 0.75 = the non-success cap) --------------------------
    w_aboard: float = info(0.30)
    w_lift: float = info(0.45)
    score_cap: float = info(0.75)

    def __post_init__(self) -> None:
        c = self
        th = math.radians(c.theta_deg)
        self.theta = th
        tan_t, cos_t, sin_t = math.tan(th), math.cos(th), math.sin(th)
        # contact-plane constants: wedge top plane (wedge local) z = zeta + tan*x;
        # car foot bottom plane (car local)  z = foot_c + tan*x
        zeta = c.incline_cz + c.incline_t / (2 * cos_t) - tan_t * c.incline_cx
        foot_c = c.foot_cz - c.foot_t / (2 * cos_t)
        self.q_c0 = (c.wedge_hover + zeta) - (c.deck_z_lo + foot_c)
        # q_at(xw): car lift when riding the ram at station-x = xw
        q_at = lambda xw: self.q_c0 - tan_t * xw  # noqa: E731
        self.q_at_coeff = (self.q_c0, tan_t)
        # 1. hard-stop lift clears the requirement
        assert q_at(c.xw_lim[0]) >= c.q_req + 0.004, \
            f"hard-stop lift {q_at(c.xw_lim[0]):.4f} must clear q_req {c.q_req}"
        assert c.q_lim[1] > q_at(c.xw_lim[0]) + 0.008, "car upper limit never engaged"
        # 2. reset gap positive at EVERY sampled start (withdrawal is outward only)
        g_in = -q_at(c.xw0_base)                     # gap at the innermost spawn
        g_out = -q_at(c.xw0_base + c.wedge_pullout)  # gap at the outermost spawn
        assert 0.0015 <= g_in and g_out <= 0.020, \
            f"reset gap band [{g_in:.4f}, {g_out:.4f}] must stay positive and small"
        assert c.xw0_base + c.wedge_pullout <= c.xw_lim[1] - 0.003, "spawn inside limits"
        # 3. incline covers the foot over the whole ENGAGEMENT range (flush contact).
        # Contact only exists for xw <= xw_e = q_c0/tan (outside it there is a gap),
        # so coverage is checked at the engagement point and at the hard stop.
        inc_lo = c.incline_cx - (c.incline_len / 2) * cos_t
        inc_hi = c.incline_cx + (c.incline_len / 2) * cos_t
        foot_half = (c.foot_len / 2) * cos_t
        xw_e = self.q_c0 / tan_t
        assert c.xw0_base + c.wedge_pullout > xw_e, "spawn band starts outside engagement"
        for xw in (c.xw_lim[0], xw_e):
            assert inc_lo <= -xw - foot_half and -xw + foot_half <= inc_hi, \
                f"incline must cover the foot at xw={xw}"
        # 4. incline never digs into the slab (wedge-local lowest corner above 0), and
        # the foot's lowest sweep point clears the base bar top at car-down
        low = c.incline_cz - (c.incline_len / 2) * sin_t - (c.incline_t / 2) * cos_t
        assert low >= 0.0015, f"incline lowest corner {low:.4f} must clear the base plane"
        foot_bot_min = c.deck_z_lo + foot_c - tan_t * foot_half
        assert foot_bot_min >= c.wedge_hover + c.wedge_base_t + 0.002, \
            "foot low corner must clear the base bar top at car-down"
        # 5. self-locking friction margin
        mu_pair = c.jack_mu_s  # both faces authored identically; average combine
        assert mu_pair >= 2.0 * tan_t, "wedge must self-lock with >= 2x margin"
        # 6. a crate under the foot cannot prop the car near the delivery level
        foot_bot_min = c.deck_z_lo + foot_c - tan_t * foot_half
        assert c.crate_size - foot_bot_min <= c.q_req - 0.020, \
            "crate-prop lift must fall far short of the delivery level"
        # 7. handle stays clear of the deck at max insertion; nose stays over the slab
        assert c.xw_lim[0] + c.wedge_base_half >= c.deck_half + 0.02, "handle clears deck"
        assert c.xw_lim[0] - c.wedge_base_half >= c.slab_x[0] + 0.005, "nose over slab"
        # 8. beam marks the delivery level exactly; beam/pillars clear the moving parts
        assert abs(c.beam_z0 - (c.deck_z_lo + c.q_req)) < 1e-9, "beam bottom = delivery level"
        assert c.pillar_x1 <= -c.deck_half - 0.015, "gantry clear of the deck sweep"
        assert c.pillar_y0 >= c.wedge_w / 2 + 0.015, "ram passes between the pillars"
        # 9. crate lane clear of the ram sweep and on the slab
        diag = c.crate_size * math.sqrt(2) / 2
        assert c.crate_y - diag > c.wedge_w / 2 + 0.010, "crate lane clear of the ram"
        assert c.crate_y + diag < c.slab_y - 0.005, "crate lane on the slab"
        assert c.crate_x1 + diag < c.slab_x[1] - 0.02, "crate band on the slab"
        # 10. rim interior accepts the crate; aboard box honest by construction
        interior = c.deck_half - c.rim_t
        assert interior >= c.crate_size / 2 + 0.015, "rim interior accepts the crate"
        assert c.aboard_xy_tol <= interior - c.crate_size / 2 + 0.004, \
            "aboard box must not admit a crate resting on the rim edge"
        # 11. plant discrete stability
        dt = 1.0 / 120.0
        assert c.ram_b * dt / c.wedge_mass < 0.5, "plant damping discretely stable"
        assert c.f_max / c.wedge_mass * dt < 0.25, "per-step force kick bounded"
        # 12. insertion force budget: F ~ W*(tan+mu_d)/(1-mu_d*tan) << f_max
        w_tot = (c.car_mass + c.crate_mass) * 9.81
        f_need = w_tot * (tan_t + c.jack_mu_d) / (1 - c.jack_mu_d * tan_t)
        assert f_need < 0.5 * c.f_max, f"insertion force {f_need:.1f} N within budget"
        assert abs(c.w_aboard + c.w_lift - c.score_cap) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
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


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("wedge_jack")
class WedgeJackScene(BaseScene):
    cfg: WedgeJackSceneCfg

    def __init__(self, cfg: WedgeJackSceneCfg | None = None) -> None:
        super().__init__(cfg or WedgeJackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            slab_x=c.slab_x, slab_y=c.slab_y, slab_t=c.slab_t,
            pillar_x0=c.pillar_x0, pillar_x1=c.pillar_x1,
            pillar_y0=c.pillar_y0, pillar_y1=c.pillar_y1, pillar_h=c.pillar_h,
            beam_z0=c.beam_z0, beam_z1=c.beam_z1,
            station_mass=c.station_mass, station_inertia=c.station_inertia,
            contact_offset=c.contact_offset, base_mu_s=c.base_mu_s, base_mu_d=c.base_mu_d)
        car_spawn = cls["car"](
            theta=self.cfg.theta, deck_half=c.deck_half, deck_t=c.deck_t,
            rim_t=c.rim_t, rim_h=c.rim_h, col_half=c.col_half, col_z0=c.col_z0,
            foot_cz=c.foot_cz, foot_len=c.foot_len, foot_w=c.foot_w, foot_t=c.foot_t,
            car_mass=c.car_mass, car_inertia=c.car_inertia,
            contact_offset=c.contact_offset, jack_mu_s=c.jack_mu_s, jack_mu_d=c.jack_mu_d)
        wedge_spawn = cls["wedge"](
            theta=self.cfg.theta, wedge_base_half=c.wedge_base_half,
            wedge_base_t=c.wedge_base_t, wedge_w=c.wedge_w,
            incline_cx=c.incline_cx, incline_cz=c.incline_cz,
            incline_len=c.incline_len, incline_t=c.incline_t,
            handle_t=c.handle_t, handle_h=c.handle_h,
            wedge_mass=c.wedge_mass, wedge_inertia=c.wedge_inertia,
            contact_offset=c.contact_offset, jack_mu_s=c.jack_mu_s, jack_mu_d=c.jack_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
            "car": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Car", spawn=car_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.base_z + c.deck_z_lo))),
            "wedge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wedge", spawn=wedge_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.xw_anchor, 0.0, c.base_z + c.wedge_hover))),
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=sim_utils.CuboidCfg(
                    size=(c.crate_size, c.crate_size, c.crate_size),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.crate_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.jack_mu_s, dynamic_friction=c.jack_mu_d,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.crate_color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.30, 0.14, c.base_z + c.crate_size / 2 + 0.003))),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                # Without this, external wrenches are under-applied across TGS
                # iterations and the ram plant stalls far below equilibrium.
                "enable_external_forces_every_iteration": True,
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
        self.car: RigidObject = env.iscene["car"]
        self.wedge: RigidObject = env.iscene["wedge"]
        self.crate: RigidObject = env.iscene["crate"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # episode readbacks (verified by smoke)
        self.xw0 = torch.zeros(n, device=dev)          # sampled ram start (station x)
        self.crate_slot = torch.zeros(n, 2, device=dev)  # sampled crate slot (station xy)
        # drive plant state (post_step OWNS the wedge's external-wrench slot; solve/smoke
        # write ram_drive only — never call set_external_force_and_torque on the wedge).
        # ram_drive > 0 pushes the ram INWARD (station -x).
        self.ram_drive = torch.zeros(n, device=dev)
        self.ins_rate = torch.zeros(n, device=dev)     # FD rate of ram insertion (m/s, +inward)
        self.lift_rate = torch.zeros(n, device=dev)    # FD rate of car lift (m/s)
        self._xw_prev = torch.full((n,), self.cfg.xw0_base, device=dev)
        self._q_prev = torch.zeros(n, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._aboard = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lift = torch.zeros(n, device=dev)

    def _author_joints(self) -> None:
        """Per env: prismatic-Z joint station->car (the elevator guide) and prismatic-X
        joint station->wedge (the ram rail). Body0 is the heavy DYNAMIC station root so
        the anchors follow reset teleports (a kinematic body0 anchor stays world-fixed
        at the spawn pose on this stack). Joint-pair collision keeps its default
        (filtered); station<->car and station<->wedge never need contact — the only
        load-bearing pair is wedge<->car, which is unfiltered."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/car_guide")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Car"])
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.deck_z_lo))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.q_lim[0]))
            j.CreateUpperLimitAttr(float(c.q_lim[1]))

            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/ram_rail")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Wedge"])
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.xw_anchor, 0.0, c.wedge_hover))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(c.xw_lim[0] - c.xw_anchor))
            j.CreateUpperLimitAttr(float(c.xw_lim[1] - c.xw_anchor))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (free yaw + xy jitter) and BOTH jointed
        bodies together (whole-linkage write: car at q=0, ram at its sampled start),
        drop the crate at its sampled apron slot, clear drive, FD refs and latches.
        A burn draw precedes all sampling (the first post-seed draw is degenerate on
        this stack); the side choice derives from a later rand draw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        _ = torch.rand(m, 3, device=dev)  # burn (first post-seed draw is degenerate)
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_h = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.base_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_h
        self.station.write_root_state_to_sim(st, env_ids)

        # car: q = 0 (resting on the lower guide stop), same heading
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 2] = c.deck_z_lo
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_h, loc)
        st[:, 3:7] = q_h
        self.car.write_root_state_to_sim(st, env_ids)

        # ram: sampled start, withdrawal OUTWARD only (gap positive by construction)
        u = torch.rand(m, device=dev) * c.wedge_pullout
        xw0 = c.xw0_base + u
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = xw0
        loc[:, 2] = c.wedge_hover
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_h, loc)
        st[:, 3:7] = q_h
        self.wedge.write_root_state_to_sim(st, env_ids)
        self.xw0[env_ids] = xw0

        # crate: sampled side lane, x band, free yaw
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
        cx = c.crate_x0 + torch.rand(m, device=dev) * (c.crate_x1 - c.crate_x0)
        cy = side * c.crate_y
        self.crate_slot[env_ids, 0] = cx
        self.crate_slot[env_ids, 1] = cy
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = cx
        loc[:, 1] = cy
        loc[:, 2] = c.crate_size / 2 + 0.003
        yaw_c = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_h, loc)
        st[:, 3:7] = _qmul(q_h, _qz(yaw_c))
        self.crate.write_root_state_to_sim(st, env_ids)

        self.ram_drive[env_ids] = 0.0
        self.ins_rate[env_ids] = 0.0
        self.lift_rate[env_ids] = 0.0
        self._xw_prev[env_ids] = xw0
        self._q_prev[env_ids] = 0.0
        self._aboard[env_ids] = False
        self._lift[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "car": self.car.data.root_state_w[env_ids].clone(),
            "wedge": self.wedge.data.root_state_w[env_ids].clone(),
            "crate": self.crate.data.root_state_w[env_ids].clone(),
            "xw0": self.xw0[env_ids].clone(),
            "crate_slot": self.crate_slot[env_ids].clone(),
            "drive": self.ram_drive[env_ids].clone(),
            "ins_rate": self.ins_rate[env_ids].clone(),
            "lift_rate": self.lift_rate[env_ids].clone(),
            "xw_prev": self._xw_prev[env_ids].clone(),
            "q_prev": self._q_prev[env_ids].clone(),
            "aboard": self._aboard[env_ids].clone(),
            "lift": self._lift[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.car.write_root_state_to_sim(state["car"], env_ids)
        self.wedge.write_root_state_to_sim(state["wedge"], env_ids)
        self.crate.write_root_state_to_sim(state["crate"], env_ids)
        self.xw0[env_ids] = state["xw0"]
        self.crate_slot[env_ids] = state["crate_slot"]
        self.ram_drive[env_ids] = state["drive"]
        self.ins_rate[env_ids] = state["ins_rate"]
        self.lift_rate[env_ids] = state["lift_rate"]
        self._xw_prev[env_ids] = state["xw_prev"]
        self._q_prev[env_ids] = state["q_prev"]
        self._aboard[env_ids] = state["aboard"]
        self._lift[env_ids] = state["lift"]

    def resync_rate(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-anchor the FD rate references to the CURRENT poses (call after any
        manual car/wedge teleport, once the sim buffers reflect it)."""
        xw, q = self.xw(), self.q()
        if env_ids is None:
            self._xw_prev[:] = xw
            self._q_prev[:] = q
            self.ins_rate[:] = 0.0
            self.lift_rate[:] = 0.0
        else:
            self._xw_prev[env_ids] = xw[env_ids]
            self._q_prev[env_ids] = q[env_ids]
            self.ins_rate[env_ids] = 0.0
            self.lift_rate[env_ids] = 0.0

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A FREIGHT JACK stands on a grey floor slab. At its heart a GREEN elevator "
            f"platform ({int(2000 * c.deck_half)} mm square deck with a low dark-green rim) "
            f"rides a vertical guide between two dark gantry pillars; a YELLOW beam spans "
            f"the pillars — its UNDERSIDE marks the DELIVERY LEVEL, "
            f"{int(1000 * c.q_req)} mm above the platform's resting height. Under the "
            f"platform a BLUE WEDGE RAM lies on a horizontal rail running out toward the "
            f"open side of the slab; its outer end carries a tall pale PUSH PLATE. "
            f"Pushing the plate toward the gantry slides the ram's "
            f"{c.theta_deg:.0f}-degree incline under the platform's matching foot and "
            f"JACKS THE PLATFORM UP (about {int(1000 * c.q_req)} mm of lift over roughly "
            f"{int(1000 * (c.q_req / math.tan(c.theta)))} mm of push); friction "
            f"self-locks the ram, so the platform stays up when you let go. Pulling the "
            f"ram back out lowers the platform. An ORANGE CRATE "
            f"({int(1000 * c.crate_size)} mm cube) rests on the slab beside the rail "
            f"(which side, and where along it, varies by episode; the ram's starting "
            f"position and the whole station's heading vary too).\n"
            f"Goal: the crate must sit ON the platform deck, inside its rim, with the "
            f"deck jacked up to the delivery level — deck top at or above the yellow "
            f"beam's underside — and the platform must be RESTING ON THE DRIVEN RAM, "
            f"with everything at rest and nothing held. Holding the crate or the "
            f"platform in the air counts for nothing: only the settled, ram-supported "
            f"configuration succeeds. You may load the crate first and then jack, or "
            f"jack first and then load the crate onto the raised deck — the deck is open "
            f"from above at every height. The ram is driven by pushing its pale end "
            f"plate; the crate is a plain graspable cube."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the orange crate on the green elevator platform, then push the blue "
            "ram's pale end plate toward the gantry until the platform top rises to the "
            "yellow beam's underside. Leave everything at rest with the crate on the "
            "raised, ram-supported platform."
        )

    # ----- frames / live readbacks ----------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def q(self) -> torch.Tensor:
        """(N,) car lift: station-local z of the deck-top centre minus deck_z_lo."""
        return self._station_local(self.car.data.root_pos_w)[:, 2] - self.cfg.deck_z_lo

    def xw(self) -> torch.Tensor:
        """(N,) ram position: station-local x of the wedge origin (smaller = inserted)."""
        return self._station_local(self.wedge.data.root_pos_w)[:, 0]

    def q_from_ram(self) -> torch.Tensor:
        """(N,) the lift the ram currently PROVIDES: q_at(xw), clamped to the guide."""
        q0, tan_t = self.q_at_coeff()
        return (q0 - tan_t * self.xw()).clamp(min=0.0, max=self.cfg.q_lim[1])

    def q_at_coeff(self) -> tuple[float, float]:
        return self.cfg.q_at_coeff

    def aboard(self) -> torch.Tensor:
        """(N,) bool, geometric: crate centre inside the rim box, resting on the deck
        (car frame — judged identically at every lift height)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.car.data.root_quat_w,
                                 self.crate.data.root_pos_w - self.car.data.root_pos_w)
        return (loc[:, 0].abs() < c.aboard_xy_tol) & (loc[:, 1].abs() < c.aboard_xy_tol) \
            & ((loc[:, 2] - c.crate_size / 2).abs() < c.aboard_z_tol)

    def supported(self) -> torch.Tensor:
        """(N,) bool: the car's height is CONSISTENT with the ram insertion — the
        platform is resting on the driven ram, not hoisted by anything else."""
        return self.q() <= self.q_from_ram() + self.cfg.support_tol

    def settled(self) -> torch.Tensor:
        """(N,) bool: crate slow, ram insertion and car lift FD rates slow."""
        c = self.cfg
        return (self.crate.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.ins_rate.abs() < c.settle_rate) & (self.lift_rate.abs() < c.settle_rate)

    def _finite(self) -> torch.Tensor:
        ps = [self.station.data.root_pos_w, self.car.data.root_pos_w,
              self.wedge.data.root_pos_w, self.crate.data.root_pos_w]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Ram plant: clamped external drive force + viscous damping on the FD
        insertion rate, applied along the wedge's BODY -x axis (the ram never yaws
        relative to the station, so the body axis IS the rail axis — immune to the
        stack's world-frame wrench drag); then latch rubric credit. Owns the wedge's
        external-wrench slot."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        dt = self.env.dt

        xw, q = self.xw(), self.q()
        fin = torch.isfinite(xw) & torch.isfinite(q)
        raw_i = (self._xw_prev - xw) / dt  # +inward
        raw_l = (q - self._q_prev) / dt
        self.ins_rate = torch.where(
            fin, raw_i.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw_i))
        self.lift_rate = torch.where(
            fin, raw_l.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw_l))
        self._xw_prev = torch.where(fin, xw, self._xw_prev)
        self._q_prev = torch.where(fin, q, self._q_prev)

        # drive > 0 pushes inward = body -x; damping opposes the FD insertion rate
        f_ax = -(self.ram_drive.clamp(-c.f_max, c.f_max)) + c.ram_b * self.ins_rate
        f_ax = torch.nan_to_num(f_ax, nan=0.0, posinf=0.0, neginf=0.0)
        fr = torch.zeros(n, 1, 3, device=dev)
        fr[:, 0, 0] = f_ax
        self.wedge.set_external_force_and_torque(fr, torch.zeros(n, 1, 3, device=dev))

        self._update_latches(q)

    def _update_latches(self, q: torch.Tensor | None = None) -> None:
        c = self.cfg
        if q is None:
            q = self.q()
        fin = self._finite()
        ab = self.aboard() & fin
        self._aboard |= ab
        # lift progress is ABOARD- and SUPPORTED-gated: jacking the empty platform
        # earns nothing, and hoisting the loaded platform by hand (height NOT
        # consistent with the ram insertion) earns nothing either — only lift the
        # driven ram actually provides can latch.
        p = (q / c.q_req).clamp(0.0, 1.0)
        gate = ab & self.supported() & torch.isfinite(p)
        self._lift = torch.where(gate, torch.maximum(self._lift, p), self._lift)

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the crate seated on the deck inside the rim, the deck at/above
        the delivery level, the platform RESTING ON THE DRIVEN RAM (height consistent
        with the insertion — a hoisted platform never counts), everything settled and
        finite — all live physical outcomes."""
        self._update_latches()
        return self.aboard() & (self.q() >= self.cfg.q_req) & self.supported() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 crate-ever-aboard (latched) + 0.45 * latched
        aboard-AND-supported-gated lift fraction, capped at 0.75; exactly 1.0 iff
        success() holds live. Doing nothing scores ~0; jacking the empty platform
        scores ~0; holding the crate (or the loaded platform) at altitude scores at
        most the aboard latch — lift credit only accrues while the platform rests on
        the driven ram."""
        c = self.cfg
        self._update_latches()
        base = (c.w_aboard * self._aboard.float() + c.w_lift * self._lift).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="wedge_jack", robot="null"))
