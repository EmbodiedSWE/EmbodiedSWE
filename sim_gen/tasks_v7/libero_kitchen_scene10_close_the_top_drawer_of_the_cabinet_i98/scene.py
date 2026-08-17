"""CamGateCabinetScene — close a cabinet drawer by swinging a wide CAM GATE shut
(sim_gen task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i98`).

Derived from libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet, but
STRATEGICALLY different: the seed's whole skill is a guided push on the drawer front
along its prismatic travel until the joint reads closed — the hand actuates the judged
part directly, one translation. Here the intended actuation is a ROTATION OF A
DIFFERENT PART: a wide blue GATE hangs on a real geometric hinge (two knuckle rings
around a kinematic steel pin, resting on collars, captured under a cap) beside the
cabinet's open drawer. The gate's inner face, as it swings shut, presses the red
ROLLER standing on the drawer's front and CAMS the drawer closed — a
rotation-to-translation transmission that lives entirely in sliding contact dynamics
(no joints, no scripts: the hinge is captured geometry, the drawer is a free body
captured in its channel, and the cam is a real moving contact). The gate comes to
rest flat on its stop post with the drawer seated; success demands BOTH the gate flush
(within `gate_closed_deg`) and the drawer front seated (within `q_closed_tol`),
settled. Plan-level contrast with the seed: the seed translates the judged part with
the hand; here the hand's skill is a sustained arc push on a revolute cover, and the
drawer's closing translation is DELIVERED by the machine's own cam contact.

Cam geometry (all cabinet-local; face plane x=0, +x out, channel centre y=0):
the hinge pin stands at (pin_x, pin_y); the gate panel's inner face rides
`panel_offset` from the hinge axis; the roller (radius nose_r, centre set back
`nose_setback` behind the drawer's main front face, at y=nose_y) is the SOLE cam
contact by construction — the drawer's front is stepped back around it and a
support-function sweep in `__post_init__` asserts the roller beats every front corner
over the whole working range. Contact relation: q(theta) = nose_setback + pin_x +
((pin_y - nose_y) sin(theta) - (panel_offset + nose_r)) / cos(theta); at theta=0 the
gate rests EXACTLY on its stop post (post protrusion == pin_x - panel_offset) and
presses the drawer to q ~ 3 mm, inside the success tolerance, while the drawer's own
rear hard stop sits 7 mm deeper — the terminal contact chain is gate-on-post +
gate-on-roller. Everything sliding is bound to a slick physics material (min
combine): with PhysX's default ~0.5 friction the cam would jam at first touch
(transmission cos(49 deg) - 2*0.5*sin(49 deg) < 0).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): station (KINEMATIC: plinth, floor slab + apron, channel walls, rear
wall, roof, stop post, hinge pin + collars + cap), drawer (DYNAMIC open-top box with
the red roller nose), gate (DYNAMIC panel + knuckle rings + yellow handle bar).

Per-episode randomization (readback-verified by smoke): station yaw FREE (+/-180 deg)
+ xy jitter, the drawer's initial opening q0, and the gate's initial angle theta0.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.45  gate progress   — latched max closure fraction of the gate's initial angle
  0.45  drawer progress — latched max closure fraction of the drawer's initial
                          opening, counted ONLY while the drawer is genuinely in its
                          channel (a drawer stolen out of the cabinet earns nothing)
capped at 0.90; exactly 1.0 iff success(): gate flush on its stops AND drawer seated,
both settled and finite, judged LIVE. Null policy ~0 (nothing moves). The seed's
strategy — push the drawer front shut — earns the drawer credit only (~0.45) and can
never reach success: the gate is still standing open.

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


def _cyl(stage, path: str, *, cx, cy, z, r, color, collide: Callable):
    """Vertical cylinder child from centre (cx, cy), z span (z0, z1), radius r."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(r))
    cyl.CreateHeightAttr(float(z[1] - z[0]))
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -(z[1] - z[0]) / 2),
                          Gf.Vec3f(r, r, (z[1] - z[0]) / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(cx), float(cy), (z[0] + z[1]) / 2))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


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
    +x runs OUT of the cabinet toward the apron; the hinge pin stands at +y."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    Y = c.chan_hw + c.wall_t                       # channel outer half-width

    # --- plinth (raises everything to Franka-friendly heights) ----------------------
    _span(stage, f"{prim_path}/plinth", x=(-0.24, 0.20), y=(-0.17, 0.27),
          z=(-c.plinth_h, 0.0), color=c.plinth_color, collide=collide)
    # --- floor slab: channel floor + apron the drawer glides on ---------------------
    _span(stage, f"{prim_path}/slab", x=(-c.chan_len - c.back_t, c.apron_len),
          y=(-Y, Y), z=(0.0, c.slab_t), color=c.body_color, collide=collide)
    # --- pedestal arm carrying the hinge pin -----------------------------------------
    _span(stage, f"{prim_path}/pedestal", x=(0.0, 0.06), y=(Y, c.pin_y + 0.038),
          z=(0.0, c.slab_t), color=c.body_color, collide=collide)
    # --- channel: side walls, rear wall (the drawer's inward hard stop), roof --------
    zw = (c.slab_t, c.roof_z1)
    _span(stage, f"{prim_path}/wall_p", x=(-c.chan_len - c.back_t, 0.0),
          y=(c.chan_hw, Y), z=zw, color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/wall_n", x=(-c.chan_len - c.back_t, 0.0),
          y=(-Y, -c.chan_hw), z=zw, color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/rear", x=(-c.chan_len - c.back_t, -c.chan_len),
          y=(-Y, Y), z=zw, color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof", x=(-c.chan_len - c.back_t, -0.03),
          y=(-Y, Y), z=(c.roof_z0, c.roof_z1), color=c.body_color, collide=collide)
    # --- gate stop post (its protrusion IS the gate's closed rest: pin_x-panel_offset)
    _span(stage, f"{prim_path}/stop_post", x=(0.0, c.post_t),
          y=(-0.13, -0.11), z=(c.slab_t, 0.19), color=c.post_color, collide=collide)
    # --- hinge: pin + two support collars + capture cap ------------------------------
    _cyl(stage, f"{prim_path}/pin", cx=c.pin_x, cy=c.pin_y, z=(c.slab_t, 0.20),
         r=c.pin_r, color=c.steel_color, collide=collide)
    _cyl(stage, f"{prim_path}/collar_lo", cx=c.pin_x, cy=c.pin_y,
         z=(c.seat_z - 0.008, c.seat_z), r=c.collar_r, color=c.steel_color,
         collide=collide)
    _cyl(stage, f"{prim_path}/collar_hi", cx=c.pin_x, cy=c.pin_y,
         z=(c.seat_z + 0.091, c.seat_z + 0.099), r=c.collar_r, color=c.steel_color,
         collide=collide)
    _cyl(stage, f"{prim_path}/cap", cx=c.pin_x, cy=c.pin_y,
         z=(c.seat_z + 0.124, c.seat_z + 0.132), r=c.collar_r, color=c.steel_color,
         collide=collide)

    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at `prim_path`: DYNAMIC compound open-top box. Local frame:
    origin at the xy CENTRE with z=0 at the BOTTOM face; +x is the front. The main
    front face lies at local x=+L/2; the hinge-side front section is STEPPED BACK by
    `step_back` and carries the red roller nose (the sole cam contact)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L2, B2 = c.drawer_l / 2, c.drawer_w / 2
    t, zt = c.drawer_wall_t, c.drawer_h
    ystep = c.nose_y_local - 0.012                 # step spans the roller's band
    _span(stage, f"{prim_path}/floor", x=(-L2, L2), y=(-B2, B2),
          z=(0.0, c.drawer_floor_t), color=c.drawer_color, collide=collide)
    zw = (c.drawer_floor_t, zt)
    _span(stage, f"{prim_path}/w_rear", x=(-L2, -L2 + t), y=(-B2, B2),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-L2, L2 - t), y=(-B2, -B2 + t),
          z=zw, color=c.drawer_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-L2, L2 - c.step_back), y=(B2 - t, B2),
          z=zw, color=c.drawer_color, collide=collide)
    # main front face (full height band away from the hinge side)
    _span(stage, f"{prim_path}/w_front", x=(L2 - t, L2), y=(-B2, ystep),
          z=zw, color=c.drawer_color, collide=collide)
    # stepped-back front section on the hinge side (recessed by step_back)
    _span(stage, f"{prim_path}/w_front_step", x=(L2 - c.step_back - t, L2 - c.step_back),
          y=(ystep, B2), z=zw, color=c.drawer_color, collide=collide)
    # the red roller nose: the cam follower (embedded in the stepped section)
    _cyl(stage, f"{prim_path}/nose", cx=L2 - c.nose_setback, cy=c.nose_y_local,
         z=(0.015, 0.095), r=c.nose_r, color=c.nose_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.drawer_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(float(c.drawer_damping))
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gate at `prim_path`: DYNAMIC compound. Local frame: the HINGE AXIS
    is the local z axis through the origin; z=0 at the lower knuckle ring's bottom;
    at the CLOSED pose the local axes align with the station's. The panel extends to
    local -y (over the drawer front); its inner face lies at local x=-panel_offset."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    ap, bar = c.ring_aperture, c.ring_bar          # square-annulus knuckle rings
    for tag, z0 in (("lo", 0.0), ("hi", 0.100)):
        zr = (z0, z0 + 0.020)
        _span(stage, f"{prim_path}/ring_{tag}_xp", x=(ap, ap + bar),
              y=(-ap - bar, ap + bar), z=zr, color=c.steel_color, collide=collide)
        _span(stage, f"{prim_path}/ring_{tag}_xn", x=(-ap - bar, -ap),
              y=(-ap - bar, ap + bar), z=zr, color=c.steel_color, collide=collide)
        _span(stage, f"{prim_path}/ring_{tag}_yp", x=(-ap, ap),
              y=(ap, ap + bar), z=zr, color=c.steel_color, collide=collide)
        _span(stage, f"{prim_path}/ring_{tag}_yn", x=(-ap, ap),
              y=(-ap - bar, -ap), z=zr, color=c.steel_color, collide=collide)
    # panel: inner face at local x = -panel_offset (rides that far off the axis)
    _span(stage, f"{prim_path}/panel", x=(-c.panel_offset, 0.0),
          y=(-c.panel_len, -ap - bar), z=(-0.028, 0.132),
          color=c.gate_color, collide=collide)
    # yellow handle bar on the outer face near the far edge
    _span(stage, f"{prim_path}/handle", x=(0.0, 0.020),
          y=(-c.panel_len + 0.015, -c.panel_len + 0.035), z=(0.005, 0.125),
          color=c.handle_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.gate_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.5)
    prb.CreateAngularDampingAttr(0.8)
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
            chan_hw: float = 0.082
            wall_t: float = 0.012
            chan_len: float = 0.204
            back_t: float = 0.012
            slab_t: float = 0.020
            apron_len: float = 0.160
            roof_z0: float = 0.125
            roof_z1: float = 0.145
            plinth_h: float = 0.250
            post_t: float = 0.010
            pin_x: float = 0.030
            pin_y: float = 0.188
            pin_r: float = 0.010
            collar_r: float = 0.020
            seat_z: float = 0.058
            body_color: tuple = (0.45, 0.38, 0.30)
            plinth_color: tuple = (0.22, 0.22, 0.24)
            post_color: tuple = (0.25, 0.25, 0.28)
            steel_color: tuple = (0.35, 0.35, 0.40)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            drawer_l: float = 0.200
            drawer_w: float = 0.160
            drawer_wall_t: float = 0.008
            drawer_floor_t: float = 0.010
            drawer_h: float = 0.100
            step_back: float = 0.015
            nose_setback: float = 0.005
            nose_y_local: float = 0.062
            nose_r: float = 0.012
            drawer_mass: float = 0.60
            drawer_damping: float = 2.0
            drawer_color: tuple = (0.62, 0.50, 0.34)
            nose_color: tuple = (0.85, 0.10, 0.10)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        @configclass
        class GateSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            ring_aperture: float = 0.012
            ring_bar: float = 0.008
            panel_offset: float = 0.020
            panel_len: float = 0.340
            gate_mass: float = 0.50
            gate_color: tuple = (0.15, 0.35, 0.80)
            handle_color: tuple = (0.90, 0.80, 0.15)
            steel_color: tuple = (0.35, 0.35, 0.40)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.10
            slide_mu_d: float = 0.08

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
        _SPAWNER_CACHE["gate"] = GateSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CamGateCabinetSceneCfg(BaseCfg):
    """Config for `CamGateCabinetScene`. The cam contract is asserted in
    `__post_init__`: the roller is the sole cam contact over the whole working range,
    the slick material keeps the transmission alive at first touch (default friction
    would jam it), the gate's stop post IS its closed rest, and the closed gate
    presses the drawer inside the success tolerance."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    gate_closed_deg: float = tunable(5.0)   # gate within this of its flush rest
    q_closed_tol: float = tunable(0.012)    # drawer main front face within this of the face plane
    settle_lin: float = tunable(0.05)       # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.25)       # max gate |ang vel| when judging (rad/s)
    lane_y_tol: float = tunable(0.020)      # drawer centred in its channel when judged

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)         # station yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)        # station xy jitter (+/- m)
    q0_range: tuple = tunable((0.090, 0.130))       # initial drawer opening (m)
    gate0_range_deg: tuple = tunable((58.0, 80.0))  # initial gate angle (deg open)

    # --- info: station (local frame: face plane x=0, channel centre y=0, plinth top z=0) ---------
    plinth_h: float = info(0.250)           # station stands on a plinth (Franka heights)
    chan_hw: float = info(0.082)            # channel interior half-width (2 mm/side clearance)
    wall_t: float = info(0.012)
    chan_len: float = info(0.204)           # rear wall inner face at x=-chan_len
    back_t: float = info(0.012)
    slab_t: float = info(0.020)
    apron_len: float = info(0.160)
    roof_z0: float = info(0.125)
    roof_z1: float = info(0.145)
    post_t: float = info(0.010)             # stop post protrusion == pin_x - panel_offset
    pin_x: float = info(0.030)              # hinge pin axis (station frame)
    pin_y: float = info(0.188)
    pin_r: float = info(0.010)
    collar_r: float = info(0.020)
    seat_z: float = info(0.058)             # collar top: the gate's seated ring-bottom height
    # --- info: drawer ------------------------------------------------------------------------------
    drawer_l: float = info(0.200)
    drawer_w: float = info(0.160)
    drawer_wall_t: float = info(0.008)
    drawer_floor_t: float = info(0.010)
    drawer_h: float = info(0.100)
    step_back: float = info(0.015)          # hinge-side front section recessed by this
    nose_setback: float = info(0.005)       # roller centre behind the main front face
    nose_y: float = info(0.062)             # roller centre y (station frame == drawer local y)
    nose_r: float = info(0.012)
    drawer_mass: float = info(0.60)
    drawer_damping: float = info(2.0)
    # --- info: gate --------------------------------------------------------------------------------
    ring_aperture: float = info(0.012)      # knuckle ring inner half-aperture (2 mm on the pin)
    ring_bar: float = info(0.008)
    panel_offset: float = info(0.020)       # panel inner face's distance from the hinge axis
    panel_len: float = info(0.340)
    gate_mass: float = info(0.50)
    # --- info: materials ---------------------------------------------------------------------------
    slide_mu_s: float = info(0.10)          # slick everywhere sliding: default ~0.5 jams the cam
    slide_mu_d: float = info(0.08)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.45*2 = 0.90 = the non-success cap) --------------------------------
    w_gate: float = info(0.45)
    w_drawer: float = info(0.45)

    # Derived (filled in __post_init__).
    q_rest: float = field(default=None, init=False)   # drawer q pressed by the flush gate

    def cam_q(self, theta_rad: float) -> float:
        """Drawer opening q at which the gate's inner face first touches the roller,
        as a function of the gate angle (the cam relation)."""
        arm = self.pin_y - self.nose_y
        return self.nose_setback + self.pin_x + (
            arm * math.sin(theta_rad) - (self.panel_offset + self.nose_r)
        ) / math.cos(theta_rad)

    def __post_init__(self) -> None:
        c = self
        # the stop post IS the gate's closed rest: flush gate plane == post face
        assert abs(c.post_t - (c.pin_x - c.panel_offset)) < 1e-9, \
            "post protrusion must equal pin_x - panel_offset"
        # the flush gate presses the drawer INSIDE the success tolerance...
        self.q_rest = self.cam_q(0.0)
        assert 0.0 < self.q_rest < c.q_closed_tol - 0.005, \
            "flush gate must press the drawer inside q_closed_tol with margin"
        # ...and the drawer's own rear hard stop sits deeper (never the binding stop)
        q_min = -(c.chan_len - c.drawer_l)
        assert q_min < self.q_rest - 0.003, "rear stop must sit below the gate-pressed pose"
        # the roller is the SOLE cam contact over the working range (support sweep):
        # main-face corner (y = nose_y - nose_r), stepped corner (y = drawer_w/2)
        arm_nose = c.pin_y - c.nose_y
        arm_main = c.pin_y - (c.nose_y - c.nose_r)
        arm_step = c.pin_y - c.drawer_w / 2
        for deg in range(0, 53):
            th = math.radians(deg)
            s, co = math.sin(th), math.cos(th)
            beat_main = c.nose_r - c.nose_setback * co + (arm_main - arm_nose) * s
            beat_step = c.nose_r + (c.step_back - c.nose_setback) * co \
                - (arm_nose - arm_step) * s
            assert beat_main > 0.002 and beat_step > 0.002, \
                f"roller must be the sole cam contact at {deg} deg"
        # cam alive at first touch even at the widest opening (slick is load-bearing:
        # with PhysX default mu ~0.5 this transmission margin goes NEGATIVE)
        th_c = math.atan2(c.q0_range[1] + c.panel_offset + c.nose_r - c.pin_x
                          - c.nose_setback, c.pin_y - c.nose_y)  # first-touch angle (approx)
        assert math.cos(th_c) - 2.0 * c.slide_mu_s * math.sin(th_c) > 0.25, \
            "cam transmission must be alive at first touch"
        # the gate spawns clear of the drawer at every sampled pair (no overlap)
        assert self.cam_q(math.radians(c.gate0_range_deg[0])) > c.q0_range[1] + 0.010, \
            "gate at its minimum start angle must be clear of the widest drawer"
        # roller clears the channel wall all the way in
        assert c.nose_y + c.nose_r < c.chan_hw - 0.004, "roller must clear the channel wall"
        # drawer-channel clearance is 2 mm per side; drawer passes under the roof
        assert abs(c.chan_hw * 2 - c.drawer_w - 0.004) < 1e-9
        assert c.slab_t + c.drawer_h + 0.004 < c.roof_z0, "drawer must pass under the roof"
        # hinge capture: ring rides the pin with 2 mm clearance, collars/cap capture it
        assert c.ring_aperture > c.pin_r + 0.0015, "ring must ride the pin freely"
        assert c.collar_r > c.ring_aperture + 0.004, "collar/cap must capture the ring"
        # the panel fully spans the roller's height band once seated
        assert c.seat_z - 0.028 < c.slab_t + 0.015 and c.seat_z + 0.132 > c.slab_t + 0.095
        assert abs(c.w_gate + c.w_drawer - 0.90) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _yaw_of(q: torch.Tensor) -> torch.Tensor:
    """(N,) yaw angle of quats (wxyz)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cam_gate_cabinet")
class CamGateCabinetScene(BaseScene):
    cfg: CamGateCabinetSceneCfg

    def __init__(self, cfg: CamGateCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or CamGateCabinetSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            chan_hw=c.chan_hw, wall_t=c.wall_t, chan_len=c.chan_len, back_t=c.back_t,
            slab_t=c.slab_t, apron_len=c.apron_len, roof_z0=c.roof_z0,
            roof_z1=c.roof_z1, plinth_h=c.plinth_h, post_t=c.post_t, pin_x=c.pin_x,
            pin_y=c.pin_y, pin_r=c.pin_r, collar_r=c.collar_r, seat_z=c.seat_z,
            contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        drawer_spawn = cls["drawer"](
            drawer_l=c.drawer_l, drawer_w=c.drawer_w, drawer_wall_t=c.drawer_wall_t,
            drawer_floor_t=c.drawer_floor_t, drawer_h=c.drawer_h,
            step_back=c.step_back, nose_setback=c.nose_setback,
            nose_y_local=c.nose_y, nose_r=c.nose_r, drawer_mass=c.drawer_mass,
            drawer_damping=c.drawer_damping, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        gate_spawn = cls["gate"](
            ring_aperture=c.ring_aperture, ring_bar=c.ring_bar,
            panel_offset=c.panel_offset, panel_len=c.panel_len,
            gate_mass=c.gate_mass, contact_offset=c.contact_offset,
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
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate", spawn=gate_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.8, 0.30))),
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
        self.gate: RigidObject = env.iscene["gate"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.q0 = torch.zeros(n, device=dev)          # initial drawer opening
        self.theta0 = torch.zeros(n, device=dev)      # initial gate angle (rad)
        # latches (partial credit survives transients; success is judged live)
        self._fgate = torch.zeros(n, device=dev)
        self._fdrawer = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (free yaw + xy jitter), sample the drawer
        opening q0 and gate angle theta0, seat the drawer in its channel and the gate
        on its hinge pin (rings around the pin, resting on the collars)."""
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

        # sampled knobs
        q0 = c.q0_range[0] + torch.rand(m, device=dev) * (c.q0_range[1] - c.q0_range[0])
        th0 = torch.deg2rad(
            c.gate0_range_deg[0]
            + torch.rand(m, device=dev) * (c.gate0_range_deg[1] - c.gate0_range_deg[0]))
        self.q0[env_ids] = q0
        self.theta0[env_ids] = th0

        # drawer: in its channel at opening q0 (origin = centre, so x = q0 - L/2)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = q0 - c.drawer_l / 2
        loc[:, 2] = c.slab_t + 0.002
        s = torch.zeros(m, 13, device=dev)
        s[:, 0:3] = dp + origin + quat_apply(q_st, loc)
        s[:, 3:7] = q_st
        self.drawer.write_root_state_to_sim(s, env_ids)

        # gate: rings around the pin, 1 mm above the collar seat, open by theta0
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.pin_x
        loc[:, 1] = c.pin_y
        loc[:, 2] = c.seat_z + 0.001
        s = torch.zeros(m, 13, device=dev)
        s[:, 0:3] = dp + origin + quat_apply(q_st, loc)
        s[:, 3:7] = _qz(yaw + th0)
        self.gate.write_root_state_to_sim(s, env_ids)

        self._fgate[env_ids] = 0.0
        self._fdrawer[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "q0": self.q0[env_ids].clone(),
            "theta0": self.theta0[env_ids].clone(),
            "fgate": self._fgate[env_ids].clone(),
            "fdrawer": self._fdrawer[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.q0[env_ids] = state["q0"]
        self.theta0[env_ids] = state["theta0"]
        self._fgate[env_ids] = state["fgate"]
        self._fdrawer[env_ids] = state["fdrawer"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden CABINET stands on a dark plinth. Its single drawer (wood-tone, "
            f"open-topped, {c.drawer_w * 1000:.0f} mm wide) sticks OUT of the cabinet "
            f"face by roughly {c.q0_range[0] * 100:.0f}-{c.q0_range[1] * 100:.0f} cm "
            f"(the opening varies by episode). A red vertical ROLLER stands on the "
            f"drawer's front, near one corner. Beside the cabinet a steel HINGE PIN "
            f"rises from a pedestal, and a wide BLUE GATE (a flat panel, "
            f"{c.panel_len * 1000:.0f} mm wide, with a yellow HANDLE bar on its outer "
            f"face near the far edge) hangs on that pin by two knuckle rings, standing "
            f"open at {c.gate0_range_deg[0]:.0f}-{c.gate0_range_deg[1]:.0f} degrees "
            f"(the angle varies by episode; the whole station's position and heading "
            f"also vary).\n"
            f"The machine is a CAM: as the gate swings shut, its inner face meets the "
            f"red roller and PUSHES THE DRAWER CLOSED — you drive the gate (push its "
            f"panel or its yellow handle through the arc), and the gate drives the "
            f"drawer. Swing the gate all the way to its stop post, flat across the "
            f"cabinet face.\n"
            f"Goal: the gate resting flush against the cabinet front (within "
            f"{c.gate_closed_deg:.0f} degrees of flat) AND the drawer fully closed "
            f"(its front face within {c.q_closed_tol * 1000:.0f} mm of the cabinet "
            f"face), everything at rest. Both conditions are required: closing the "
            f"drawer by hand but leaving the gate standing open does NOT succeed. "
            f"The episode ends settled — a gate merely swinging past flush counts "
            f"for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Swing the wide blue gate fully shut; as it closes, its inner face "
            "presses the red roller and pushes the open drawer into the cabinet. "
            "Finish with the gate flush against the cabinet front and the drawer "
            "fully closed, everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _station_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.station.data.root_quat_w,
                                  pos_w - self.station.data.root_pos_w)

    def drawer_q(self) -> torch.Tensor:
        """(N,) drawer opening: the MAIN front face's x in the station frame
        (0 = flush with the cabinet face plane)."""
        return self._station_local(self.drawer.data.root_pos_w)[:, 0] \
            + self.cfg.drawer_l / 2

    def gate_theta(self) -> torch.Tensor:
        """(N,) gate angle in rad: 0 = flush on the stop post, positive = open."""
        return _wrap(_yaw_of(self.gate.data.root_quat_w)
                     - _yaw_of(self.station.data.root_quat_w))

    def drawer_in_channel(self) -> torch.Tensor:
        """(N,) bool: drawer genuinely riding in its channel (guards every drawer
        clause and the drawer-progress latch against a drawer stolen out of the
        cabinet)."""
        c = self.cfg
        loc = self._station_local(self.drawer.data.root_pos_w)
        return (loc[:, 1].abs() < c.lane_y_tol) \
            & ((loc[:, 2] - c.slab_t).abs() < 0.030) \
            & (loc[:, 0] + c.drawer_l / 2 > -0.020) \
            & (loc[:, 0] + c.drawer_l / 2 < 0.200)

    def drawer_closed(self) -> torch.Tensor:
        """(N,) bool: drawer seated — main front face within q_closed_tol of the
        cabinet face plane, riding in its channel."""
        return (self.drawer_q() < self.cfg.q_closed_tol) & self.drawer_in_channel()

    def gate_on_hinge(self) -> torch.Tensor:
        """(N,) bool: the gate's rings still ride the pin at seat height."""
        c = self.cfg
        loc = self._station_local(self.gate.data.root_pos_w)
        return ((loc[:, 0] - c.pin_x).abs() < 0.010) \
            & ((loc[:, 1] - c.pin_y).abs() < 0.010) \
            & ((loc[:, 2] - c.seat_z).abs() < 0.012)

    def gate_closed(self) -> torch.Tensor:
        """(N,) bool: gate flush on its stops (angle within gate_closed_deg), still
        hanging on its hinge."""
        return (self.gate_theta().abs() < math.radians(self.cfg.gate_closed_deg)) \
            & self.gate_on_hinge()

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer translation AND gate rotation at rest. NOTE the gate's
        CoM sits on the hinge axis, so its LINEAR velocity is ~0 even mid-swing —
        the angular gate is the honest stillness signal."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.gate.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < 0.5)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.gate.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        th = self.gate_theta()
        fg = ((self.theta0 - th) / self.theta0.clamp(min=1e-6)).clamp(0.0, 1.0)
        fg = torch.where(self.gate_on_hinge() & fin, fg, torch.zeros_like(fg))
        self._fgate = torch.maximum(self._fgate, fg)
        q = self.drawer_q()
        span = (self.q0 - self.cfg.q_rest).clamp(min=1e-6)
        fd = ((self.q0 - q) / span).clamp(0.0, 1.0)
        fd = torch.where(self.drawer_in_channel() & fin, fd, torch.zeros_like(fd))
        self._fdrawer = torch.maximum(self._fdrawer, fd)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: gate flush on its stop post AND drawer seated in its channel —
        live physical outcomes — settled and finite. The rubric neither knows nor
        cares what pushed what: the flush gate + seated drawer + rest is a state the
        cam geometry only admits together."""
        self._update_latches()
        return self.gate_closed() & self.drawer_closed() & self.settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.45 * latched gate-closure fraction + 0.45 *
        latched drawer-closure fraction (channel-guarded), capped at 0.90; exactly
        1.0 iff success() holds live. Doing nothing scores ~0. The seed's strategy —
        push the drawer front shut by hand — earns the drawer credit only (~0.45):
        the gate still stands open and success never fires."""
        c = self.cfg
        self._update_latches()
        base = (c.w_gate * self._fgate + c.w_drawer * self._fdrawer).clamp(max=0.90)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="cam_gate_cabinet", robot="null"))
