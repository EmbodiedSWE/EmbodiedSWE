"""ReturnDockScene — clear the obstructions so a gravity return tray closes ITSELF
(sim_gen task `close_drawer_i58`).

Derived from rlbench/close_drawer, but STRATEGICALLY different: the seed's whole
skill is a guided push on the drawer front until its prismatic joint reads closed —
the hand actuates the judged part directly. Here the ROBOT NEVER ACTUATES THE TRAY.
The station is a roofed dock standing on a wedge plinth, visibly pitched nose-up by
`incline_deg`, so its tray (a free rigid body captured between guide rails on a
polished, low-friction runway) rolls SHUT on its own the moment its path is clear.
What holds it open is a pair of obstructions, and clearing them IS the task:

  1. a red STOP PIN standing in one of three sockets sunk in the runway apron,
     wedged between the doorway and the tray's inner face — the tray presses on it
     under its own downhill weight. The pin must be lifted OUT of its socket
     (vertical extraction under that press) and set down clear of the runway.
  2. a blue BOTTLE standing inside the open-topped tray, taller than the doorway:
     even with the pin gone, the tray stalls when the bottle meets the roof header.
     The bottle must be lifted out through the tray's open top and DOCKED standing
     upright inside the green rimmed pad on the station's roof.

When both are cleared, gravity finishes the job: the tray glides down the runway,
through the doorway, and seats against the rear wall — that terminal seated pose is
the "drawer closed" outcome, but no contact the robot makes ever pushes the tray.
Plan-level contrast with the seed: the seed is one guided translation of the judged
part; here the judged part is untouched and the plan is REMOVAL — extract a loaded
pin, relocate a container's payload to a rooftop dock — with the closing motion
delegated to the scene's own mechanism. Pushing the tray (the seed's skill) is
provably useless: the socketed pin arrests it (smoke proves this with real force).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): dock (KINEMATIC: plinth, runway slab with three socket wells, guide
rails, enclosure walls, roof with header + green rimmed pad, end stop), tray
(DYNAMIC open-top box with a yellow grab bar), pin (DYNAMIC square post), bottle
(DYNAMIC cylinder). Runway/tray surfaces carry a bound polished physics material
(mu ~0.06, min combine) — without it PhysX's default ~0.5 friction beats the 8 deg
incline and the mechanism is dead (tan 8 deg ~ 0.14).

Per-episode randomization (readback-verified by smoke): dock yaw FREE (+/-180 deg)
+ xy jitter, WHICH socket holds the pin (three different openings), bottle xy
jitter inside the tray.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.20  bottle_out  — bottle ever fully out of the tray box (latched)
  0.20  docked      — bottle ever standing upright, settled, inside the roof pad
                      (latched)
  0.20  pin_clear   — pin ever fully clear of the runway corridor (latched)
  0.30  closure     — latched max closure fraction of the tray's initial opening
capped at 0.90; exactly 1.0 iff success(): tray seated closed in its channel, pin
clear of the corridor, bottle docked upright on the pad, all settled and finite.
Null policy ~0 (the pin holds the tray; nothing moves). The seed's strategy —
push the tray — earns ~0: the pin arrests the tray after ~4 mm.

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


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _spawn_dock(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dock at `prim_path`: KINEMATIC compound. Local frame: origin at the
    DOORWAY plane (x=0) on the runway floor top (z=0); +x runs OUT along the apron
    (uphill once the root is pitched); the enclosure extends to -x; the roof carries
    the green rimmed docking pad. The runway slab has three open socket wells."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    W2 = c.chan_w / 2                              # channel interior half-width
    Y = W2 + c.wall_t                              # outer half-width
    x_rear = -(c.encl_len + c.back_t)              # rear outer face
    hw = c.well_w / 2

    # --- plinth (solid wedge foundation, partly buried once the root is pitched) ----
    _span(stage, f"{prim_path}/plinth", x=(x_rear, c.apron_len + c.stop_t),
          y=(-Y, Y), z=(-c.plinth_h - c.base_t, -c.base_t), color=c.plinth_color,
          collide=collide)

    # --- runway slab: continuous bottom layer + top layer pierced by 3 socket wells --
    _span(stage, f"{prim_path}/slab_bot", x=(x_rear, c.apron_len + c.stop_t),
          y=(-Y, Y), z=(-c.base_t, -c.well_d), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/slab_yp", x=(x_rear, c.apron_len + c.stop_t),
          y=(hw, Y), z=(-c.well_d, 0.0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/slab_yn", x=(x_rear, c.apron_len + c.stop_t),
          y=(-Y, -hw), z=(-c.well_d, 0.0), color=c.body_color, collide=collide)
    xs = [x_rear] + [s for si in c.sockets for s in (si - hw, si + hw)] \
        + [c.apron_len + c.stop_t]
    for i in range(0, len(xs), 2):
        _span(stage, f"{prim_path}/slab_c{i}", x=(xs[i], xs[i + 1]),
              y=(-hw, hw), z=(-c.well_d, 0.0), color=c.body_color, collide=collide)

    # --- apron guide rails + end stop (the OPEN hard stop) ---------------------------
    _span(stage, f"{prim_path}/rail_p", x=(0.0, c.apron_len), y=(W2, Y),
          z=(0.0, c.rail_h), color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/rail_n", x=(0.0, c.apron_len), y=(-Y, -W2),
          z=(0.0, c.rail_h), color=c.frame_color, collide=collide)
    _span(stage, f"{prim_path}/end_stop", x=(c.apron_len, c.apron_len + c.stop_t),
          y=(-Y, Y), z=(0.0, c.stop_h), color=c.frame_color, collide=collide)

    # --- enclosure: side walls, rear wall (the CLOSED hard stop), roof ---------------
    _span(stage, f"{prim_path}/encl_p", x=(x_rear, 0.0), y=(W2, Y),
          z=(0.0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/encl_n", x=(x_rear, 0.0), y=(-Y, -W2),
          z=(0.0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/rear", x=(x_rear, -c.encl_len), y=(-Y, Y),
          z=(0.0, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/roof", x=(x_rear, 0.0), y=(-Y, Y),
          z=(c.door_h, c.roof_z1), color=c.body_color, collide=collide)

    # --- rooftop docking pad: plate + 4 rim walls (grippy override) ------------------
    px, py = c.pad_center
    P2, R2 = c.pad_w / 2, c.pad_w / 2 + c.rim_t
    zp = (c.roof_z1, c.pad_z1)
    zr = (c.roof_z1, c.rim_z1)
    _span(stage, f"{prim_path}/pad", x=(px - P2, px + P2), y=(py - P2, py + P2),
          z=zp, color=c.pad_color, collide=collide)
    _span(stage, f"{prim_path}/rim_xp", x=(px + P2, px + R2), y=(py - R2, py + R2),
          z=zr, color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/rim_xn", x=(px - R2, px - P2), y=(py - R2, py + R2),
          z=zr, color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/rim_yp", x=(px - P2, px + P2), y=(py + P2, py + R2),
          z=zr, color=c.rim_color, collide=collide)
    _span(stage, f"{prim_path}/rim_yn", x=(px - P2, px + P2), y=(py - R2, py - P2),
          z=zr, color=c.rim_color, collide=collide)

    # --- materials: polished everywhere (incl. the header edge: a slick header pushes
    # a stalled bottle purely horizontally and cannot roll it over the tray wall);
    # grippy override ONLY on the pad plate + rim so the docked bottle holds at tilt --
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    grippy = _mk_material(prim_path, "grippy", c.pad_mu_s, c.pad_mu_d, "average")
    bind_physics_material(prim_path, slick)
    for child in ("pad", "rim_xp", "rim_xn", "rim_yp", "rim_yn"):
        bind_physics_material(f"{prim_path}/{child}", grippy)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tray at `prim_path`: DYNAMIC compound open-top box. Local frame:
    origin at the tray's xy CENTRE with z=0 at the BOTTOM face; +x is the tray's
    outer (front) direction; the yellow grab bar protrudes from the front face."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L2, B2 = c.tray_l / 2, c.tray_w / 2
    _span(stage, f"{prim_path}/floor", x=(-L2, L2), y=(-B2, B2),
          z=(0.0, c.tray_floor_t), color=c.tray_color, collide=collide)
    zw = (c.tray_floor_t, c.tray_wall_top)
    _span(stage, f"{prim_path}/w_front", x=(L2 - c.tray_wall_t, L2), y=(-B2, B2),
          z=zw, color=c.tray_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(-L2, -L2 + c.tray_wall_t), y=(-B2, B2),
          z=zw, color=c.tray_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-L2, L2), y=(B2 - c.tray_wall_t, B2),
          z=zw, color=c.tray_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-L2, L2), y=(-B2, -B2 + c.tray_wall_t),
          z=zw, color=c.tray_color, collide=collide)
    _span(stage, f"{prim_path}/bar", x=(L2, L2 + c.bar_len),
          y=(-c.bar_hw, c.bar_hw), z=(c.bar_z0, c.bar_z1),
          color=c.bar_color, collide=collide)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.tray_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(float(cfg.tray_damping))
    prb.CreateAngularDampingAttr(4.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    slick = _mk_material(prim_path, "slick", cfg.slide_mu_s, cfg.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "dock" not in _SPAWNER_CACHE:

        @configclass
        class DockSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dock)
            chan_w: float = 0.120
            wall_t: float = 0.010
            base_t: float = 0.030
            well_d: float = 0.022
            well_w: float = 0.021
            sockets: tuple = (0.022, 0.058, 0.094)
            apron_len: float = 0.300
            rail_h: float = 0.050
            stop_t: float = 0.010
            stop_h: float = 0.045
            encl_len: float = 0.121
            back_t: float = 0.012
            door_h: float = 0.128
            roof_z1: float = 0.136
            pad_center: tuple = (-0.060, 0.0)
            pad_w: float = 0.084
            rim_t: float = 0.006
            pad_z1: float = 0.138
            rim_z1: float = 0.146
            plinth_h: float = 0.080
            body_color: tuple = (0.35, 0.38, 0.45)
            frame_color: tuple = (0.45, 0.48, 0.55)
            plinth_color: tuple = (0.28, 0.28, 0.30)
            pad_color: tuple = (0.10, 0.65, 0.15)
            rim_color: tuple = (0.06, 0.45, 0.10)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05
            pad_mu_s: float = 0.90
            pad_mu_d: float = 0.80

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_l: float = 0.120
            tray_w: float = 0.116
            tray_floor_t: float = 0.008
            tray_wall_t: float = 0.008
            tray_wall_top: float = 0.078
            bar_len: float = 0.030
            bar_hw: float = 0.030
            bar_z0: float = 0.035
            bar_z1: float = 0.055
            tray_mass: float = 0.50
            tray_damping: float = 8.0
            tray_color: tuple = (0.58, 0.55, 0.50)
            bar_color: tuple = (0.85, 0.75, 0.15)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        _SPAWNER_CACHE["dock"] = DockSpawnerCfg
        _SPAWNER_CACHE["tray"] = TraySpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ReturnDockSceneCfg(BaseCfg):
    """Config for `ReturnDockScene`. The interlock contract is asserted in
    `__post_init__`: the incline beats the polished friction (the tray really
    self-closes), the standing bottle really fouls the doorway header while the
    empty tray (and a fallen bottle) really pass, the pin really stands proud of
    everything a grasp must clear, and the dock pad really holds a bottle at the
    station's tilt."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max |lin vel| when judging (m/s)
    closed_tol: float = tunable(0.008)     # tray front face within this of the doorway plane (m)
    dock_xy_tol: float = tunable(0.030)    # bottle base centre within this of the pad centre (m)
    dock_tilt_max_deg: float = tunable(10.0)  # bottle axis within this of the dock's local up
    lane_y_tol: float = tunable(0.012)     # tray centred in its channel when judged closed (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # dock yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # dock xy jitter (+/- m)
    randomize_socket: bool = tunable(True)  # sample WHICH socket holds the pin per episode
    bottle_jitter: float = tunable(0.020)  # bottle xy jitter inside the tray (+/- m)

    # --- info: station (local frame: origin at the doorway plane on the runway floor) ------------
    incline_deg: float = info(8.0)         # nose-up pitch: -x (inward) is downhill
    dock_z: float = info(0.050)            # root height so the pitched slab rear sits on the ground
    chan_w: float = info(0.120)
    wall_t: float = info(0.010)
    base_t: float = info(0.030)
    well_d: float = info(0.022)            # socket well depth below the runway floor
    well_w: float = info(0.021)            # socket well square side
    sockets: tuple = info((0.022, 0.058, 0.094))  # socket centres along the apron
    apron_len: float = info(0.300)
    rail_h: float = info(0.050)
    stop_t: float = info(0.010)
    stop_h: float = info(0.045)
    encl_len: float = info(0.121)          # rear wall inner face (the CLOSED hard stop)
    back_t: float = info(0.012)
    door_h: float = info(0.128)            # doorway clearance (roof underside)
    roof_z1: float = info(0.136)
    pad_center: tuple = info((-0.060, 0.0))
    pad_w: float = info(0.084)             # pad plate square side (rim inner 0.072)
    rim_t: float = info(0.006)
    pad_z1: float = info(0.138)            # pad plate top (bottle stands here)
    rim_z1: float = info(0.146)
    plinth_h: float = info(0.080)
    # --- info: tray ------------------------------------------------------------------------------
    tray_l: float = info(0.120)
    tray_w: float = info(0.116)
    tray_floor_t: float = info(0.008)
    tray_wall_t: float = info(0.008)
    tray_wall_top: float = info(0.078)     # wall top above the tray bottom face
    bar_len: float = info(0.030)
    tray_mass: float = info(0.50)
    tray_damping: float = info(8.0)        # terminal glide ~0.11 m/s at 8 deg
    # --- info: pin -------------------------------------------------------------------------------
    pin_w: float = info(0.018)             # square section
    pin_h: float = info(0.140)
    pin_mass: float = info(0.10)
    # --- info: bottle ----------------------------------------------------------------------------
    bottle_r: float = info(0.0225)
    bottle_h: float = info(0.140)
    bottle_mass: float = info(0.25)
    # --- info: materials / colors ----------------------------------------------------------------
    slide_mu_s: float = info(0.06)         # polished runway: default ~0.5 would beat the incline
    slide_mu_d: float = info(0.05)
    pad_mu_s: float = info(0.90)           # grippy roof/pad: the docked bottle must not creep
    pad_mu_d: float = info(0.80)
    contact_offset: float = info(0.0015)
    pin_color: tuple = info((0.80, 0.10, 0.10))
    bottle_color: tuple = info((0.10, 0.30, 0.85))
    # --- info: rubric weights (0.2*3 + 0.3 = 0.90 = the non-success cap) -------------------------
    w_out: float = info(0.20)
    w_dock: float = info(0.20)
    w_pin: float = info(0.20)
    w_close: float = info(0.30)

    # Derived (filled in __post_init__).
    openings: tuple = field(default=None, init=False)  # tray front-face x per socket

    def __post_init__(self) -> None:
        th = math.radians(self.incline_deg)
        # the mechanism is alive: gravity beats the polished friction with margin
        assert math.tan(th) > 2.0 * self.slide_mu_s, "incline must beat runway friction 2x"
        # the pad holds a docked bottle at the tilt: no tip, no slide
        assert math.atan2(self.bottle_r, self.bottle_h / 2) > th + math.radians(3.0), \
            "bottle must stand stably on the pitched pad"
        assert self.pad_mu_s > 2.0 * math.tan(th), "pad friction must pin the docked bottle"
        # doorway interlock: standing bottle fouls the header, everything else passes
        bot_top = self.tray_floor_t + self.bottle_h + 0.002
        assert bot_top > self.door_h + 0.020, "standing bottle must foul the doorway header"
        # a fallen bottle's WORST-case height is the bridged pose (lying across both
        # wall tops, near horizontal): wall_top + one full diameter of shaft above it
        lean_top = self.tray_wall_top + 2.0 * self.bottle_r + 0.004
        assert lean_top < self.door_h, "a fallen bottle must pass under the header"
        assert self.tray_wall_top + 0.004 < self.door_h - 0.008, "empty tray must pass"
        assert 0.055 + 0.004 < self.door_h - 0.008, "grab bar must pass under the header"
        # pin: stands proud for a top grasp, seats in its well, blocks the tray face
        pin_top = -self.well_d + self.pin_h
        assert pin_top > self.tray_wall_top + 0.030, "pin top must stand proud of the tray walls"
        assert self.pin_w < self.well_w - 0.002, "pin must drop freely into its well"
        assert self.well_d > self.pin_w, "well must be deep enough to hold the pin upright"
        # travel bookkeeping: openings on the apron, inside the end stop
        self.openings = tuple(s + self.pin_w / 2 + self.tray_l for s in self.sockets)
        assert all(o + 0.010 < self.apron_len for o in self.openings), \
            "tray at every socket must sit clear of the end stop"
        assert self.encl_len > self.tray_l, "closed tray must fit inside the enclosure"
        assert self.encl_len - self.tray_l < self.closed_tol, \
            "the rear-wall hard stop must BE the closed pose"
        assert abs(self.chan_w - self.tray_w - 0.004) < 1e-9, \
            "tray-channel clearance is 2 mm per side"
        assert self.bottle_r * 2 < self.tray_w - 2 * self.tray_wall_t - 0.01, \
            "bottle must fit the tray interior"
        assert abs(self.w_out + self.w_dock + self.w_pin + self.w_close - 0.90) < 1e-9


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


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("return_dock")
class ReturnDockScene(BaseScene):
    cfg: ReturnDockSceneCfg

    def __init__(self, cfg: ReturnDockSceneCfg | None = None) -> None:
        super().__init__(cfg or ReturnDockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        dock_spawn = cls["dock"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            chan_w=c.chan_w, wall_t=c.wall_t, base_t=c.base_t, well_d=c.well_d,
            well_w=c.well_w, sockets=c.sockets, apron_len=c.apron_len,
            rail_h=c.rail_h, stop_t=c.stop_t, stop_h=c.stop_h, encl_len=c.encl_len,
            back_t=c.back_t, door_h=c.door_h, roof_z1=c.roof_z1,
            pad_center=c.pad_center, pad_w=c.pad_w, rim_t=c.rim_t, pad_z1=c.pad_z1,
            rim_z1=c.rim_z1, plinth_h=c.plinth_h, contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d,
            pad_mu_s=c.pad_mu_s, pad_mu_d=c.pad_mu_d)
        tray_spawn = cls["tray"](
            tray_l=c.tray_l, tray_w=c.tray_w, tray_floor_t=c.tray_floor_t,
            tray_wall_t=c.tray_wall_t, tray_wall_top=c.tray_wall_top,
            bar_len=c.bar_len, tray_mass=c.tray_mass, tray_damping=c.tray_damping,
            contact_offset=c.contact_offset,
            slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
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
            "dock": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dock", spawn=dock_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.dock_z))),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray", spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.0, 0.05))),
            "pin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pin_w, c.pin_w, c.pin_h),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pin_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pin_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.6, 0.10))),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.CylinderCfg(
                    radius=c.bottle_r, height=c.bottle_h, axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.45, dynamic_friction=0.40, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bottle_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, -0.6, 0.10))),
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
        self.dock: RigidObject = env.iscene["dock"]
        self.tray: RigidObject = env.iscene["tray"]
        self.pin: RigidObject = env.iscene["pin"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.socket_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.x0 = torch.zeros(n, device=dev)          # initial tray front-face x
        # latches (partial credit survives transients; success is judged live)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dock_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pin_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._close_f = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the dock (free yaw + xy jitter, fixed nose-up pitch),
        sample WHICH socket holds the pin, seat the pin in it, rest the tray 4 mm
        uphill of the pin (it settles onto it), stand the bottle in the tray with xy
        jitter, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        half = -math.radians(c.incline_deg) / 2          # nose-up: local +x tilts UP
        q_pitch = torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0],
                               device=dev).expand(m, 4)
        q_dock = _qmul(_qz(yaw), q_pitch)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.dock_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_dock
        self.dock.write_root_state_to_sim(st, env_ids)

        def place(body, local: torch.Tensor, q_extra: torch.Tensor | None = None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_dock, local)
            s[:, 3:7] = q_dock if q_extra is None else _qmul(q_dock, q_extra)
            body.write_root_state_to_sim(s, env_ids)

        # which socket holds the pin
        if c.randomize_socket:
            k = torch.randint(0, len(c.sockets), (m,), device=dev)
        else:
            k = torch.zeros(m, dtype=torch.long, device=dev)
        self.socket_idx[env_ids] = k
        s_x = torch.tensor(c.sockets, device=dev)[k]
        self.x0[env_ids] = torch.tensor(c.openings, device=dev)[k]

        # pin: seated in its well
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = s_x
        loc[:, 2] = -c.well_d + c.pin_h / 2 + 0.001
        place(self.pin, loc)

        # tray: front face 4 mm uphill of the settled opening (it glides onto the pin)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = self.x0[env_ids] + 0.004 - c.tray_l / 2
        loc[:, 2] = 0.0015
        place(self.tray, loc)

        # bottle: standing in the tray with xy jitter
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = self.x0[env_ids] + 0.004 - c.tray_l / 2 \
            + (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
        loc[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.bottle_jitter
        loc[:, 2] = 0.0015 + c.tray_floor_t + c.bottle_h / 2 + 0.002
        place(self.bottle, loc)

        self._out[env_ids] = False
        self._dock_l[env_ids] = False
        self._pin_l[env_ids] = False
        self._close_f[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "dock": self.dock.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "pin": self.pin.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "socket_idx": self.socket_idx[env_ids].clone(),
            "x0": self.x0[env_ids].clone(),
            "out": self._out[env_ids].clone(),
            "dock_l": self._dock_l[env_ids].clone(),
            "pin_l": self._pin_l[env_ids].clone(),
            "close_f": self._close_f[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.dock.write_root_state_to_sim(state["dock"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.pin.write_root_state_to_sim(state["pin"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.socket_idx[env_ids] = state["socket_idx"]
        self.x0[env_ids] = state["x0"]
        self._out[env_ids] = state["out"]
        self._dock_l[env_ids] = state["dock_l"]
        self._pin_l[env_ids] = state["pin_l"]
        self._close_f[env_ids] = state["close_f"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A RETURN STATION stands on the ground on a dark wedge plinth, visibly "
            f"pitched nose-up by {c.incline_deg:.0f} degrees: a slate-blue roofed "
            f"enclosure at the low end, and a polished open runway (the apron, with "
            f"low guide rails) running uphill from its doorway. A light-grey open-top "
            f"TRAY with a yellow grab bar on its front face sits on the runway; the "
            f"station is built so the tray GLIDES DOWNHILL AND SEATS ITSELF fully "
            f"inside the enclosure whenever its path is clear — nobody ever needs to "
            f"push it. Right now two obstructions hold it open:\n"
            f"1. a RED STOP PIN ({c.pin_w * 1000:.0f} mm square, "
            f"{c.pin_h * 1000:.0f} mm tall) standing in one of three square sockets "
            f"sunk in the runway between the doorway and the tray (which socket "
            f"varies by episode, so the tray's opening varies too). The tray presses "
            f"against it under its own weight. Lift the pin straight up out of its "
            f"socket and set it down anywhere well clear of the runway — do not "
            f"leave it standing in a socket, lying in the tray's path, or inside the "
            f"tray. Pushing the tray against the pin cannot close it: the socket "
            f"holds.\n"
            f"2. a BLUE BOTTLE (cylinder, {2 * c.bottle_r * 1000:.0f} mm across, "
            f"{c.bottle_h * 1000:.0f} mm tall) standing inside the open-top tray, "
            f"taller than the doorway: even with the pin gone the tray stalls when "
            f"the bottle meets the roof edge. Lift the bottle out through the tray's "
            f"open top and stand it UPRIGHT on the GREEN RIMMED PAD on the "
            f"enclosure's roof, inside the rim. A bottle knocked over into the tray "
            f"slips under the doorway and rides inside — that does not count and "
            f"must be fished out again (pull the yellow grab bar to reopen).\n"
            f"Goal: bottle standing upright inside the green pad, pin clear of the "
            f"runway, and the tray fully seated in the enclosure (front face flush "
            f"with the doorway) — which it does by itself once you clear its path. "
            f"Either obstruction may be cleared first. The episode ends settled: "
            f"tray seated, bottle docked, pin clear, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift the red stop pin out of its socket and set it clear of the "
            "runway, then move the blue bottle from the tray onto the green rimmed "
            "pad on the station's roof, standing upright. The tilted station rolls "
            "its tray shut by itself; finish with the tray fully seated in the "
            "enclosure and everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _dock_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.dock.data.root_quat_w,
                                  pos_w - self.dock.data.root_pos_w)

    def face_x(self) -> torch.Tensor:
        """(N,) tray FRONT-FACE x in the dock frame: ~0 = seated closed at the
        doorway plane; `openings[k]` = held open on socket k."""
        return self._dock_local(self.tray.data.root_pos_w)[:, 0] + self.cfg.tray_l / 2

    def tray_closed(self) -> torch.Tensor:
        """(N,) bool: tray seated in its channel with the front face at the doorway."""
        c = self.cfg
        loc = self._dock_local(self.tray.data.root_pos_w)
        return (loc[:, 0] + c.tray_l / 2 < c.closed_tol) \
            & (loc[:, 1].abs() < c.lane_y_tol) & ((loc[:, 2] - 0.0015).abs() < 0.020)

    def pin_clear(self) -> torch.Tensor:
        """(N,) bool: pin centre fully OUTSIDE the runway corridor (the volume the
        tray or its payload can sweep, doorway to end stop plus the enclosure)."""
        c = self.cfg
        loc = self._dock_local(self.pin.data.root_pos_w)
        inside = (loc[:, 0] > -(c.encl_len + c.back_t)) \
            & (loc[:, 0] < c.apron_len + c.stop_t + 0.005) \
            & (loc[:, 1].abs() < c.chan_w / 2 + c.wall_t + 0.005) \
            & (loc[:, 2] > -c.well_d - 0.010) & (loc[:, 2] < 0.100)
        return ~inside

    def _bottle_axis_dot(self) -> torch.Tensor:
        """(N,) cos(angle) between the bottle axis and the dock's local up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        ax = quat_apply(self.bottle.data.root_quat_w, ez)
        up = quat_apply(self.dock.data.root_quat_w, ez)
        return (ax * up).sum(-1)

    def bottle_out(self) -> torch.Tensor:
        """(N,) bool: bottle centre fully OUT of the tray box (tray body frame,
        generous margins so the latch only fires when genuinely clear)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.tray.data.root_quat_w,
                                 self.bottle.data.root_pos_w - self.tray.data.root_pos_w)
        inside = (loc[:, 0].abs() < c.tray_l / 2 + 0.015) \
            & (loc[:, 1].abs() < c.tray_w / 2 + 0.015) \
            & (loc[:, 2] > -0.020) & (loc[:, 2] < c.tray_wall_top + 0.030)
        return ~inside

    def bottle_docked(self) -> torch.Tensor:
        """(N,) bool: bottle standing upright inside the roof pad (dock frame):
        base centre within `dock_xy_tol` of the pad centre, base on the pad plate,
        axis within `dock_tilt_max_deg` of the dock's local up."""
        c = self.cfg
        loc = self._dock_local(self.bottle.data.root_pos_w)
        px, py = c.pad_center
        near = ((loc[:, 0] - px).abs() < c.dock_xy_tol) \
            & ((loc[:, 1] - py).abs() < c.dock_xy_tol)
        on_pad = ((loc[:, 2] - (c.pad_z1 + c.bottle_h / 2)).abs() < 0.012)
        upright = self._bottle_axis_dot() >= math.cos(math.radians(c.dock_tilt_max_deg))
        return near & on_pad & upright

    def settled(self) -> torch.Tensor:
        """(N,) bool: tray, pin and bottle |lin vel| below the gate."""
        c = self.cfg
        return (self.tray.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.pin.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.tray.data.root_pos_w, self.pin.data.root_pos_w,
                         self.bottle.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._out |= self.bottle_out() & fin
        bslow = self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        self._dock_l |= self.bottle_docked() & bslow & fin
        self._pin_l |= self.pin_clear() & fin
        frac = ((self.x0 - self.face_x()) / (self.x0 - 0.001).clamp(min=1e-6)).clamp(0.0, 1.0)
        self._close_f = torch.where(fin, torch.maximum(self._close_f, frac), self._close_f)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: tray seated closed in its channel, pin clear of the runway
        corridor, bottle docked upright on the roof pad — all live physical
        outcomes — settled and finite. The tray got there under gravity alone; the
        rubric neither knows nor cares who pushed what, only that the path was
        genuinely cleared (a blocked path physically cannot yield this state)."""
        self._update_latches()
        return self.tray_closed() & self.pin_clear() & self.bottle_docked() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 bottle-out + 0.20 docked + 0.20 pin-clear
        (all latched) + 0.30 * latched max closure fraction, capped at 0.90;
        exactly 1.0 iff success() holds live. Doing nothing scores ~0 (the pin
        holds the tray at its opening); pushing the tray — the seed's whole
        strategy — moves it ~4 mm onto the pin and also scores ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_out * self._out.float() + c.w_dock * self._dock_l.float()
                + c.w_pin * self._pin_l.float() + c.w_close * self._close_f).clamp(max=0.90)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="return_dock", robot="null"))
