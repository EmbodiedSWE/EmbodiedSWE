"""ShiftParkGrillScene — open the grill by a SLIDE-then-ROTATE-then-SLIDE gated lid
sequence, park the lid leaning on a high rest bar, then deliver a patty through the
opened top mouth onto the internal grate.

Derived from rlbench/open_grill ("open the grill": an articulated barbecue USD whose
hinged lid the robot swings up about its hinge, judged by the lid joint angle). The
hinged-lid idea is KEPT — but the seed's plan (grab the lid and rotate it up) is made
geometrically impossible AT SPAWN: the lid rides a 2-DOF worn mount (a generic D6
joint: 80 mm of horizontal SLIDE + a pitch hinge capped at 85 deg — strictly under
over-center, so the lid can NEVER rest open on its own joint), and at spawn the lid
sits at the FORWARD end of its slide with its front tip tucked UNDER an overhanging
CATCH bar: rotating up jams on the catch within ~3 deg. Opening requires a genuinely
different, physically ORDERED plan:

  1. SLIDE the lid BACK its full 80 mm stroke (the tip leaves the catch's overhang;
     only now can the lid pitch freely);
  2. ROTATE it up to ~80 deg at the REAR slide position — and only there: the fixed
     REST BAR spanning high above the mouth is OUT OF REACH of the lid's sweep from
     the rear pin (radial clearance, asserted), so the upswing clears it;
  3. SLIDE the raised lid FORWARD again: now the bar radius from the forward pin is
     SHORTER than the plate, so on RELEASE gravity drops the lid ~20 deg until its
     underside lands on the rest bar and a stop CLEAT on the underside seats against
     the bar — the lid PARKS leaning at ~60 deg, gravity-held, hands off. Releasing
     at the rear instead drops the lid fully closed (the bar is out of reach; there
     is no over-center rest): the forward slide is load-bearing.
  4. With the mouth open, DELIVER the patty from the side board down through the top
     mouth onto the grate. The mouth is the chamber's only aperture (the closed lid
     hovers 24 mm over the rim — less than the patty's 32 mm thickness).

Assets are fully procedural:
  - grill: HEAVY DYNAMIC compound (40 kg — dynamic so the spawn-authored joint
    teleports with it at reset): chamber (floor + 4 walls, top mouth), 4 internal
    grate slats, front CATCH posts + overhang bar, rear slide-rail posts + cosmetic
    rails, two tall side TOWERS carrying the high REST BAR, and a side BOARD (the
    patty's spawn table) — one rigid body.
  - lid: DYNAMIC plate (250 x 240 x 12 mm, 0.35 kg) with a top HANDLE bar and the
    underside stop CLEAT, on a spawn-authored generic D6 joint to the grill:
    transX in [0, 80 mm] (slide), rotY in [-85 deg, 0] (pitch; 0 = closed), all
    other axes locked; lid<->grill contact stays ON (catch/bar/rim contacts ARE
    lid-grill contacts).
  - patty: DYNAMIC 60 x 60 x 32 mm brown block on the side board.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.15 * released — lid ever at the rear of its slide while still low (latched)
  0.20 * raised   — lid pitch ever >= 70 deg (unreachable at spawn: the catch jams
                    the upswing at ~3 deg)                             (latched)
  0.30 * parked   — lid ever settled in the park band (50..80 deg) with the slide
                    forward                                           (latched)
  1.0 iff success() — lid PARKED live (pitch in 50..80 deg, slide forward, at rest
                    on the bar), patty ON THE GRATE (grill-frame window, below the
                    mouth), grill upright, everything settled and finite.
                    Non-success capped at 0.70; null policy ~0.

Honesty geometry (asserted in `__post_init__`):
  - the catch jams the spawn-position upswing at a few degrees, FAR below the
    `raised` latch, and stays engaged over the whole spawn slide jitter;
  - the rest bar is radially OUT of the sweep from the rear pin (clearances with
    contact offsets) and radially INSIDE the plate from the forward pin, with the
    park contact angle inside the success band with margin;
  - the pitch cap (85 deg) is under over-center: no joint-stop rest exists, so an
    in-band settled pitch implies a real lean on the bar;
  - the closed lid's hover gap over the rim is smaller than the patty, and a patty
    on the closed lid rests far above the on-grate window: the mouth is the only
    way in, and it is open only while the lid is parked/raised;
  - the parked lid leaves a straight vertical drop corridor through the mouth onto
    the grate; slat gaps are too narrow for the patty to fall through.

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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4), pure torch."""
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, *, center, size, color, contact_offset: float):
    """One collidable box child: translate + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(box.GetPrim(), contact_offset)
    return box.GetPrim()


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_grill(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the HEAVY DYNAMIC grill compound at `prim_path`. Local frame: origin at
    the chamber footprint centre on the ground; forward = +x, mouth up; the patty
    board sits on the +y side. One rigid body: chamber + grate + catch + slide-rail
    posts + rest-bar towers + board."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.04, 0.08))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.5, 1.5, 2.0))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    co = c.contact_offset
    body_c, grate_c = c.body_color, c.grate_color
    catch_c, tower_c, board_c = c.catch_color, c.tower_color, c.board_color
    kids = []

    def add(path, center, size, color):
        kids.append(_box(stage, f"{prim_path}/{path}", center=center, size=size,
                         color=color, contact_offset=co))

    # --- chamber: floor + 4 walls (inner cavity open at the top mouth) ---
    wall_hz = c.wall_h - c.floor_t
    wall_cz = (c.floor_t + c.wall_h) / 2
    add("floor", (0.0, 0.0, c.floor_t / 2), (2 * c.hx, 2 * c.hy, c.floor_t), body_c)
    add("wall_f", (c.hx - c.wall_t / 2, 0.0, wall_cz),
        (c.wall_t, 2 * c.hy, wall_hz), body_c)
    add("wall_r", (-(c.hx - c.wall_t / 2), 0.0, wall_cz),
        (c.wall_t, 2 * c.hy, wall_hz), body_c)
    add("wall_l", (0.0, c.hy - c.wall_t / 2, wall_cz),
        (2 * (c.hx - c.wall_t), c.wall_t, wall_hz), body_c)
    add("wall_rt", (0.0, -(c.hy - c.wall_t / 2), wall_cz),
        (2 * (c.hx - c.wall_t), c.wall_t, wall_hz), body_c)
    # --- grate slats (tops at z = grate_z), spanning y inside the chamber ---
    for i, sx in enumerate(c.slat_xs):
        add(f"slat{i}", (float(sx), 0.0, c.grate_z - c.slat_t / 2),
            (c.slat_w, 2 * (c.hy - c.wall_t) - 0.004, c.slat_t), grate_c)
    # --- front CATCH: two posts + the overhang bar over the lid tip ---
    cx = (c.catch_x0 + c.catch_x1) / 2
    cwx = c.catch_x1 - c.catch_x0
    post_y = (c.post_y0 + c.post_y1) / 2
    post_w = c.post_y1 - c.post_y0
    add("catch_post_l", (cx, post_y, c.catch_z1 / 2), (cwx, post_w, c.catch_z1), catch_c)
    add("catch_post_r", (cx, -post_y, c.catch_z1 / 2), (cwx, post_w, c.catch_z1), catch_c)
    add("catch_bar", (cx, 0.0, (c.catch_z0 + c.catch_z1) / 2),
        (cwx, 2 * c.post_y1, c.catch_z1 - c.catch_z0), catch_c)
    # --- rear slide-rail posts + cosmetic side rails (the visible "worn mount") ---
    add("rail_post_l", (c.rail_post_x, post_y, c.rail_z1 / 2),
        (0.03, post_w, c.rail_z1), tower_c)
    add("rail_post_r", (c.rail_post_x, -post_y, c.rail_z1 / 2),
        (0.03, post_w, c.rail_z1), tower_c)
    add("rail_l", ((c.rail_x0 + c.rail_x1) / 2, post_y, (c.rail_z0 + c.rail_z1) / 2),
        (c.rail_x1 - c.rail_x0, post_w, c.rail_z1 - c.rail_z0), tower_c)
    add("rail_rt", ((c.rail_x0 + c.rail_x1) / 2, -post_y, (c.rail_z0 + c.rail_z1) / 2),
        (c.rail_x1 - c.rail_x0, post_w, c.rail_z1 - c.rail_z0), tower_c)
    # --- rest-bar TOWERS + the high REST BAR the parked lid leans on ---
    tw_y = (c.tower_y0 + c.tower_y1) / 2
    tw_w = c.tower_y1 - c.tower_y0
    add("tower_l", (c.bar_x, tw_y, c.tower_h / 2), (0.03, tw_w, c.tower_h), tower_c)
    add("tower_r", (c.bar_x, -tw_y, c.tower_h / 2), (0.03, tw_w, c.tower_h), tower_c)
    add("rest_bar", (c.bar_x, 0.0, c.bar_z),
        (c.bar_s, 2 * c.tower_y1, c.bar_s), catch_c)
    # --- side BOARD (patty spawn table): solid pedestal on the +y side ---
    add("board", (0.0, (c.board_y0 + c.board_y1) / 2, c.board_h / 2),
        (2 * c.board_hx, c.board_y1 - c.board_y0, c.board_h), board_c)
    for k in kids:
        _bind_material(k, mat)
    return root
def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC lid at `prim_path`: body origin ON the pitch pin axis,
    plate extending toward local +x (pitch 0 = closed flat over the mouth), with the
    top HANDLE, the underside stop CLEAT, and two cosmetic side lugs. The 2-DOF worn
    mount — a generic D6 joint to the sibling grill (transX slide in [0, s], rotY
    pitch in [-pitch_max, 0], others locked) — is authored in-spawn. Lid<->grill
    contact stays ON (the catch jam, the rim hover and the bar park are contacts)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    c = cfg
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(float(c.com_x), 0.0, 0.006))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.7e-3, 2.2e-3, 3.9e-3))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    co = c.contact_offset
    kids = [
        # plate: local x plate_x0..plate_x1, mid-plane through the pin axis
        _box(stage, f"{prim_path}/plate",
             center=((c.plate_x0 + c.plate_x1) / 2, 0.0, 0.0),
             size=(c.plate_x1 - c.plate_x0, c.width, c.t), color=c.color,
             contact_offset=co),
        # underside stop CLEAT (seats against the rest bar when parked)
        _box(stage, f"{prim_path}/cleat",
             center=((c.cleat_x0 + c.cleat_x1) / 2, 0.0, -(c.t / 2 + c.cleat_h / 2)),
             size=(c.cleat_x1 - c.cleat_x0, c.width, c.cleat_h), color=c.color,
             contact_offset=co),
        # top handle: two stanchions + crossbar (parallel-jaw graspable)
        _box(stage, f"{prim_path}/stanchion_l", center=(c.handle_x, 0.05, c.t / 2 + 0.015),
             size=(0.014, 0.014, 0.03), color=c.handle_color, contact_offset=co),
        _box(stage, f"{prim_path}/stanchion_r", center=(c.handle_x, -0.05, c.t / 2 + 0.015),
             size=(0.014, 0.014, 0.03), color=c.handle_color, contact_offset=co),
        _box(stage, f"{prim_path}/handle", center=(c.handle_x, 0.0, c.t / 2 + 0.037),
             size=(0.014, 0.13, 0.014), color=c.handle_color, contact_offset=co),
        # cosmetic side lugs riding the rails
        _box(stage, f"{prim_path}/lug_l", center=(0.0, 0.127, 0.0),
             size=(0.024, 0.010, 0.024), color=c.handle_color, contact_offset=co),
        _box(stage, f"{prim_path}/lug_r", center=(0.0, -0.127, 0.0),
             size=(0.024, 0.010, 0.024), color=c.handle_color, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    # ---- the WORN MOUNT: generic D6 joint to the sibling grill. transX = the
    # slide (0 = rear, s = forward), rotY = the pitch (0 = closed, raising the tip
    # = negative rotY). Limits in UsdPhysics convention: low > high locks an axis.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/worn_mount")
    j.CreateBody0Rel().SetTargets([f"{base}/Grill"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(True)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.pin_x_rear), 0.0, float(c.pin_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim = j.GetPrim()
    for axis in ("transY", "transZ", "rotX", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)  # low > high = locked
    lim = UsdPhysics.LimitAPI.Apply(prim, "transX")
    lim.CreateLowAttr(0.0)
    lim.CreateHighAttr(float(c.slide_s))
    lim = UsdPhysics.LimitAPI.Apply(prim, "rotY")
    lim.CreateLowAttr(-float(c.pitch_max_deg))
    lim.CreateHighAttr(0.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "grill" not in _SPAWNER_CACHE:

        @configclass
        class GrillSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_grill)
            mass: float = 40.0
            hx: float = 0.13
            hy: float = 0.10
            wall_t: float = 0.010
            wall_h: float = 0.16
            floor_t: float = 0.010
            grate_z: float = 0.075
            slat_t: float = 0.012
            slat_w: float = 0.036
            slat_xs: tuple = (-0.09, -0.03, 0.03, 0.09)
            catch_x0: float = 0.150
            catch_x1: float = 0.185
            catch_z0: float = 0.204
            catch_z1: float = 0.220
            post_y0: float = 0.140
            post_y1: float = 0.166
            rail_post_x: float = -0.145
            rail_x0: float = -0.195
            rail_x1: float = -0.085
            rail_z0: float = 0.196
            rail_z1: float = 0.208
            bar_x: float = 0.020
            bar_z: float = 0.398
            bar_s: float = 0.012
            tower_y0: float = 0.140
            tower_y1: float = 0.170
            tower_h: float = 0.42
            board_hx: float = 0.08
            board_y0: float = 0.18
            board_y1: float = 0.30
            board_h: float = 0.10
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            body_color: tuple = (0.20, 0.22, 0.24)
            grate_color: tuple = (0.35, 0.36, 0.38)
            catch_color: tuple = (0.75, 0.20, 0.12)
            tower_color: tuple = (0.80, 0.65, 0.20)
            board_color: tuple = (0.55, 0.40, 0.22)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            mass: float = 0.35
            com_x: float = 0.14
            plate_x0: float = 0.02
            plate_x1: float = 0.27
            width: float = 0.24
            t: float = 0.012
            cleat_x0: float = 0.256
            cleat_x1: float = 0.268
            cleat_h: float = 0.014
            handle_x: float = 0.20
            pin_x_rear: float = -0.18
            pin_z: float = 0.19
            slide_s: float = 0.08
            pitch_max_deg: float = 85.0
            mu_static: float = 0.40
            mu_dynamic: float = 0.35
            color: tuple = (0.72, 0.74, 0.78)
            handle_color: tuple = (0.15, 0.30, 0.70)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["grill"] = GrillSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShiftParkGrillSceneCfg(BaseCfg):
    """Config for `ShiftParkGrillScene`. The catch jams the spawn-position upswing,
    the rest bar clears the rear-pin sweep and intercepts the forward-pin fall inside
    the park band, the pitch cap is under over-center, and the mouth is the chamber's
    only patty-sized aperture. All asserted in `__post_init__`."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    park_lo_deg: float = tunable(50.0)      # lid pitch band that counts as parked...
    park_hi_deg: float = tunable(80.0)      # ...(the bar lean sits at ~60 deg)
    slide_fwd_min: float = tunable(0.05)    # slide readback >= this -> forward (parked)
    slide_rear_max: float = tunable(0.015)  # slide readback <= this -> rear (released)
    raised_deg: float = tunable(70.0)       # pitch ever past this -> `raised` latch
    released_low_deg: float = tunable(15.0)  # `released` latch needs pitch under this
    patty_x_tol: float = tunable(0.105)     # patty grill-frame |x| on the grate
    patty_y_tol: float = tunable(0.080)     # patty grill-frame |y| on the grate
    patty_z_lo: float = tunable(0.055)      # patty grill-frame z band on the grate
    patty_z_hi: float = tunable(0.130)      # (far below the mouth plane / closed lid)
    upright_deg: float = tunable(10.0)      # grill up-axis cone
    settle_speed: float = tunable(0.05)     # max |lin vel| (lid, patty) when judging

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    grill_yaw_deg: float = tunable(20.0)    # grill heading jitter (+/- deg)
    grill_jitter: float = tunable(0.030)    # grill xy jitter (+/- m)
    slide_jitter: float = tunable(0.006)    # lid spawn slide back-off from the stop (0..this)
    patty_x_jitter: float = tunable(0.025)  # patty spawn jitter on the board (+/- m)
    patty_y_jitter: float = tunable(0.012)

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    fix_pos: tuple = info((0.35, 0.0))      # grill origin on the ground
    fix_yaw_nom_deg: float = info(0.0)
    # --- info: grill chamber (must match GrillSpawnerCfg) ----------------------------------------
    hx: float = info(0.13)
    hy: float = info(0.10)
    wall_t: float = info(0.010)
    wall_h: float = info(0.16)
    floor_t: float = info(0.010)
    grate_z: float = info(0.075)
    slat_t: float = info(0.012)
    slat_w: float = info(0.036)
    slat_xs: tuple = info((-0.09, -0.03, 0.03, 0.09))
    grill_mass: float = info(40.0)
    # --- info: catch / rails / towers / rest bar -------------------------------------------------
    catch_x0: float = info(0.150)
    catch_x1: float = info(0.185)
    catch_z0: float = info(0.204)
    catch_z1: float = info(0.220)
    post_y0: float = info(0.140)
    post_y1: float = info(0.166)
    rail_post_x: float = info(-0.145)
    rail_x0: float = info(-0.195)
    rail_x1: float = info(-0.085)
    rail_z0: float = info(0.196)
    rail_z1: float = info(0.208)
    bar_x: float = info(0.020)
    bar_z: float = info(0.398)
    bar_s: float = info(0.012)
    tower_y0: float = info(0.140)
    tower_y1: float = info(0.170)
    tower_h: float = info(0.42)
    # --- info: side board ------------------------------------------------------------------------
    board_hx: float = info(0.08)
    board_y0: float = info(0.18)
    board_y1: float = info(0.30)
    board_h: float = info(0.10)
    # --- info: lid + worn mount ------------------------------------------------------------------
    lid_mass: float = info(0.35)
    lid_com_x: float = info(0.14)
    plate_x0: float = info(0.02)
    plate_x1: float = info(0.27)
    lid_w: float = info(0.24)
    lid_t: float = info(0.012)
    cleat_x0: float = info(0.256)
    cleat_x1: float = info(0.268)
    cleat_h: float = info(0.014)
    handle_x: float = info(0.20)
    pin_x_rear: float = info(-0.18)
    pin_z: float = info(0.19)
    slide_s: float = info(0.08)
    pitch_max_deg: float = info(85.0)
    # --- info: patty -----------------------------------------------------------------------------
    patty_w: float = info(0.060)
    patty_t: float = info(0.034)
    patty_mass: float = info(0.10)
    patty_color: tuple = info((0.55, 0.30, 0.15))
    # --- info: materials / misc ------------------------------------------------------------------
    mu_static: float = info(0.60)           # grill compound material
    mu_dynamic: float = info(0.50)
    lid_mu_static: float = info(0.40)
    lid_mu_dynamic: float = info(0.35)
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.60)
    # rubric weights (0.15 + 0.20 + 0.30 = 0.65; non-success cap 0.70)
    w_released: float = info(0.15)
    w_raised: float = info(0.20)
    w_parked: float = info(0.30)

    # ----- derived (computed, not tuned) ---------------------------------------------------------
    @property
    def pin_x_fwd(self) -> float:
        """Grill-frame pin x at the forward end of the slide (spawn position)."""
        return self.pin_x_rear + self.slide_s

    @property
    def reach(self) -> float:
        """Plate radial extent from the pin."""
        return self.plate_x1

    @property
    def plate_top_closed(self) -> float:
        return self.pin_z + self.lid_t / 2

    @property
    def hover_gap(self) -> float:
        """Gap between the closed plate's underside and the chamber rim top."""
        return self.pin_z - self.lid_t / 2 - self.wall_h

    @property
    def rho_fwd(self) -> float:
        """Rest-bar centre radius from the FORWARD pin (must be inside the plate)."""
        return math.hypot(self.bar_x - self.pin_x_fwd, self.bar_z - self.pin_z)

    @property
    def rho_rear(self) -> float:
        """Rest-bar centre radius from the REAR pin (must clear the whole sweep)."""
        return math.hypot(self.bar_x - self.pin_x_rear, self.bar_z - self.pin_z)

    @property
    def theta_bar_deg(self) -> float:
        """Pitch at which the falling forward-slid plate face meets the bar centre."""
        return math.degrees(math.atan2(self.bar_z - self.pin_z,
                                       self.bar_x - self.pin_x_fwd))

    @property
    def max_sweep_r(self) -> float:
        """Largest radial extent of any lid point (plate tip corner / cleat corner)."""
        return max(math.hypot(self.plate_x1, self.lid_t / 2),
                   math.hypot(self.cleat_x1, self.lid_t / 2 + self.cleat_h))

    @property
    def jam_deg(self) -> float:
        """Pitch at which the spawn-position upswing hits the catch underside."""
        return math.degrees(math.asin(
            (self.catch_z0 - self.plate_top_closed) / self.reach))

    @property
    def exit_deg(self) -> float:
        """Pitch the tip would need to leave the catch's x-span (never reached)."""
        return math.degrees(math.acos(
            (self.catch_x0 - self.pin_x_fwd) / self.reach))

    @property
    def slide_park_lo(self) -> float:
        """Lower bound on the settled park slide after the bar seats on the cleat."""
        slack = self.cleat_x0 - (self.rho_fwd + self.bar_s / 2)
        return self.slide_s - slack / math.cos(math.radians(self.theta_bar_deg)) - 0.004

    def __post_init__(self) -> None:
        # --- pitch cap strictly under over-center: no joint-stop rest, gravity always
        # closes a free lid — an in-band settled pitch implies a real lean on the bar
        assert self.park_hi_deg < self.pitch_max_deg < 90.0, "pitch cap must be under 90"
        # --- catch gating at spawn: jam within a few degrees, engagement over jitter
        assert 0.5 < self.jam_deg < 6.0, f"catch jam angle off ({self.jam_deg:.1f} deg)"
        assert self.exit_deg > self.jam_deg + 8.0, "tip could exit the catch mid-jam"
        assert self.raised_deg > self.jam_deg + 50.0, "`raised` latch reachable while caught"
        tip_min = self.pin_x_fwd - self.slide_jitter + self.plate_x1
        assert tip_min >= self.catch_x0 + 0.008, "catch engagement lost under spawn jitter"
        assert self.pin_x_fwd + self.plate_x1 <= self.catch_x1 - 0.004, \
            "tip past the catch bar at spawn"
        # --- rear pin: the WHOLE lid sweep clears the rest bar (with contact offsets)
        bar_half_diag = self.bar_s * math.sqrt(2) / 2
        clear = (self.rho_rear - bar_half_diag) - (self.max_sweep_r + 2 * self.contact_offset)
        assert clear >= 0.005, f"rear sweep would clip the rest bar ({clear * 1000:.1f}mm)"
        # --- forward pin: the bar is well inside the plate, park angle inside the band
        assert self.plate_x0 + 0.02 < self.rho_fwd, "bar under the plate root"
        assert self.rho_fwd + self.bar_s / 2 <= self.cleat_x0 - 0.004, \
            "bar would land on the cleat, not the plate face"
        assert self.park_lo_deg + 5.0 <= self.theta_bar_deg <= self.park_hi_deg - 5.0, \
            f"park contact angle {self.theta_bar_deg:.1f} outside the band"
        assert self.pitch_max_deg >= self.theta_bar_deg + 15.0, "no fall gap onto the bar"
        # --- the seated lid creeps back onto the cleat (slide has no detent): the
        # bar's plate-normal reaction must beat pair-averaged friction
        mu_pair = (self.mu_static + self.lid_mu_static) / 2
        assert math.tan(math.radians(self.theta_bar_deg)) > 1.5 * mu_pair, \
            "friction could park the lid off the cleat"
        assert self.slide_park_lo > self.slide_fwd_min + 0.003, \
            "settled park slide could fail the forward clause"
        assert self.slide_rear_max < self.slide_fwd_min - 0.02, "slide clauses overlap"
        # --- the mouth is the only aperture: closed lid hover gap under-sizes the patty,
        # and a patty ON the closed lid rests far above the on-grate window
        assert self.hover_gap >= 0.006, "closed lid would rest on the rim (prop-masked stop)"
        assert self.hover_gap <= self.patty_t - 0.008, "patty could slip under the closed lid"
        assert self.plate_top_closed + self.patty_t / 2 > self.patty_z_hi + 0.05, \
            "a patty on the closed lid could pass the on-grate window"
        # --- grate: gaps under-size the patty; on-grate rest inside the z window
        edges = sorted(self.slat_xs)
        for a, b in zip(edges[:-1], edges[1:]):
            gap = (b - a) - self.slat_w
            assert gap <= self.patty_w - 0.020, "patty could fall through a slat gap"
        rest_z = self.grate_z + self.patty_t / 2
        assert self.patty_z_lo + 0.015 < rest_z < self.patty_z_hi - 0.015, \
            "on-grate rest outside the accepted z band"
        assert self.patty_z_hi < self.wall_h - 0.015, "z window pokes above the mouth"
        assert self.patty_x_tol < self.hx - self.wall_t - 0.010, "x window inside the cavity"
        assert self.patty_y_tol < self.hy - self.wall_t - 0.005, "y window inside the cavity"
        # --- drop corridor: with the lid parked (slide in its settled band) a straight
        # vertical column above the front grate clears the leaning plate and the walls
        worst_pin = self.pin_x_rear + min(self.slide_s, self.slide_park_lo + 0.010)
        th = math.radians(self.theta_bar_deg)
        plate_max_x = worst_pin + self.plate_x1 * math.cos(th) \
            + (self.lid_t / 2) * math.sin(th) + 2 * self.contact_offset
        corridor = (self.hx - self.wall_t) - plate_max_x
        assert corridor > self.patty_w + 0.020, \
            f"no patty drop corridor through the parked mouth ({corridor * 1000:.0f}mm)"
        # --- cleat clears the front wall top while sliding back closed
        cleat_bot = self.pin_z - self.lid_t / 2 - self.cleat_h
        assert cleat_bot >= self.wall_h + 0.008, "cleat would snag the rim on the slide"
        # --- towers / posts clear the lid width everywhere
        assert self.post_y0 >= self.lid_w / 2 + 0.008, "catch/rail posts inside the lid width"
        assert self.tower_y0 >= self.lid_w / 2 + 0.008, "towers inside the lid width"
        # --- board: patty spawn window (with yaw -> half-diagonal) stays on the board,
        # and the board clears the rest-bar towers
        half_diag = self.patty_w * math.sqrt(2) / 2
        assert self.patty_x_jitter + half_diag < self.board_hx - 0.005, "patty off the board (x)"
        by_half = (self.board_y1 - self.board_y0) / 2
        assert self.patty_y_jitter + half_diag < by_half - 0.004, "patty off the board (y)"
        assert self.board_y0 >= self.tower_y1 + 0.008, "board overlaps a tower"
        # --- board rest height differs from the on-grate window (board rest must fail)
        board_rest = self.board_h + self.patty_t / 2
        assert board_rest > self.patty_z_hi - 0.020 or self.board_y0 > self.patty_y_tol, \
            "a patty still on the board could satisfy the grate window"
        assert self.board_y0 > self.patty_y_tol + 0.05, "board inside the y window"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shift_park_grill")
class ShiftParkGrillScene(BaseScene):
    cfg: ShiftParkGrillSceneCfg

    def __init__(self, cfg: ShiftParkGrillSceneCfg | None = None) -> None:
        super().__init__(cfg or ShiftParkGrillSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        grill_spawn = cls["grill"](
            mass=c.grill_mass, hx=c.hx, hy=c.hy, wall_t=c.wall_t, wall_h=c.wall_h,
            floor_t=c.floor_t, grate_z=c.grate_z, slat_t=c.slat_t, slat_w=c.slat_w,
            slat_xs=c.slat_xs, catch_x0=c.catch_x0, catch_x1=c.catch_x1,
            catch_z0=c.catch_z0, catch_z1=c.catch_z1, post_y0=c.post_y0,
            post_y1=c.post_y1, rail_post_x=c.rail_post_x, rail_x0=c.rail_x0,
            rail_x1=c.rail_x1, rail_z0=c.rail_z0, rail_z1=c.rail_z1, bar_x=c.bar_x,
            bar_z=c.bar_z, bar_s=c.bar_s, tower_y0=c.tower_y0, tower_y1=c.tower_y1,
            tower_h=c.tower_h, board_hx=c.board_hx, board_y0=c.board_y0,
            board_y1=c.board_y1, board_h=c.board_h, mu_static=c.mu_static,
            mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)
        lid_spawn = cls["lid"](
            mass=c.lid_mass, com_x=c.lid_com_x, plate_x0=c.plate_x0,
            plate_x1=c.plate_x1, width=c.lid_w, t=c.lid_t, cleat_x0=c.cleat_x0,
            cleat_x1=c.cleat_x1, cleat_h=c.cleat_h, handle_x=c.handle_x,
            pin_x_rear=c.pin_x_rear, pin_z=c.pin_z, slide_s=c.slide_s,
            pitch_max_deg=c.pitch_max_deg, mu_static=c.lid_mu_static,
            mu_dynamic=c.lid_mu_dynamic, contact_offset=c.contact_offset)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu - 0.05,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "grill": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Grill",
                spawn=grill_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0], c.fix_pos[1], 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + c.pin_x_fwd, c.fix_pos[1], c.pin_z)),
            ),
            "patty": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Patty",
                spawn=sim_utils.CuboidCfg(
                    size=(c.patty_w, c.patty_w, c.patty_t),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.patty_color),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.patty_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0],
                         c.fix_pos[1] + (c.board_y0 + c.board_y1) / 2,
                         c.board_h + c.patty_t / 2 + 0.002)),
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
        self.grill: RigidObject = env.iscene["grill"]
        self.lid: RigidObject = env.iscene["lid"]
        self.patty: RigidObject = env.iscene["patty"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._released = torch.zeros(n, dtype=torch.bool, device=dev)
        self._raised = torch.zeros(n, dtype=torch.bool, device=dev)
        self._parked = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: grill heading + xy jitter; lid written CONSISTENTLY at the
        forward-closed spawn (slide jitter back from the stop, pitch 0); patty on the
        side board (grill-frame jitter + yaw); latches cleared. Grill and lid are one
        linkage — both root states are written back to back with no stepping."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for _ in range(4):  # burn post-seed draws (early Philox draws are seed-correlated)
            torch.rand(2 * m, device=dev)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, pos, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = q
            body.write_root_state_to_sim(st, env_ids)

        # --- grill: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.fix_yaw_nom_deg) + rnd(math.radians(c.grill_yaw_deg))
        q_g = _qz(yaw)
        gpos = torch.zeros(m, 3, device=dev)
        gpos[:, 0] = c.fix_pos[0] + rnd(c.grill_jitter)
        gpos[:, 1] = c.fix_pos[1] + rnd(c.grill_jitter)
        write(self.grill, gpos, q_g)

        # --- lid: forward-closed, tucked under the catch (one linkage with the grill) ---
        slide0 = c.slide_s - torch.rand(m, device=dev) * c.slide_jitter
        pin = torch.zeros(m, 3, device=dev)
        pin[:, 0] = c.pin_x_rear + slide0
        pin[:, 2] = c.pin_z
        write(self.lid, gpos + _qapply(q_g, pin), q_g.clone())

        # --- patty: on the side board, grill-frame jitter + yaw ---
        ploc = torch.zeros(m, 3, device=dev)
        ploc[:, 0] = rnd(c.patty_x_jitter)
        ploc[:, 1] = (c.board_y0 + c.board_y1) / 2 + rnd(c.patty_y_jitter)
        ploc[:, 2] = c.board_h + c.patty_t / 2 + 0.002
        write(self.patty, gpos + _qapply(q_g, ploc), _qmul(q_g, _qz(rnd(math.pi))))

        self._released[env_ids] = False
        self._raised[env_ids] = False
        self._parked[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "grill": self.grill.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "patty": self.patty.data.root_state_w[env_ids].clone(),
            "released": self._released[env_ids].clone(),
            "raised": self._raised[env_ids].clone(),
            "parked": self._parked[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.grill.write_root_state_to_sim(state["grill"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.patty.write_root_state_to_sim(state["patty"], env_ids)
        self._released[env_ids] = state["released"]
        self._raised[env_ids] = state["raised"]
        self._parked[env_ids] = state["parked"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A charcoal GRILL stands on the floor: a dark chamber "
            f"({2 * c.hx * 1000:.0f} x {2 * c.hy * 1000:.0f} mm) open at the TOP, with "
            f"a grey GRATE of four slats inside, red CATCH posts with an overhang bar "
            f"at its front, yellow slide-rail posts behind, two tall yellow TOWERS "
            f"carrying a thin red REST BAR high above the mouth, and a wooden side "
            f"BOARD holding a brown PATTY block. The steel LID (with a blue handle) "
            f"rides a worn 2-DOF mount: it can SLIDE {c.slide_s * 1000:.0f} mm "
            f"horizontally and PITCH up to {c.pitch_max_deg:.0f} deg — strictly short "
            f"of vertical, so gravity always drops a free lid shut. At spawn the lid "
            f"is closed at the FORWARD end of its slide with its front tip tucked "
            f"UNDER the red catch: swinging it up jams on the catch within a few "
            f"degrees. The grill's position and heading, the lid's exact slide spot "
            f"and the patty's place on the board all vary per episode.\n"
            f"Goal: OPEN the grill and load it. Slide the lid fully BACK (freeing the "
            f"tip from the catch), swing it up past {c.raised_deg:.0f} deg (the rest "
            f"bar clears the sweep only at the rear), slide it FORWARD again while "
            f"raised, and release: the lid drops onto the rest bar and PARKS leaning "
            f"at ~{c.theta_bar_deg:.0f} deg, held by gravity. Then move the patty "
            f"from the board down through the open top mouth onto the grate. To "
            f"count, the lid must rest in the {c.park_lo_deg:.0f}-"
            f"{c.park_hi_deg:.0f} deg band with its slide forward, the patty must "
            f"lie on the grate inside the chamber, and everything must be settled. "
            f"A lid released at the rear falls fully closed; a patty balanced on a "
            f"closed lid, left on the board or dropped outside does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Open the grill the long way: slide the steel lid back out from under "
            "the red catch, swing it up, slide it forward and let it rest leaning "
            "on the high red bar, then put the brown patty through the open top "
            "onto the grate."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _grill_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the grill body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.grill.data.root_quat_w),
                       pos_w - self.grill.data.root_pos_w)

    def slide(self) -> torch.Tensor:
        """(N,) lid slide readback in metres: 0 = rear stop, slide_s = forward stop."""
        return self._grill_local(self.lid.data.root_pos_w)[:, 0] - self.cfg.pin_x_rear

    def pitch(self) -> torch.Tensor:
        """(N,) lid pitch in RADIANS: 0 = closed flat, positive = tip raised."""
        q_rel = _qmul(_qinv(self.grill.data.root_quat_w), self.lid.data.root_quat_w)
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        d = _qapply(q_rel, ex)
        return torch.atan2(d[:, 2], d[:, 0])

    def patty_local(self) -> torch.Tensor:
        """(N,3) patty centre in the grill frame."""
        return self._grill_local(self.patty.data.root_pos_w)

    def upright(self) -> torch.Tensor:
        """(N,) bool: grill up-axis within the upright cone."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        u = _qapply(self.grill.data.root_quat_w, ez)
        return u[:, 2] > math.cos(math.radians(self.cfg.upright_deg))

    def parked(self) -> torch.Tensor:
        """(N,) bool: lid pitch in the park band with the slide forward."""
        c = self.cfg
        th = self.pitch()
        return (th > math.radians(c.park_lo_deg)) & (th < math.radians(c.park_hi_deg)) \
            & (self.slide() >= c.slide_fwd_min)

    def patty_on_grate(self) -> torch.Tensor:
        """(N,) bool: patty inside the chamber, resting in the grate z band."""
        c = self.cfg
        p = self.patty_local()
        return (p[:, 0].abs() < c.patty_x_tol) & (p[:, 1].abs() < c.patty_y_tol) \
            & (p[:, 2] > c.patty_z_lo) & (p[:, 2] < c.patty_z_hi)

    def settled(self) -> torch.Tensor:
        """(N,) bool: lid and patty slow (lin + lid angular)."""
        c = self.cfg
        v_lid = self.lid.data.root_lin_vel_w.norm(dim=-1)
        v_pat = self.patty.data.root_lin_vel_w.norm(dim=-1)
        w_lid = self.lid.data.root_ang_vel_w.norm(dim=-1)
        return (v_lid < c.settle_speed) & (v_pat < c.settle_speed) & (w_lid < 0.5)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in (self.grill, self.lid, self.patty)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite() & self.upright()
        th = self.pitch()
        self._released |= fin & (self.slide() <= c.slide_rear_max) \
            & (th < math.radians(c.released_low_deg))
        self._raised |= fin & (th >= math.radians(c.raised_deg))
        self._parked |= fin & self.parked() & self.settled()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: lid parked live on the rest bar (band + forward slide), patty on
        the grate, grill upright, everything settled and finite. The pitch cap is
        under over-center and the only free object is the patty itself, so a settled
        in-band pitch cannot be faked by a joint stop or a substitute prop."""
        self._update_latches()
        return self.parked() & self.patty_on_grate() & self.upright() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*released + 0.20*raised + 0.30*parked (latched;
        ~0 for doing nothing — the lid spawns forward-closed under the catch), capped
        at 0.70 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_released * self._released.float() + c.w_raised * self._raised.float()
                + c.w_parked * self._parked.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="shift_park_grill", robot="null"))
