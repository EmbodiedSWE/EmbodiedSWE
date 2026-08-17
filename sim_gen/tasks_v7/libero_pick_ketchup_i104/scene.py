"""RockerLockScene — load the ketchup into the tray of a gravity-biased rocker beam and
press the rocker's paddle: the tray tilts, the beam's own tip un-plugs the letterbox
window of a SEALED pantry bin, and the bottle rolls through the window into the bin
(sim_gen task `libero_pick_ketchup_i104`).

Derived from libero `pick_ketchup` ("pick up the ketchup and place it in the basket"),
but STRATEGICALLY different: the seed is identify-grasp-drop — pick the one named
bottle out of tabletop grocery clutter, carry it over an OPEN basket, release, and a
bounding-box containment check ends the episode. Here the receptacle is a fully SEALED
box: roof, four walls, and one letterbox window in the wall facing a rocker beam — and
that window is PLUGGED by the rocker's own tip at rest. The beam is a see-saw riding a
free journal (its axle rests in capped U-notches — pure contact, no joints) with a
built-in mass bias that parks it delivery-end-UP: in that pose the beam tip + its
hanging skirt fill the window to sub-bottle gaps (asserted in code: every static gap
around the closed gate is smaller than the bottle diameter), so NO amount of direct
carrying, dropping or pushing puts a bottle inside. The only way in is to OPERATE THE
MACHINE: lay the bottle into the rocker's walled tray at the low loading end, then
press the side paddle down and hold — the beam pitches onto its press stop, the tip
sinks to sill level (opening the window aperture), and the tray becomes a ramp that
gravity-ferries the bottle along the channel, over the tip and through the window.
Releasing the paddle lets the bias re-park the beam, sealing the bottle inside. The
task ends gate-CLOSED with the ketchup captive — a state a direct placement can never
produce. A same-shape YELLOW mustard bottle must stay outside (identification by
color, exclusion clause).

Strategy vs the seed and vs every reference task read for this construction
(pen_holder: multi-insert into an open cup; cellar_tow i43: build/unmake a peg-in-eye
tool coupling and tow; roll_in_garage i292: relocate a free blocking post, then
manually push-roll the target through a floor doorway): here the solver never pushes
or carries the target to the goal at all — after loading, every centimetre of the
bottle's travel is produced by the MACHINE the solver actuates from a handle a
forearm's length away from the goal. The plan is load-then-actuate-then-release
(operate a gated ferry), not declutter-then-push and not couple-then-tow; the code a
solver needs is a press-and-hold force schedule on a mechanism, not object transport
control. The seed's entire plan (carry the bottle over the receptacle and release)
lands it on the bin ROOF and scores nothing (constructed and rejected in smoke).

success(): the RED ketchup rests INSIDE the bin (rig-frame interior box, below the
window sill band), settled; the YELLOW mustard is NOT inside; the gate is CLOSED
(beam re-parked near its rest tilt and settled) with the beam's axle still seated in
its journal (machine intact); all states finite. score(): latched, non-decreasing
stages anchored in the demonstrated solve — 0.15 bottle loaded in the tray, +0.20
gate opened while loaded (beam pressed past the open threshold), +0.40 bottle
delivered into the bin, capped 0.75; exactly 1.0 iff success(). Null policy ~0.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - rig (one 40 kg dynamic fixture, teleported coherently at reset): two journal
    stands (capped U-notches), rest stop, press stop, and the sealed pantry bin
    (floor, roof, walls, letterbox window with sill/lintel/mullions).
  - beam (0.6 kg, authored center-of-mass bias): axle + end flanges riding the
    journal, walled tray channel, load-end stop wall, tip skirt (the gate plug), and
    a side press paddle.
  - ketchup (red) / mustard (yellow): identical cylinders, 5.5 cm dia x 14 cm.

Per-episode randomization (readback-verified in smoke): rig xy jitter + yaw, the two
bottles slot-SWAPPED between their floor spawn slots plus xy jitter + free yaw.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
# Rig frame: origin = journal (pivot) axis projected to the ground, +x toward the bin.
Z_PIV = 0.150  # journal axis height (= beam axle center at seat)
TILT_REST = 8.0  # rest tilt, deg, delivery end UP (mass bias + rest stop)
TILT_PRESS = 6.5  # full-press tilt, deg, delivery end DOWN (press stop)
AXLE_R = 0.012  # beam axle radius
AXLE_HL = 0.148  # axle half-length (along y)
FLANGE_R = 0.024  # axle end flange radius (axial retention)
FLANGE_T = 0.008
FLANGE_Y = 0.1415  # flange centers +/- y (inner faces +/-0.1375)
FLOOR_L = 0.46  # tray channel floor length
FLOOR_W = 0.175  # beam outer width
FLOOR_T = 0.012
FLOOR_CX = -0.070  # floor center x (beam frame) -> spans x [-0.30, +0.16]
FLOOR_TOP = 0.030  # floor top, beam frame z
TIP_X = 0.16  # beam tip (front faces of floor / rails / skirt)
RAIL_W = 0.010
RAIL_H = 0.040  # side rail height above the floor
CHAN_W = FLOOR_W - 2 * RAIL_W  # tray interior width = 0.155
ENDWALL_X = -0.294  # load-end stop wall center
SKIRT_T = 0.010
SKIRT_H = 0.070  # tip skirt (the gate plug below the floor lip)
BALLAST = (0.080, 0.140, 0.016)  # bias ballast under the load side
BALLAST_C = (-0.20, 0.0, 0.009)
PADDLE = (0.050, 0.060, 0.012)  # side press paddle (the handle)
PADDLE_C = (0.100, 0.1175, 0.064)
COM_BEAM = (-0.055, 0.0, 0.005)  # authored center of mass (the gravity bias)
STAND_Y = 0.1205  # journal stand centers +/- y
NOTCH_HW = 0.014  # U-notch half-width (2 mm play around the axle)
WALL_T = 0.012  # bin wall thickness
WIN_X = 0.175  # window plane (bin near-wall inner face)
SILL_Z = 0.140  # window sill top
LINTEL_Z = 0.240  # window lintel bottom
WIN_HW = 0.080  # window half-width (y)
BIN_X0, BIN_X1 = 0.187, 0.447  # bin interior x span
BIN_HY = 0.130  # bin interior half-width
BIN_ROOF = 0.254  # roof underside
REST_STOP_X = -0.260
PRESS_STOP_X = 0.119
BOT_R = 0.0275  # bottle radius (55 mm dia < ~80 mm parallel jaw)
BOT_L = 0.140  # bottle length


def _tip_floor_z(tilt_deg: float) -> float:
    """World z of the tray floor TOP at the beam tip, for a given tilt (deg, + = up)."""
    t = math.radians(tilt_deg)
    return Z_PIV + TIP_X * math.sin(t) + FLOOR_TOP * math.cos(t)


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
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


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, center, axis: str, color,
         contact_offset: float, material=None) -> None:
    """Author one colliding cylinder child prim."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr(axis)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _body_root(stage, prim_path: str, translation, orientation, mass: float,
               lin_damp: float, ang_damp: float, pos_iters: int = 16,
               vel_iters: int = 4, com=None):
    """Author the dynamic rigid-body root xform shared by the compound spawners.
    vel_iters=4 kills the GPU cylinder-creep artifact; `com` (optional) authors an
    explicit center of mass — MassAPI mass alone leaves the CoM at the body origin."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(pos_iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    return root


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the fixture as ONE heavy DYNAMIC compound body (teleportable at reset):
    journal stands with capped U-notches, rest/press stops, and the sealed bin with
    its letterbox window. Origin = pivot axis at ground, +x toward the bin."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    journal = _friction_material(stage, f"{prim_path}/journal_mat",
                                 cfg.mu_journal_s, cfg.mu_journal_d)
    binfloor = _friction_material(stage, f"{prim_path}/binfloor_mat",
                                  cfg.mu_binfloor_s, cfg.mu_binfloor_d)
    gray = (0.45, 0.48, 0.50)
    dark = (0.28, 0.30, 0.34)
    blue = (0.20, 0.35, 0.60)

    # journal stands: post (notch floor on top), two U-notch walls, retaining cap
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/post_{tag}", (0.060, 0.030, Z_PIV - AXLE_R),
             (0.0, sy * STAND_Y, (Z_PIV - AXLE_R) / 2), gray, 0.0012, material=journal)
        for wt, sx in (("f", 1.0), ("b", -1.0)):
            _box(stage, f"{prim_path}/notch_{tag}{wt}", (0.016, 0.030, 0.034),
                 (sx * (NOTCH_HW + 0.008), sy * STAND_Y, Z_PIV - AXLE_R + 0.017),
                 gray, 0.0012, material=journal)
        _box(stage, f"{prim_path}/cap_{tag}", (0.060, 0.030, 0.008),
             (0.0, sy * STAND_Y, Z_PIV + AXLE_R + 0.002 + 0.004), gray, 0.0012,
             material=journal)

    # rest / press stops (under the beam floor line, contact at the design tilts)
    t = math.radians(TILT_REST)
    rest_top = Z_PIV + REST_STOP_X * math.sin(t) + (FLOOR_TOP - FLOOR_T) * math.cos(t)
    _box(stage, f"{prim_path}/rest_stop", (0.030, 0.100, rest_top),
         (REST_STOP_X, 0.0, rest_top / 2), dark, 0.0012, material=body)
    p = math.radians(TILT_PRESS)
    press_top = Z_PIV - PRESS_STOP_X * math.sin(p) + (FLOOR_TOP - FLOOR_T) * math.cos(p)
    _box(stage, f"{prim_path}/press_stop", (0.030, 0.100, press_top),
         (PRESS_STOP_X, 0.0, press_top / 2), dark, 0.0012, material=body)

    # sealed bin: floor, near wall (sill band / mullions / lintel band), sides, far, roof
    bw = 2 * (BIN_HY + WALL_T)  # full outer width = 0.284
    _box(stage, f"{prim_path}/bin_floor", (BIN_X1 - BIN_X0, 2 * BIN_HY, 0.012),
         ((BIN_X0 + BIN_X1) / 2, 0.0, 0.006), blue, 0.0015, material=binfloor)
    wx = WIN_X + WALL_T / 2  # near wall center x
    _box(stage, f"{prim_path}/wall_sill", (WALL_T, bw, SILL_Z),
         (wx, 0.0, SILL_Z / 2), blue, 0.0015, material=body)
    mull_w = BIN_HY + WALL_T - WIN_HW
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/mullion_{tag}", (WALL_T, mull_w, LINTEL_Z - SILL_Z),
             (wx, sy * (WIN_HW + mull_w / 2), (SILL_Z + LINTEL_Z) / 2), blue, 0.0015,
             material=body)
    _box(stage, f"{prim_path}/wall_lintel", (WALL_T, bw, BIN_ROOF - LINTEL_Z),
         (wx, 0.0, (LINTEL_Z + BIN_ROOF) / 2), blue, 0.0015, material=body)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_side_{tag}", (0.284, WALL_T, BIN_ROOF),
             (WIN_X + 0.142, sy * (BIN_HY + WALL_T / 2), BIN_ROOF / 2), blue, 0.0015,
             material=body)
    _box(stage, f"{prim_path}/wall_far", (WALL_T, bw, BIN_ROOF),
         (BIN_X1 + WALL_T / 2, 0.0, BIN_ROOF / 2), blue, 0.0015, material=body)
    _box(stage, f"{prim_path}/roof", (0.296, 0.296, 0.012),
         (WIN_X + 0.144, 0.0, BIN_ROOF + 0.006), dark, 0.0015, material=body)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rocker beam: axle + end flanges (journal riders), walled tray
    channel, load-end stop wall, tip skirt (gate plug), ballast, side paddle.
    Origin = axle center; +x = delivery (tip) direction."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.05, 0.15, com=COM_BEAM)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    journal = _friction_material(stage, f"{prim_path}/journal_mat",
                                 cfg.mu_journal_s, cfg.mu_journal_d)
    yellow = (0.85, 0.70, 0.15)
    steel = (0.55, 0.57, 0.60)
    green = (0.10, 0.60, 0.20)

    _cyl(stage, f"{prim_path}/axle", AXLE_R, 2 * AXLE_HL, (0.0, 0.0, 0.0), "Y", steel,
         0.0012, material=journal)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _cyl(stage, f"{prim_path}/flange_{tag}", FLANGE_R, FLANGE_T,
             (0.0, sy * FLANGE_Y, 0.0), "Y", steel, 0.0012, material=journal)
    _box(stage, f"{prim_path}/floor", (FLOOR_L, FLOOR_W, FLOOR_T),
         (FLOOR_CX, 0.0, FLOOR_TOP - FLOOR_T / 2), yellow, 0.0015, material=body)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/rail_{tag}", (FLOOR_L, RAIL_W, RAIL_H),
             (FLOOR_CX, sy * (CHAN_W / 2 + RAIL_W / 2), FLOOR_TOP + RAIL_H / 2),
             yellow, 0.0015, material=body)
    _box(stage, f"{prim_path}/end_wall", (0.012, CHAN_W, 0.045),
         (ENDWALL_X, 0.0, FLOOR_TOP + 0.0225), yellow, 0.0015, material=body)
    _box(stage, f"{prim_path}/skirt", (SKIRT_T, FLOOR_W, SKIRT_H),
         (TIP_X - SKIRT_T / 2, 0.0, FLOOR_TOP - FLOOR_T - SKIRT_H / 2), yellow,
         0.0015, material=body)
    _box(stage, f"{prim_path}/ballast", BALLAST, BALLAST_C, steel, 0.0015,
         material=body)
    _box(stage, f"{prim_path}/paddle", PADDLE, PADDLE_C, green, 0.0015, material=body)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            mu_body_s: float = 0.40
            mu_body_d: float = 0.35
            mu_journal_s: float = 0.15
            mu_journal_d: float = 0.12
            mu_binfloor_s: float = 0.60
            mu_binfloor_d: float = 0.50

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            mu_body_s: float = 0.30
            mu_body_d: float = 0.25
            mu_journal_s: float = 0.15
            mu_journal_d: float = 0.12

        _SPAWNER_CACHE.update(rig=RigSpawnerCfg, beam=BeamSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RockerLockSceneCfg(BaseCfg):
    """Config for `RockerLockScene`. The interlock is asserted in `__post_init__`:
    every static gap around the CLOSED gate (rest pose) is smaller than the bottle
    diameter, while the OPEN gate (full press) passes the bottle with real margin —
    the machine is the only way into the bin."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    closed_min_deg: float = tunable(3.0)  # "gate closed": beam tilt at least this (up)
    open_deg: float = tunable(4.0)  # score: "gate opened" once tilt <= -this while loaded
    seat_tol: float = tunable(0.020)  # axle center within this of the journal axis (m)
    bin_z_max: float = tunable(0.140)  # "resting inside": bottle center below the sill band
    mus_out_margin: float = tunable(0.010)  # mustard exclusion margin around the bin box

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    rig_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the rig (m)
    rig_yaw_max: float = tunable(12.0)  # uniform +/- rig yaw (deg)
    slot_x: float = tunable(-0.55)  # bottle spawn slots (rig frame)
    slot_y: float = tunable(0.20)
    slot_jitter: float = tunable(0.05)  # uniform +/- xy jitter per bottle

    # --- info: structure -----------------------------------------------------------------------
    bot_r: float = info(BOT_R)
    bot_l: float = info(BOT_L)
    bot_mass: float = info(0.15)
    rig_mass: float = info(40.0)  # heavy dynamic fixture (teleportable, immovable)
    beam_mass: float = info(0.60)
    mu_bot_s: float = info(0.35)
    mu_bot_d: float = info(0.30)
    mu_ground_s: float = info(0.50)
    mu_ground_d: float = info(0.40)

    # Derived (filled in __post_init__).
    rest_gap: float = field(default=None, init=False)  # closed-gate top aperture
    open_gap: float = field(default=None, init=False)  # open-gate top aperture

    def __post_init__(self) -> None:
        bot_d = 2 * BOT_R
        tip_rest = _tip_floor_z(TILT_REST)
        tip_press = _tip_floor_z(-TILT_PRESS)
        self.rest_gap = LINTEL_Z - tip_rest
        self.open_gap = LINTEL_Z - tip_press
        # -- embodiment: bottle body and paddle both fit an ~80 mm parallel jaw --
        assert bot_d < 0.08, "bottle must fit an ~80 mm parallel jaw"
        assert PADDLE[2] < 0.08, "paddle plate must fit the jaw"
        # -- the interlock: CLOSED gate blocks the bottle everywhere (allow 2 mm journal slop) --
        assert self.rest_gap + 0.004 < bot_d - 0.010, \
            "closed gate: aperture above the tip must undercut the bottle diameter"
        t = math.radians(TILT_REST)
        skirt_bot = Z_PIV + (TIP_X - SKIRT_T) * math.sin(t) \
            + (FLOOR_TOP - FLOOR_T - SKIRT_H) * math.cos(t)
        assert skirt_bot < SILL_Z - 0.010, "closed gate: skirt must overlap the sill band"
        assert WIN_X - TIP_X < bot_d - 0.010, \
            "the beam-to-wall standoff must never pass a bottle"
        assert FLOOR_W > 2 * WIN_HW + 0.010, "the beam face must mask the window sideways"
        rails_top_rest = Z_PIV + TIP_X * math.sin(t) + (FLOOR_TOP + RAIL_H) * math.cos(t)
        assert rails_top_rest > LINTEL_Z - 0.002, \
            "closed gate: the rails must overtop the lintel (no over-the-tip path)"
        # -- the OPEN gate passes the bottle with real margin --
        assert self.open_gap > bot_d + 0.020, "open gate: aperture must pass the bottle"
        assert tip_press > SILL_Z + 0.014, "open gate: tip must stay above the sill"
        assert LINTEL_Z - SILL_Z > bot_d + 0.030, "window must pass the bottle"
        assert 2 * WIN_HW > BOT_L + 0.015, "window width must pass the lying bottle"
        assert CHAN_W > BOT_L + 0.010, "tray interior must accept the lying bottle"
        p = math.radians(TILT_PRESS)
        skirt_bot_press = Z_PIV - TIP_X * math.sin(p) \
            + (FLOOR_TOP - FLOOR_T - SKIRT_H) * math.cos(p)
        assert skirt_bot_press > 0.010, "pressed skirt must clear the ground"
        # -- journal geometry: axle seated with play, capped, axially retained --
        assert NOTCH_HW > AXLE_R + 0.001, "notch must leave play around the axle"
        assert FLANGE_Y - FLANGE_T / 2 > STAND_Y + 0.015, \
            "flange inner faces must bracket the stand outer faces"
        # -- spawn slots clear of beam sweep and bin --
        assert self.slot_x + self.slot_jitter < ENDWALL_X - 0.10, \
            "bottle spawn slots must stay clear of the beam"
        # -- null policy scores 0: bottles spawn far outside the bin --
        assert self.slot_x + self.slot_jitter < 0.0 < BIN_X0


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("rocker_lock")
class RockerLockScene(BaseScene):
    cfg: RockerLockSceneCfg

    def __init__(self, cfg: RockerLockSceneCfg | None = None) -> None:
        super().__init__(cfg or RockerLockSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        bot_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_bot_s, dynamic_friction=c.mu_bot_d, restitution=0.0)
        bot_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.25, angular_damping=0.15)
        bot_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=spawners["rig"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.rig_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=spawners["beam"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, Z_PIV)),
            ),
        }
        for name, color, y0 in (("ketchup", (0.75, 0.10, 0.08), 0.20),
                                ("mustard", (0.85, 0.75, 0.10), -0.20)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CylinderCfg(
                    radius=BOT_R, height=BOT_L, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=bot_mat, rigid_props=bot_rigid,
                    collision_props=bot_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bot_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.55, y0, BOT_L / 2 + 0.002)),
            )
        return out

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
        self.rig: RigidObject = env.iscene["rig"]
        self.beam: RigidObject = env.iscene["beam"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latched-score state (zeroed at reset, kept in get_state/set_state)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + yaw the rig, seat the beam on its journal at the
        rest tilt, and stand the two bottles on SWAPPED floor slots with jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- rig: xy jitter + yaw ---
        rxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rig_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_max)
        q_rig = _qz(ryaw)
        zcol = torch.zeros(m, 1, device=dev)
        rig_pos = torch.cat([rxy, zcol], dim=-1)
        write(self.rig, rig_pos, q_rig)

        # --- beam: seated on the journal, a hair shy of the rest stop (settles onto it) ---
        piv = rig_pos + _qapply(q_rig, torch.tensor([0.0, 0.0, Z_PIV],
                                                    device=dev).expand(m, 3))
        tilt = torch.full((m,), -math.radians(TILT_REST - 0.4), device=dev)
        write(self.beam, piv, _qmul(q_rig, _qy(tilt)))

        # --- bottles: slot-swapped floor spawns (torch.rand — not the degenerate
        #     first-randint draw), xy jitter, free yaw, standing upright ---
        swap = torch.rand(m, device=dev) > 0.5
        sign = torch.where(swap, -torch.ones(m, device=dev), torch.ones(m, device=dev))
        for body, s in ((self.ketchup, sign), (self.mustard, -sign)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = c.slot_x
            loc[:, 1] = c.slot_y * s
            loc[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            loc[:, 2] = BOT_L / 2 + 0.002
            pos = rig_pos.clone()
            pos[:, 0:2] += _qapply(q_rig, loc)[:, 0:2]
            pos[:, 2] = loc[:, 2]
            write(body, pos, _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        for buf in (self._loaded, self._opened, self._delivered):
            buf[env_ids] = False

    # ----- state (full, restorable — includes the latched score state) ------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
              for nm in ("rig", "beam", "ketchup", "mustard")}
        st["_latch"] = torch.stack([
            self._loaded[env_ids].float(), self._opened[env_ids].float(),
            self._delivered[env_ids].float()], dim=-1).clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("rig", "beam", "ketchup", "mustard"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        if "_latch" in state:
            lt = state["_latch"]
            self._loaded[env_ids] = lt[:, 0] > 0.5
            self._opened[env_ids] = lt[:, 1] > 0.5
            self._delivered[env_ids] = lt[:, 2] > 0.5

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A sealed blue pantry BIN stands on the floor: four walls and a flat "
            f"roof, no open top. Its only opening is a letterbox WINDOW "
            f"({2 * WIN_HW * 100:.0f} cm wide, sill {SILL_Z * 100:.0f} cm up, lintel "
            f"{LINTEL_Z * 100:.0f} cm up) in the wall facing a yellow ROCKER BEAM — "
            f"a see-saw whose steel axle rides in two capped journal stands. The "
            f"beam carries a walled TRAY channel along its top; a mass bias parks it "
            f"with the tray's far (window) end raised, and in that rest pose the "
            f"beam's tip and hanging skirt PLUG the window: every gap around the "
            f"closed gate is smaller than a bottle ({self.cfg.rest_gap * 100:.1f} cm "
            f"top aperture vs {2 * BOT_R * 100:.1f} cm bottle), so nothing can be "
            f"put into the bin directly, from any side, in any orientation. A green "
            f"PRESS PADDLE sticks out sideways from the beam near its raised end. "
            f"Pressing the paddle down (about {25:.0f} N) pitches the beam onto its "
            f"press stop: the tip sinks to just above the sill, the window aperture "
            f"opens to {self.cfg.open_gap * 100:.1f} cm, and the tray becomes a ramp "
            f"sloping into the window. On the open floor behind the tray's low "
            f"loading end stand two visually identical bottles "
            f"({2 * BOT_R * 100:.1f} cm across, {BOT_L * 100:.0f} cm tall): a RED "
            f"ketchup and a YELLOW mustard, on randomly swapped spots — tell them "
            f"apart by color. Rig pose and bottle spots change every episode.\n"
            f"Goal: deliver the RED ketchup bottle INTO the bin, and leave the gate "
            f"closed behind it. Lay the ketchup on its side ACROSS the tray at the "
            f"low loading end (it rests against the end wall), then press the paddle "
            f"down and HOLD: the bottle rolls down the tilted tray, over the tip and "
            f"through the window into the bin. Release the paddle so the beam "
            f"re-parks and seals the window (the task only counts with the beam "
            f"settled back within {c.closed_min_deg:.0f} degrees short of its rest "
            f"tilt, i.e. gate closed, and the axle still seated in its journal). The "
            f"YELLOW mustard must stay OUTSIDE the bin. Success is judged on the "
            f"settled end state: red bottle resting inside, gate closed, machine "
            f"intact, mustard out."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the red ketchup bottle sideways into the rocker's tray at the low "
            "end, press the green paddle down until the bottle rolls through the "
            "window into the sealed bin, then release so the gate closes. Keep the "
            "yellow mustard out of the bin."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> the rig's frame."""
        return _qapply(_qinv(self.rig.data.root_quat_w),
                       pos_w - self.rig.data.root_pos_w)

    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.beam.data.root_quat_w),
                       pos_w - self.beam.data.root_pos_w)

    def beam_tilt_deg(self) -> torch.Tensor:
        """(N,) beam tilt in deg, + = delivery (tip) end UP (rest ~ +8, press ~ -6.5)."""
        ex = _qapply(self.beam.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0],
                                  device=self.env.device).expand(self.env.num_envs, 3))
        return torch.rad2deg(torch.asin(ex[:, 2].clamp(-1.0, 1.0)))

    def axle_seated(self) -> torch.Tensor:
        """(N,) bool: beam axle center within seat_tol of the journal axis (machine
        intact — the beam has not been lifted out of or knocked off its stands)."""
        loc = self._rig_local(self.beam.data.root_pos_w)
        loc = loc - torch.tensor([0.0, 0.0, Z_PIV], device=loc.device)
        return loc.norm(dim=-1) <= self.cfg.seat_tol

    def gate_closed(self) -> torch.Tensor:
        """(N,) bool: beam re-parked near its rest tilt (window plugged)."""
        return self.beam_tilt_deg() >= (TILT_REST - self.cfg.closed_min_deg - 2.0)

    def in_tray(self, body) -> torch.Tensor:
        """(N,) bool: bottle center inside the tray channel (beam frame)."""
        loc = self._beam_local(body.data.root_pos_w)
        return (loc[:, 0] - FLOOR_CX).abs().le(FLOOR_L / 2) \
            & loc[:, 1].abs().le(CHAN_W / 2) \
            & (loc[:, 2] >= FLOOR_TOP + 0.005) & (loc[:, 2] <= FLOOR_TOP + 0.090)

    def in_bin(self, body, z_max: float | None = None) -> torch.Tensor:
        """(N,) bool: bottle center inside the bin interior box (rig frame). With
        z_max=None the whole interior volume counts (the delivery latch); success
        uses cfg.bin_z_max (resting below the sill band)."""
        loc = self._rig_local(body.data.root_pos_w)
        zm = self.cfg.bin_z_max if z_max is None else z_max
        return (loc[:, 0] >= BIN_X0 + 0.003) & (loc[:, 0] <= BIN_X1 - 0.003) \
            & loc[:, 1].abs().le(BIN_HY - 0.005) \
            & (loc[:, 2] >= 0.012) & (loc[:, 2] <= zm)

    def mustard_out(self) -> torch.Tensor:
        """(N,) bool: mustard center OUTSIDE the (margin-grown) bin interior box."""
        c = self.cfg
        loc = self._rig_local(self.mustard.data.root_pos_w)
        inside = (loc[:, 0] >= BIN_X0 - c.mus_out_margin) \
            & (loc[:, 0] <= BIN_X1 + c.mus_out_margin) \
            & loc[:, 1].abs().le(BIN_HY + c.mus_out_margin) \
            & (loc[:, 2] >= 0.0) & (loc[:, 2] <= BIN_ROOF)
        return ~inside

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for nm in ("rig", "beam", "ketchup", "mustard"):
            ok &= getattr(self, nm).data.root_state_w.isfinite().all(dim=-1)
        return ok

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: RED ketchup resting inside the bin (below the sill band) and
        settled; gate CLOSED (beam near rest tilt, settled, axle seated — the machine
        re-parked and intact); YELLOW mustard outside; all states finite."""
        return self.in_bin(self.ketchup) & self.settled(self.ketchup) \
            & self.gate_closed() & self.settled(self.beam) & self.axle_seated() \
            & self.mustard_out() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched and non-decreasing along the demonstrated
        solution: 0.15 ketchup loaded in the tray; +0.20 gate opened (beam pressed
        past -open_deg) while loaded; +0.40 ketchup delivered into the bin volume;
        1.0 iff success(). The null policy holds ~0 (nothing spawns loaded)."""
        c = self.cfg
        self._loaded |= self.in_tray(self.ketchup)
        self._opened |= self._loaded & (self.beam_tilt_deg() <= -c.open_deg)
        self._delivered |= self.in_bin(self.ketchup, z_max=BIN_ROOF - 0.010)
        base = 0.15 * self._loaded.float() + 0.20 * self._opened.float() \
            + 0.40 * self._delivered.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="rocker_lock", robot="null", env_spacing=3.0))
