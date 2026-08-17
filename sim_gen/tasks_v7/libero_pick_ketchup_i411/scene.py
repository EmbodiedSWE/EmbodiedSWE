"""BasculeKeepScene — un-ballast the counterweighted bascule bridge so gravity lowers it
across the moat, then push the ketchup over the deployed deck through the keep's letterbox
doorway (sim_gen task `libero_pick_ketchup_i411`).

Derived from libero `pick_ketchup` ("pick up the ketchup and place it in the basket"), but
STRATEGICALLY different: the seed is identify-grasp-drop — pick the one named bottle out of
grocery clutter, carry it over an OPEN basket, release; a bounding-box containment check
ends the episode. Here the receptacle is a raised, fully ROOFED keep whose only opening is
a letterbox doorway 13.5 cm above the ground, fronted by an open MOAT: there is no surface
in front of the doorway, so nothing can be stood, slid, or quasi-statically pushed into it
— the seed's carry-and-drop lands the bottle on the keep ROOF (constructed and rejected in
smoke), and a ground-level bottle faces a sheer 13.5 cm wall. The only path is a BASCULE
BRIDGE: a see-saw leaf hinged on a trunnion pillar, whose far tongue, when level, rests on
the doorway sill and completes a road from hinge to keep. The leaf is authored tip-heavy
(it WANTS to fall level), but at spawn it is held RAISED (~35 deg, heel on the ground) by
two steel COUNTERWEIGHT blocks sitting in a walled ballast tray on its tail. The
actuation is SUBTRACTIVE: the solver never presses, rotates, or holds the machine — it
REMOVES the two blocks from the tray (a plain pick-and-carry each), and gravity deploys
the bridge on its own. One block still holds the bridge up (asserted torque margins), so
BOTH must go. With the bridge deployed the bottle is placed on the deck and pushed across
the tongue through the doorway into the keep. Raised, the machine is its own interlock:
the tilted deck ends short of the doorway with the tongue-to-lintel aperture smaller than
the bottle (asserted), and a bottle lying across the deck cannot pass the doorway anyway
(mouth narrower than the bottle length). A same-shape YELLOW mustard bottle on a randomly
swapped spawn spot must stay out (identification by color, exclusion clause).

Strategy vs the seed and vs every reference task read for this construction:
- rocker_lock (i104, same seed): press-and-HOLD a paddle so a see-saw ferry
  gravity-delivers the loaded bottle through a plugged window, then release to re-seal;
  end state gate CLOSED, delivery produced by the machine. Here the machine is actuated
  by REMOVING MASS (no force is ever applied to the mechanism), the end state is the
  machine DEPLOYED (bridge down and staying down), and delivery is a manual push ACROSS
  the machine — the bridge is road, not ferry.
- moat_bridge (i192) / moat_causeway (i19): lay a loose free plank so it settles spanning
  two rims, then roll/push cargo over. Here nothing is placed to build the span: the
  bridge is a jointed machine already in the scene, and the construction act is a
  counterweight REMOVAL that changes its equilibrium.
- drawbridge_vault (i332): bridge starts LOWERED as a free ramp (entry needs no machine
  work), cargo is pushed UP it, and the machine act is CLOSING the door past vertical
  after entry (end state sealed). Here the bridge starts RAISED and held by ballast, the
  machine act is un-ballasting BEFORE any approach exists, the crossing is level, and
  success REQUIRES the bridge to remain deployed — the opposite interlock direction and
  end state.

success(): RED ketchup resting INSIDE the keep interior (rig-frame box, past the doorway),
settled; bridge DEPLOYED (leaf within a few deg of level, settled, hinge intact) with the
ballast tray EMPTY (both blocks out — physics already forbids level+settled with ballast
aboard, asserted margins; the clause additionally kills kinematic-hold fakes); YELLOW
mustard outside the keep; all states finite. score(): latched, non-decreasing stages
anchored in the demonstrated solve — 0.10 first block out of the tray, +0.10 tray empty,
+0.25 bridge deployed (gated on tray empty), +0.25 bottle past the doorway plane (gated on
deployed); exactly 1.0 iff success(). Null policy ~0.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - rig (one 40 kg dynamic fixture, teleported coherently at reset): the keep — base
    pedestal (top = doorway sill), raised interior floor, mullions + lintel forming the
    letterbox doorway, side walls, far wall, roof.
  - pillar (20 kg): trunnion posts + ground base; the leaf hinges on it via a spawn-
    authored revolute joint (axis y). Joint pairs are collision-filtered, so every
    surface the leaf must TOUCH (ground heel rest, doorway sill) lives on OTHER bodies.
  - leaf (1.0 kg, authored tip-heavy CoM): deck slab + side curbs + narrowed tongue +
    walled ballast tray on the tail.
  - block_a / block_b: 5 x 5 x 3 cm steel counterweights, 0.8 kg each (jaw-sized).
  - ketchup (red) / mustard (yellow): identical 5.5 x 14 cm cylinders.

Per-episode randomization (readback-verified in smoke): rig/pillar/leaf xy jitter + yaw
(teleported coherently), the two bottles slot-SWAPPED between ground spawn slots plus xy
jitter + free yaw, block y-jitter in the tray. Heavy imports (isaaclab, pxr) are deferred
so importing this module — and registering the scene — stays app-free.
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
# Rig frame: origin = hinge axis projected to the ground, +x toward the keep.
PIV_Z = 0.143  # trunnion (hinge) axis height = deck mid-plane when level
DECK_T = 0.014  # deck slab thickness (top at leaf z +0.007 -> world 0.150 level)
DECK_X0 = -0.240  # deck tail (heel) end, leaf frame
DECK_X1 = 0.185  # main slab far end (5 mm short of the keep wall when level)
DECK_W = 0.160  # main slab width
TONGUE_X1 = 0.220  # tongue tip (rests over the doorway sill when level)
TONGUE_W = 0.070  # tongue width (fits through the 80 mm doorway)
CURB_H = 0.020  # side curb height above the deck top
CURB_W = 0.020
CHAN_W = DECK_W - 2 * CURB_W  # push channel interior = 0.120
TRAY_X0, TRAY_X1 = -0.240, -0.150  # ballast tray span (walls at both ends)
TRAY_WALL_T = 0.012
TRAY_WALL_H = 0.045  # tray wall height above the deck top
COM_LEAF = (0.050, 0.0, 0.0)  # authored tip-heavy CoM: the leaf falls level on its own
LEAF_M = 1.0
BLOCK = (0.050, 0.050, 0.030)  # counterweight block (jaw-sized)
BLOCK_M = 0.80
PILLAR_M = 20.0
POST_Y0 = 0.085  # post inner faces (5 mm clear of the deck sides)
POST_T = 0.030
POST_TOP = 0.155
WALL_X0, WALL_X1 = 0.190, 0.232  # keep front wall outer / inner faces
SILL_TOP = 0.135  # pedestal top = doorway sill (the tongue rests on its front strip)
MOUTH_HW = 0.040  # doorway half-width
LINTEL_BOT = 0.313  # doorway top
FLOOR_X0, FLOOR_X1 = 0.225, 0.465  # raised interior floor slab span
FLOOR_TOP = 0.148  # interior floor top (2 mm BELOW the tongue top: transit steps down)
COURT_Y = 0.115  # interior half-width
ROOF_BOT = 0.325  # interior ceiling
KEEP_X1 = 0.477  # keep outer far face
KEEP_HY = 0.127  # keep outer half-width
BOT_R = 0.0275  # bottle radius (55 mm dia < ~80 mm parallel jaw)
BOT_L = 0.140  # bottle length


def _theta_up() -> float:
    """Raised rest tilt (rad): heel bottom corner (DECK_X0, -DECK_T/2) on the ground."""
    th = 0.5
    for _ in range(60):
        th = math.asin((PIV_Z - (DECK_T / 2) * math.cos(th)) / -DECK_X0)
    return th


THETA_UP = _theta_up()  # ~0.609 rad ~ 34.9 deg


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force through
    unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force comes out as
    the desired world force. Callers PROBE which mode moves the body the right way and
    lock it in (`q_ref` = readback at the reference instant)."""
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
    """Author the KEEP as ONE heavy DYNAMIC compound body (teleportable at reset):
    pedestal base (top = doorway sill), raised interior floor, doorway mullions +
    lintel, side walls, far wall, roof. Origin = hinge axis at ground, +x toward
    the keep."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    floor = _friction_material(stage, f"{prim_path}/floor_mat",
                               cfg.mu_floor_s, cfg.mu_floor_d)
    stone = (0.52, 0.50, 0.46)
    dark = (0.30, 0.30, 0.34)
    blue = (0.22, 0.36, 0.60)

    xk = (WALL_X0 + KEEP_X1) / 2  # keep footprint center x
    lk = KEEP_X1 - WALL_X0  # keep footprint length
    # pedestal base: solid up to the sill; its front top strip is the tongue's porch
    _box(stage, f"{prim_path}/base", (lk, 2 * KEEP_HY, SILL_TOP),
         (xk, 0.0, SILL_TOP / 2), stone, 0.0015, material=body)
    # raised interior floor slab (top 2 mm below the tongue top)
    _box(stage, f"{prim_path}/floor",
         (FLOOR_X1 - FLOOR_X0, 2 * COURT_Y, FLOOR_TOP - SILL_TOP),
         ((FLOOR_X0 + FLOOR_X1) / 2, 0.0, (SILL_TOP + FLOOR_TOP) / 2), blue,
         0.0015, material=floor)
    # doorway mullions (front wall beside the mouth) + lintel band above it
    mull_w = KEEP_HY - MOUTH_HW
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/mullion_{tag}",
             (WALL_X1 - WALL_X0, mull_w, LINTEL_BOT - SILL_TOP),
             ((WALL_X0 + WALL_X1) / 2, sy * (MOUTH_HW + mull_w / 2),
              (SILL_TOP + LINTEL_BOT) / 2), blue, 0.0015, material=body)
    _box(stage, f"{prim_path}/lintel",
         (WALL_X1 - WALL_X0, 2 * KEEP_HY, ROOF_BOT - LINTEL_BOT),
         ((WALL_X0 + WALL_X1) / 2, 0.0, (LINTEL_BOT + ROOF_BOT) / 2), blue,
         0.0015, material=body)
    # side walls, far wall, roof
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/wall_side_{tag}",
             (KEEP_X1 - FLOOR_X0, KEEP_HY - COURT_Y, ROOF_BOT - SILL_TOP),
             ((FLOOR_X0 + KEEP_X1) / 2, sy * (COURT_Y + (KEEP_HY - COURT_Y) / 2),
              (SILL_TOP + ROOF_BOT) / 2), blue, 0.0015, material=body)
    _box(stage, f"{prim_path}/wall_far",
         (KEEP_X1 - FLOOR_X1, 2 * COURT_Y, ROOF_BOT - SILL_TOP),
         ((FLOOR_X1 + KEEP_X1) / 2, 0.0, (SILL_TOP + ROOF_BOT) / 2), blue,
         0.0015, material=body)
    _box(stage, f"{prim_path}/roof", (lk + 0.024, 2 * KEEP_HY + 0.024, 0.012),
         (xk, 0.0, ROOF_BOT + 0.006), dark, 0.0015, material=body)
    return root


def _spawn_pillar(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the trunnion pillar: two posts flanking the deck + a ground base slab.
    The leaf hinges on this body (joint authored in the leaf spawner); the joint pair
    is collision-filtered, so no leaf contact surface lives here."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    gray = (0.45, 0.48, 0.50)
    _box(stage, f"{prim_path}/slab", (0.080, 0.250, 0.018),
         (0.0, 0.0, 0.009), gray, 0.0012, material=body)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/post_{tag}", (POST_T, POST_T, POST_TOP),
             (0.0, sy * (POST_Y0 + POST_T / 2), POST_TOP / 2), gray, 0.0012,
             material=body)
    return root


def _spawn_leaf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bascule leaf: deck slab, side curbs, narrowed far tongue, and the
    walled ballast tray on the tail. Origin = hinge axis; +x = tongue (keep-ward).
    Also authors the trunnion REVOLUTE joint (axis y) to the sibling Pillar body."""
    import omni.usd
    from pxr import Gf, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.05, 0.15, com=COM_LEAF)
    body = _friction_material(stage, f"{prim_path}/body_mat", cfg.mu_body_s, cfg.mu_body_d)
    deck = _friction_material(stage, f"{prim_path}/deck_mat", cfg.mu_deck_s, cfg.mu_deck_d)
    brown = (0.55, 0.38, 0.16)
    steelc = (0.55, 0.57, 0.60)

    _box(stage, f"{prim_path}/slab", (DECK_X1 - DECK_X0, DECK_W, DECK_T),
         ((DECK_X0 + DECK_X1) / 2, 0.0, 0.0), brown, 0.0015, material=deck)
    _box(stage, f"{prim_path}/tongue", (TONGUE_X1 - DECK_X1, TONGUE_W, DECK_T),
         ((DECK_X1 + TONGUE_X1) / 2, 0.0, 0.0), brown, 0.0015, material=deck)
    for tag, sy in (("l", 1.0), ("r", -1.0)):
        _box(stage, f"{prim_path}/curb_{tag}", (DECK_X1 - TRAY_X1, CURB_W, CURB_H),
             ((TRAY_X1 + DECK_X1) / 2, sy * (CHAN_W / 2 + CURB_W / 2),
              DECK_T / 2 + CURB_H / 2), brown, 0.0015, material=body)
        _box(stage, f"{prim_path}/tray_side_{tag}",
             (TRAY_X1 - TRAY_X0, CURB_W, TRAY_WALL_H),
             ((TRAY_X0 + TRAY_X1) / 2, sy * (CHAN_W / 2 + CURB_W / 2),
              DECK_T / 2 + TRAY_WALL_H / 2), steelc, 0.0015, material=body)
    _box(stage, f"{prim_path}/tray_end", (TRAY_WALL_T, CHAN_W, TRAY_WALL_H),
         (TRAY_X0 + TRAY_WALL_T / 2, 0.0, DECK_T / 2 + TRAY_WALL_H / 2), steelc,
         0.0015, material=body)
    _box(stage, f"{prim_path}/tray_gate", (TRAY_WALL_T, CHAN_W, TRAY_WALL_H),
         (TRAY_X1 - TRAY_WALL_T / 2, 0.0, DECK_T / 2 + TRAY_WALL_H / 2), steelc,
         0.0015, material=body)

    # trunnion: revolute joint (axis Y) to the sibling pillar; both anchors on the
    # hinge axis; wide limits — the real stops are CONTACTS on other bodies (heel on
    # the ground when raised, tongue on the keep sill when level)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/trunnion")
    j.CreateBody0Rel().SetTargets([f"{base}/Pillar"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(PIV_Z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-45.0)
    j.CreateUpperLimitAttr(45.0)
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
            mu_floor_s: float = 0.60
            mu_floor_d: float = 0.50

        @configclass
        class PillarSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pillar)
            mu_body_s: float = 0.50
            mu_body_d: float = 0.40

        @configclass
        class LeafSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_leaf)
            mu_body_s: float = 0.45
            mu_body_d: float = 0.40
            mu_deck_s: float = 0.30
            mu_deck_d: float = 0.25

        _SPAWNER_CACHE.update(rig=RigSpawnerCfg, pillar=PillarSpawnerCfg,
                              leaf=LeafSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BasculeKeepSceneCfg(BaseCfg):
    """Config for `BasculeKeepScene`. The interlock is asserted in `__post_init__`:
    the counterweight statics (two blocks hold, ONE block still holds, zero blocks
    fall) and the raised-bridge geometry (no bottle path into the keep until the
    bridge is deployed) are proven from the authored constants."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)
    deploy_max_deg: float = tunable(3.0)  # "bridge deployed": |tilt| at most ~this
    deploy_min_deg: float = tunable(-5.0)  # ... and not fallen below the sill
    raised_min_deg: float = tunable(25.0)  # readback threshold "still raised"
    seat_tol: float = tunable(0.020)  # hinge intact: leaf origin near the trunnion axis
    court_x_pad: float = tunable(0.013)  # in-keep gate inset past the wall inner face
    mus_out_margin: float = tunable(0.010)  # mustard exclusion margin around the keep

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    rig_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the whole machine (m)
    rig_yaw_max: float = tunable(10.0)  # uniform +/- machine yaw (deg)
    slot_x: float = tunable(-0.52)  # bottle spawn slots (rig frame, on the ground)
    slot_y: float = tunable(0.20)
    slot_jitter: float = tunable(0.05)  # uniform +/- xy jitter per bottle
    block_jitter: float = tunable(0.004)  # uniform +/- y jitter per block in the tray

    # --- info: structure -----------------------------------------------------------------------
    bot_r: float = info(BOT_R)
    bot_l: float = info(BOT_L)
    bot_mass: float = info(0.15)
    rig_mass: float = info(40.0)  # heavy dynamic fixture (teleportable, immovable)
    pillar_mass: float = info(PILLAR_M)
    leaf_mass: float = info(LEAF_M)
    block_mass: float = info(BLOCK_M)
    mu_bot_s: float = info(0.35)
    mu_bot_d: float = info(0.30)
    mu_block_s: float = info(0.50)
    mu_block_d: float = info(0.40)
    mu_ground_s: float = info(0.50)
    mu_ground_d: float = info(0.40)

    # Derived (filled in __post_init__).
    theta_up_deg: float = field(default=None, init=False)  # raised rest tilt
    raised_aperture: float = field(default=None, init=False)  # raised tongue -> lintel

    def __post_init__(self) -> None:
        bot_d = 2 * BOT_R
        g = 9.81
        s, c = math.sin(THETA_UP), math.cos(THETA_UP)
        self.theta_up_deg = math.degrees(THETA_UP)
        # -- embodiment: bottle and counterweight blocks fit an ~80 mm parallel jaw --
        assert bot_d < 0.08, "bottle must fit an ~80 mm parallel jaw"
        assert BLOCK[0] < 0.08 and BLOCK[1] < 0.08, "blocks must fit the jaw"
        # -- counterweight statics (angle-independent: cos cancels) --
        # worst-case holding arm: block against the tray's UPHILL (gate) wall
        arm_min = (TRAY_X1 - TRAY_WALL_T) - BLOCK[0] / 2  # ~0.153 magnitude... sign:
        arm_min = abs(arm_min)
        tip_t = LEAF_M * g * COM_LEAF[0]  # leaf's own tip-down torque per unit cos
        assert BLOCK_M * g * arm_min > 1.8 * tip_t, \
            "ONE block must still hold the bridge raised (>=1.8x margin)"
        assert 2 * BLOCK_M * g * arm_min > 3.0 * tip_t, "two blocks: >=3x margin"
        assert COM_LEAF[0] >= 0.03, "unballasted leaf must be decisively tip-heavy"
        # -- raised rest: heel corner on the ground, all other leaf points clear --
        heel = PIV_Z + DECK_X0 * s - (DECK_T / 2) * c
        assert abs(heel) < 0.001, "raised rest: heel bottom corner sits on the ground"
        wall_top_z = PIV_Z + TRAY_X0 * s + (DECK_T / 2 + TRAY_WALL_H) * c
        assert wall_top_z > 0.03, "raised rest: tray wall corner stays clear of ground"
        # -- raised interlock: no path into the keep --
        tip_top = PIV_Z + TONGUE_X1 * s + (DECK_T / 2) * c
        self.raised_aperture = LINTEL_BOT - tip_top
        assert self.raised_aperture < bot_d - 0.010, \
            "raised bridge: tongue-to-lintel aperture must undercut the bottle"
        tip_x_r = TONGUE_X1 * c + (DECK_T / 2) * s
        assert tip_x_r < WALL_X0 - 0.004, "raised tongue stays clear of the keep wall"
        assert 2 * MOUTH_HW < BOT_L - 0.010, \
            "a bottle lying ACROSS the deck can never pass the doorway"
        assert SILL_TOP > bot_d + 0.010, \
            "ground-level bottle cannot climb the pedestal face"
        # -- sweep: the tongue exits the doorway below the lintel while raising --
        phi = math.acos(min(1.0, (WALL_X0 - 0.002) / TONGUE_X1))
        sweep_top = PIV_Z + TONGUE_X1 * math.sin(phi) + (DECK_T / 2) * math.cos(phi)
        assert sweep_top < LINTEL_BOT - 0.004, "tongue sweep must clear the lintel"
        # -- level rest: tongue seats on the sill porch, road continuous --
        tongue_bot = PIV_Z - DECK_T / 2
        assert 0.0 < tongue_bot - SILL_TOP <= 0.003, \
            "level rest: tongue bottom settles onto the sill porch"
        assert TONGUE_X1 > WALL_X0 + 0.020, "tongue must overlap the porch"
        assert 0.0 < FLOOR_X0 - TONGUE_X1 <= 0.012, "tongue-to-floor gap crossable"
        assert DECK_X1 < WALL_X0 - 0.004, "level main slab clears the keep wall"
        assert PIV_Z + DECK_T / 2 > FLOOR_TOP + 0.001, "transit steps DOWN into keep"
        # -- doorway passes the pushed upright bottle over the seated tongue --
        assert 2 * MOUTH_HW > bot_d + 0.020, "doorway width passes the bottle"
        assert LINTEL_BOT - (PIV_Z + DECK_T / 2) > BOT_L + 0.015, \
            "doorway headroom passes the standing bottle over the tongue"
        assert TONGUE_W < 2 * MOUTH_HW - 0.008, "tongue fits through the doorway"
        assert ROOF_BOT - FLOOR_TOP > BOT_L + 0.020, "keep interior fits the bottle"
        assert CHAN_W > bot_d + 0.030, "push channel passes the bottle"
        # -- tray really holds both blocks --
        assert (TRAY_X1 - TRAY_X0 - 2 * TRAY_WALL_T) > BLOCK[0] + 0.010
        assert CHAN_W > 2 * BLOCK[1] + 0.015, "two blocks sit side by side in the tray"
        assert TRAY_WALL_H > BLOCK[2], "tray walls retain the blocks"
        # -- spawn slots: clear of the machine, far outside the keep (null scores 0) --
        assert self.slot_x + self.slot_jitter < DECK_X0 * c - 0.05, \
            "bottle spawn slots stay clear of the raised heel"
        assert self.slot_x + self.slot_jitter < 0.0 < WALL_X0


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bascule_keep")
class BasculeKeepScene(BaseScene):
    cfg: BasculeKeepSceneCfg

    def __init__(self, cfg: BasculeKeepSceneCfg | None = None) -> None:
        super().__init__(cfg or BasculeKeepSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        free_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, linear_damping=0.25, angular_damping=0.15)
        free_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

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
            # NOTE dict order: Pillar must exist before the Leaf spawner authors the
            # trunnion joint targeting the sibling path.
            "pillar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pillar",
                spawn=spawners["pillar"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pillar_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "leaf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Leaf",
                spawn=spawners["leaf"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.leaf_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PIV_Z)),
            ),
        }
        blk_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_block_s, dynamic_friction=c.mu_block_d, restitution=0.0)
        for name, y0 in (("block_a", 0.0305), ("block_b", -0.0305)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=BLOCK,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.36, 0.40)),
                    physics_material=blk_mat, rigid_props=free_rigid,
                    collision_props=free_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.60, y0 * 10, BLOCK[2] / 2 + 0.002)),
            )
        bot_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_bot_s, dynamic_friction=c.mu_bot_d, restitution=0.0)
        for name, color, y0 in (("ketchup", (0.75, 0.10, 0.08), 0.20),
                                ("mustard", (0.85, 0.75, 0.10), -0.20)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CylinderCfg(
                    radius=BOT_R, height=BOT_L, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    physics_material=bot_mat, rigid_props=free_rigid,
                    collision_props=free_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bot_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.52, y0, BOT_L / 2 + 0.002)),
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
        self.pillar: RigidObject = env.iscene["pillar"]
        self.leaf: RigidObject = env.iscene["leaf"]
        self.block_a: RigidObject = env.iscene["block_a"]
        self.block_b: RigidObject = env.iscene["block_b"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.mustard: RigidObject = env.iscene["mustard"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latched-score state (zeroed at reset, kept in get_state/set_state)
        self._b1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._b2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._down = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crossed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + yaw the whole machine coherently (rig, pillar,
        leaf), park the leaf RAISED with its heel a hair off the ground (settles into
        contact), seat both counterweight blocks in the tray, and stand the two
        bottles on SWAPPED ground slots with jitter + free yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- machine: shared xy jitter + yaw ---
        rxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rig_jitter
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_max)
        q_rig = _qz(ryaw)
        zcol = torch.zeros(m, 1, device=dev)
        rig_pos = torch.cat([rxy, zcol], dim=-1)
        write(self.rig, rig_pos, q_rig)
        write(self.pillar, rig_pos, q_rig)

        # --- leaf: raised, heel ~1 mm above ground contact (settles onto it) ---
        piv = rig_pos + _qapply(q_rig, torch.tensor([0.0, 0.0, PIV_Z],
                                                    device=dev).expand(m, 3))
        tilt = torch.full((m,), -(THETA_UP - math.radians(0.4)), device=dev)
        q_leaf = _qmul(q_rig, _qy(tilt))
        write(self.leaf, piv, q_leaf)

        # --- counterweight blocks: seated in the tray (leaf frame), y jitter ---
        for body, sy in ((self.block_a, 1.0), (self.block_b, -1.0)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = (TRAY_X0 + TRAY_X1) / 2
            loc[:, 1] = sy * 0.0305 \
                + (torch.rand(m, device=dev) * 2 - 1) * c.block_jitter
            loc[:, 2] = DECK_T / 2 + BLOCK[2] / 2 + 0.002
            write(body, piv + _qapply(q_leaf, loc), q_leaf)

        # --- bottles: slot-swapped ground spawns (torch.rand — not the degenerate
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

        for buf in (self._b1, self._b2, self._down, self._crossed):
            buf[env_ids] = False

    # ----- state (full, restorable — includes the latched score state) ------------------------
    _BODIES = ("rig", "pillar", "leaf", "block_a", "block_b", "ketchup", "mustard")

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
              for nm in self._BODIES}
        st["_latch"] = torch.stack([
            self._b1[env_ids].float(), self._b2[env_ids].float(),
            self._down[env_ids].float(), self._crossed[env_ids].float()],
            dim=-1).clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in self._BODIES:
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        if "_latch" in state:
            lt = state["_latch"]
            self._b1[env_ids] = lt[:, 0] > 0.5
            self._b2[env_ids] = lt[:, 1] > 0.5
            self._down[env_ids] = lt[:, 2] > 0.5
            self._crossed[env_ids] = lt[:, 3] > 0.5

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised, fully ROOFED blue KEEP stands on a stone pedestal: its floor "
            f"is {FLOOR_TOP * 100:.1f} cm above the ground, it has four walls and a "
            f"roof, and its ONLY opening is a letterbox doorway "
            f"({2 * MOUTH_HW * 100:.0f} cm wide, {(LINTEL_BOT - SILL_TOP) * 100:.0f} "
            f"cm tall) in the front wall — fronted by nothing but open air above the "
            f"moat, so no bottle can be stood, slid, or dropped inside. Facing the "
            f"doorway is a wooden BASCULE BRIDGE: a see-saw leaf on a trunnion "
            f"pillar, with a narrowed tongue toward the keep, side curbs, and a "
            f"walled BALLAST TRAY on its tail. The leaf is tip-heavy and WANTS to "
            f"fall level, but two steel COUNTERWEIGHT blocks "
            f"({BLOCK[0] * 100:.0f} x {BLOCK[1] * 100:.0f} x {BLOCK[2] * 100:.0f} "
            f"cm) sit in the tray and hold it RAISED at about "
            f"{self.cfg.theta_up_deg:.0f} degrees, heel on the ground. Raised, the "
            f"bridge blocks its own road: the tilted tongue ends short of the "
            f"doorway with only a {self.cfg.raised_aperture * 100:.1f} cm gap to the "
            f"lintel — smaller than the {2 * BOT_R * 100:.1f} cm bottles. EITHER "
            f"single block alone still holds the bridge up: BOTH must be lifted out "
            f"of the tray (each is an easy jaw-sized pick) and set on the ground. "
            f"With the tray empty, gravity swings the bridge down on its own until "
            f"the tongue rests on the doorway sill, completing a level road into the "
            f"keep. On the ground behind the bridge stand two visually identical "
            f"bottles ({2 * BOT_R * 100:.1f} cm across, {BOT_L * 100:.0f} cm tall): "
            f"a RED ketchup and a YELLOW mustard, on randomly swapped spots — tell "
            f"them apart by color. Machine pose and bottle spots change every "
            f"episode.\nGoal: remove BOTH counterweight blocks from the tray so the "
            f"bridge lowers itself, then place the RED ketchup onto the deck and "
            f"push it along the curb channel, over the tongue, through the doorway, "
            f"onto the keep floor. The bridge must END deployed (leaf settled within "
            f"{c.deploy_max_deg:.0f} degrees of level on its sill, hinge intact, "
            f"tray empty) with the ketchup resting inside the keep and the YELLOW "
            f"mustard outside. Success is judged on the settled end state."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lift both steel counterweight blocks out of the bridge's tail tray and "
            "set them on the ground so the bascule bridge lowers itself level. Then "
            "put the red ketchup bottle on the deck and push it across the bridge "
            "through the keep's doorway so it rests inside. Keep the yellow mustard "
            "bottle out of the keep."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> the rig's frame."""
        return _qapply(_qinv(self.rig.data.root_quat_w),
                       pos_w - self.rig.data.root_pos_w)

    def _leaf_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return _qapply(_qinv(self.leaf.data.root_quat_w),
                       pos_w - self.leaf.data.root_pos_w)

    def leaf_tilt_deg(self) -> torch.Tensor:
        """(N,) leaf tilt in deg, + = tongue end UP (raised ~ +34.9, level ~ 0)."""
        ex = _qapply(self.leaf.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0],
                                  device=self.env.device).expand(self.env.num_envs, 3))
        return torch.rad2deg(torch.asin(ex[:, 2].clamp(-1.0, 1.0)))

    def hinge_intact(self) -> torch.Tensor:
        """(N,) bool: leaf origin still on the trunnion axis (machine intact)."""
        loc = self._rig_local(self.leaf.data.root_pos_w)
        loc = loc - torch.tensor([0.0, 0.0, PIV_Z], device=loc.device)
        return loc.norm(dim=-1) <= self.cfg.seat_tol

    def bridge_down(self) -> torch.Tensor:
        """(N,) bool: leaf within the deployed band (tongue seated on the sill)."""
        t = self.leaf_tilt_deg()
        return (t <= self.cfg.deploy_max_deg) & (t >= self.cfg.deploy_min_deg) \
            & self.hinge_intact()

    def bridge_raised(self) -> torch.Tensor:
        return self.leaf_tilt_deg() >= self.cfg.raised_min_deg

    def in_tray(self, body) -> torch.Tensor:
        """(N,) bool: block center inside the ballast tray (leaf frame)."""
        loc = self._leaf_local(body.data.root_pos_w)
        return (loc[:, 0] >= TRAY_X0 - 0.005) & (loc[:, 0] <= TRAY_X1 + 0.005) \
            & loc[:, 1].abs().le(CHAN_W / 2 + 0.005) \
            & (loc[:, 2] >= 0.0) & (loc[:, 2] <= 0.090)

    def tray_empty(self) -> torch.Tensor:
        return ~self.in_tray(self.block_a) & ~self.in_tray(self.block_b)

    def in_keep(self, body) -> torch.Tensor:
        """(N,) bool: bottle center inside the keep interior (rig frame), past the
        doorway inner face."""
        c = self.cfg
        loc = self._rig_local(body.data.root_pos_w)
        return (loc[:, 0] >= WALL_X1 + c.court_x_pad) & (loc[:, 0] <= FLOOR_X1 - 0.010) \
            & loc[:, 1].abs().le(COURT_Y - 0.010) \
            & (loc[:, 2] >= FLOOR_TOP - 0.002) & (loc[:, 2] <= ROOF_BOT - 0.015)

    def past_doorway(self, body) -> torch.Tensor:
        """(N,) bool: bottle center past the doorway OUTER plane at road height (the
        crossing latch zone: doorway corridor + keep interior)."""
        loc = self._rig_local(body.data.root_pos_w)
        return (loc[:, 0] >= WALL_X0 + 0.010) & (loc[:, 0] <= FLOOR_X1) \
            & loc[:, 1].abs().le(COURT_Y) \
            & (loc[:, 2] >= SILL_TOP) & (loc[:, 2] <= ROOF_BOT - 0.010)

    def mustard_out(self) -> torch.Tensor:
        """(N,) bool: mustard center OUTSIDE the (margin-grown) keep box."""
        c = self.cfg
        loc = self._rig_local(self.mustard.data.root_pos_w)
        inside = (loc[:, 0] >= WALL_X0 - c.mus_out_margin) \
            & (loc[:, 0] <= KEEP_X1 + c.mus_out_margin) \
            & loc[:, 1].abs().le(KEEP_HY + c.mus_out_margin) \
            & (loc[:, 2] >= 0.0) & (loc[:, 2] <= ROOF_BOT + 0.020)
        return ~inside

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for nm in self._BODIES:
            ok &= getattr(self, nm).data.root_state_w.isfinite().all(dim=-1)
        return ok

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: RED ketchup resting inside the keep, settled; bridge DEPLOYED
        (leaf level on its sill, settled, hinge intact) with the ballast tray EMPTY
        (kills kinematic-hold fakes; physically forced anyway — asserted margins);
        YELLOW mustard outside; all states finite."""
        return self.in_keep(self.ketchup) & self.settled(self.ketchup) \
            & self.bridge_down() & self.settled(self.leaf) & self.tray_empty() \
            & self.mustard_out() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched and non-decreasing along the demonstrated
        solution: 0.10 first counterweight out of the tray; +0.10 tray empty; +0.25
        bridge deployed (gated on tray empty — pressing it down with ballast aboard
        earns nothing); +0.25 ketchup past the doorway plane (gated on deployed);
        1.0 iff success(). The null policy holds ~0 (blocks spawn in the tray)."""
        one_out = ~self.in_tray(self.block_a) | ~self.in_tray(self.block_b)
        self._b1 |= one_out
        self._b2 |= self.tray_empty()
        self._down |= self._b2 & self.bridge_down()
        self._crossed |= self._down & self.past_doorway(self.ketchup)
        base = 0.10 * self._b1.float() + 0.10 * self._b2.float() \
            + 0.25 * self._down.float() + 0.25 * self._crossed.float()
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="bascule_keep", robot="null", env_spacing=3.0))
