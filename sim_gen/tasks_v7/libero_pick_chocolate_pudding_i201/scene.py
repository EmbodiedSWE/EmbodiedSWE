"""ShuttleVaultScene — deposit the chocolate pudding into a SEALED vault chamber by
operating an internal shuttle car through a through-wall handle.

Derived from libero/libero_pick_chocolate_pudding ("pick up the chocolate pudding and
put it in the basket": one target among tabletop distractors, one OPEN goal container,
one grasp-carry-lower, one bbox check). Kept from the seed: a brown "chocolate pudding"
target that must end up inside a goal vessel, among loose objects that are not the
target. Strategically inverted: the goal vessel here is the lower CHAMBER of a sealed
VAULT that no hand, jaw or dropped object can reach directly —

  - the chamber is roofed by a full-width GALLERY FLOOR whose only opening is a square
    DROP HOLE at station B (housing-local +x end);
  - the gallery above it is roofed by the vault ROOF whose only opening is a square
    LOADING WINDOW at station A (-x end) — HORIZONTALLY OFFSET from the drop hole, with
    solid roof/floor overlap between them, so nothing can fall from outside into the
    chamber on any straight (or bounced: restitution 0) path;
  - riding a prismatic RAIL inside the gallery is a SHUTTLE CAR: an open-top,
    open-BOTTOM pocket collar that skims the gallery floor. Its only external interface
    is a thin HANDLE BAR that exits through a small slot in the -x end wall to a T-KNOB
    outside. The slot passes the 12 mm bar but nothing as large as any scene object.

The only way in: park the shuttle at A (under the window), drop the pudding through the
window into the pocket, push the knob so the pocket CONVEYS the pudding — sliding it
across the gallery floor — to B, where the floor disappears (the pocket interior is
fully over the drop hole) and the pudding falls into the chamber by gravity. Then pull
the shuttle back HOME to A (the declared end state: loader returned). The two
distractors (a ketchup box, a milk cube) must stay out of the vault.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * loaded    — pudding ever inside the shuttle pocket in the gallery   (latched)
  0.15 * conveyed  — ever loaded AND carried past the vault midline (+x)     (latched)
  0.30 * deposited — pudding ever in the chamber                             (latched)
  0.10 * homed     — shuttle ever back home AFTER the deposit                (latched)
  1.0 iff success() — pudding settled in the chamber, both distractors outside the
                     vault, shuttle home, everything settled and finite.
Non-success is capped at 0.75. All success clauses are live physical outcomes.

Honesty geometry (asserted in `__post_init__`):
  - window (72 mm) < pocket interior (78 mm) and window sits inside the pocket
    footprint at home: any clean window pass lands INSIDE the pocket;
  - drop hole (80 mm) >= pocket interior: at B the whole pocket interior is over the
    hole, so a conveyed pudding MUST fall (worst-case tumbling width 68 mm < 80 mm);
  - window and hole projections are separated by solid overlap: no straight drop path;
  - the wall slot (24 mm) passes only the handle bar; every object out-spans it;
  - the roof-to-pocket-wall gap (8 mm) and floor skim gap (2 mm) out-span nothing:
    contents cannot escape the pocket except through the drop hole.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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
    """One collidable box child: translate + scale, displayColor, collider."""
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


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC vault housing. Local frame: origin at the footprint centre
    on the ground, +x toward station B (drop hole), -x toward station A (window +
    handle slot). Children of one body never self-collide; the vault never moves."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c, co = cfg, cfg.contact_offset
    body, deck, roofc = c.body_color, c.deck_color, c.roof_color
    hx, hy, hz, wt = c.out_hx, c.out_hy, c.out_h, c.wall_t          # 0.113/0.062/0.215/0.010
    ix, iy = hx - wt, hy - wt                                       # interior half spans
    xa, xb = c.x_a, c.x_b                                           # station A / B centres
    win, hole = c.window, c.hole
    gz0, gz1 = c.gal_z0, c.gal_z1                                   # gallery floor slab z-band
    rz0, rz1 = c.roof_z0, c.roof_z1                                 # roof slab z-band
    sz, sy = c.slot, c.slot                                         # handle slot (square)
    scz = c.rail_z                                                  # slot centred at rail height
    kids = [
        # base floor (chamber floor), z 0..wt
        _box(stage, f"{prim_path}/base", center=(0.0, 0.0, wt / 2),
             size=(2 * ix, 2 * iy, wt), color=deck, contact_offset=co),
        # +/-y walls, full outer footprint, full height
        _box(stage, f"{prim_path}/wall_yp", center=(0.0, hy - wt / 2, hz / 2),
             size=(2 * hx, wt, hz), color=body, contact_offset=co),
        _box(stage, f"{prim_path}/wall_yn", center=(0.0, -(hy - wt / 2), hz / 2),
             size=(2 * hx, wt, hz), color=body, contact_offset=co),
        # +x end wall, full height
        _box(stage, f"{prim_path}/wall_xp", center=(hx - wt / 2, 0.0, hz / 2),
             size=(wt, 2 * iy, hz), color=body, contact_offset=co),
        # -x end wall with the handle slot (square, centred at y=0, z=rail_z)
        _box(stage, f"{prim_path}/wall_xn_lo", center=(-(hx - wt / 2), 0.0, (scz - sz / 2) / 2),
             size=(wt, 2 * iy, scz - sz / 2), color=body, contact_offset=co),
        _box(stage, f"{prim_path}/wall_xn_hi",
             center=(-(hx - wt / 2), 0.0, (scz + sz / 2 + hz) / 2),
             size=(wt, 2 * iy, hz - (scz + sz / 2)), color=body, contact_offset=co),
        _box(stage, f"{prim_path}/wall_xn_sp",
             center=(-(hx - wt / 2), (sy / 2 + iy) / 2, scz),
             size=(wt, iy - sy / 2, sz), color=body, contact_offset=co),
        _box(stage, f"{prim_path}/wall_xn_sn",
             center=(-(hx - wt / 2), -(sy / 2 + iy) / 2, scz),
             size=(wt, iy - sy / 2, sz), color=body, contact_offset=co),
        # gallery floor (z gz0..gz1) with the DROP HOLE at station B
        _box(stage, f"{prim_path}/gal_a",
             center=((-ix + (xb - hole / 2)) / 2, 0.0, (gz0 + gz1) / 2),
             size=((xb - hole / 2) + ix, 2 * iy, gz1 - gz0), color=deck, contact_offset=co),
        _box(stage, f"{prim_path}/gal_b",
             center=(((xb + hole / 2) + ix) / 2, 0.0, (gz0 + gz1) / 2),
             size=(ix - (xb + hole / 2), 2 * iy, gz1 - gz0), color=deck, contact_offset=co),
        _box(stage, f"{prim_path}/gal_sp",
             center=(xb, (hole / 2 + iy) / 2, (gz0 + gz1) / 2),
             size=(hole, iy - hole / 2, gz1 - gz0), color=deck, contact_offset=co),
        _box(stage, f"{prim_path}/gal_sn",
             center=(xb, -(hole / 2 + iy) / 2, (gz0 + gz1) / 2),
             size=(hole, iy - hole / 2, gz1 - gz0), color=deck, contact_offset=co),
        # roof (z rz0..rz1) with the LOADING WINDOW at station A
        _box(stage, f"{prim_path}/roof_a",
             center=((-hx + (xa - win / 2)) / 2, 0.0, (rz0 + rz1) / 2),
             size=((xa - win / 2) + hx, 2 * hy, rz1 - rz0), color=roofc, contact_offset=co),
        _box(stage, f"{prim_path}/roof_b",
             center=(((xa + win / 2) + hx) / 2, 0.0, (rz0 + rz1) / 2),
             size=(hx - (xa + win / 2), 2 * hy, rz1 - rz0), color=roofc, contact_offset=co),
        _box(stage, f"{prim_path}/roof_sp",
             center=(xa, (win / 2 + hy) / 2, (rz0 + rz1) / 2),
             size=(win, hy - win / 2, rz1 - rz0), color=roofc, contact_offset=co),
        _box(stage, f"{prim_path}/roof_sn",
             center=(xa, -(win / 2 + hy) / 2, (rz0 + rz1) / 2),
             size=(win, hy - win / 2, rz1 - rz0), color=roofc, contact_offset=co),
    ]
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC shuttle car + its PRISMATIC rail joint to the sibling
    housing (joints must be authored at spawn — post-play joints are dead; the
    joint pair is left collision-FILTERED, the USD default, so the car rides the
    rail without rubbing the vault and the handle bar passes its slot freely —
    travel is bounded by the authored joint limits).

    Local frame: origin at the pocket centre, z=0 at rail height. Children:
      pocket — open-top, open-BOTTOM square collar (interior `pock_in`, walls
               `pock_wt` thick x `pock_wh` tall);
      bar    — 12 mm square handle bar along -x, through the wall slot;
      knob   — T-cap plate at the outer end (the graspable handle)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c, co = cfg, cfg.contact_offset
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(c.lin_damping))
    px.CreateAngularDampingAttr(2.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    hi = c.pock_in / 2                      # pocket interior half span
    ho = hi + c.pock_wt                     # pocket exterior half span
    wh = c.pock_wh
    kids = []
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/wall_x{s}",
                         center=(sgn * (hi + c.pock_wt / 2), 0.0, 0.0),
                         size=(c.pock_wt, 2 * ho, wh), color=c.color, contact_offset=co))
        kids.append(_box(stage, f"{prim_path}/wall_y{s}",
                         center=(0.0, sgn * (hi + c.pock_wt / 2), 0.0),
                         size=(2 * hi, c.pock_wt, wh), color=c.color, contact_offset=co))
    bar_x0, bar_x1 = -c.bar_len - ho, -ho   # bar spans pocket wall -> outside
    kids.append(_box(stage, f"{prim_path}/bar",
                     center=((bar_x0 + bar_x1) / 2, 0.0, 0.0),
                     size=(c.bar_len, c.bar_t, c.bar_t), color=c.bar_color,
                     contact_offset=co))
    kids.append(_box(stage, f"{prim_path}/knob",
                     center=(bar_x0 - c.knob_t / 2, 0.0, 0.0),
                     size=(c.knob_t, c.knob_w, c.knob_w), color=c.knob_color,
                     contact_offset=co))
    for k in kids:
        _bind_material(k, mat)

    # prismatic rail to the sibling housing, axis = housing local X
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/rail")
    j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    # joint origin: station A at rail height in the housing frame -> q=0 is HOME
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.x_a), 0.0, float(c.rail_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-0.001)
    j.CreateUpperLimitAttr(float(c.travel) + 0.001)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            out_hx: float = 0.113
            out_hy: float = 0.062
            out_h: float = 0.215
            wall_t: float = 0.010
            x_a: float = -0.052
            x_b: float = 0.052
            window: float = 0.072
            hole: float = 0.080
            gal_z0: float = 0.115
            gal_z1: float = 0.125
            roof_z0: float = 0.205
            roof_z1: float = 0.215
            slot: float = 0.024
            rail_z: float = 0.162
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            body_color: tuple = (0.34, 0.38, 0.46)
            deck_color: tuple = (0.45, 0.49, 0.56)
            roof_color: tuple = (0.28, 0.31, 0.38)
            contact_offset: float = 0.002

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            pock_in: float = 0.078
            pock_wt: float = 0.008
            pock_wh: float = 0.070
            bar_len: float = 0.138
            bar_t: float = 0.012
            knob_t: float = 0.010
            knob_w: float = 0.032
            x_a: float = -0.052
            rail_z: float = 0.162
            travel: float = 0.104
            mass: float = 0.30
            lin_damping: float = 8.0
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            color: tuple = (0.88, 0.52, 0.12)
            bar_color: tuple = (0.75, 0.75, 0.78)
            knob_color: tuple = (0.95, 0.83, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg
        _SPAWNER_CACHE["shuttle"] = ShuttleSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ShuttleVaultSceneCfg(BaseCfg):
    """Config for `ShuttleVaultScene`. The chamber is honest by construction: its only
    opening (the drop hole) is roofed by the vault roof whose only opening (the
    window) is horizontally offset with solid overlap between them — the sole path
    from outside to the chamber runs through the shuttle pocket."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    home_tol: float = tunable(0.015)        # shuttle joint coordinate q at/below this = HOME
    conv_x_min: float = tunable(0.005)      # housing-local pudding x beyond this = conveyed
    pocket_xy_tol: float = tunable(0.030)   # |pudding centre| per axis in the shuttle frame
    cham_x_tol: float = tunable(0.098)      # chamber window, housing frame (per axis)
    cham_y_tol: float = tunable(0.047)
    cham_z_lo: float = tunable(0.012)       # pudding centre height band inside the chamber
    cham_z_hi: float = tunable(0.080)       # (well BELOW the gallery-floor sill at 0.115)
    gal_z_lo: float = tunable(0.130)        # pudding centre height band inside the gallery
    gal_z_hi: float = tunable(0.200)
    settle_speed: float = tunable(0.05)     # max |lin vel| of every judged body (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    vault_jitter: float = tunable(0.030)    # BUILD-time vault xy jitter (+/- m)
    vault_yaw_deg: float = tunable(15.0)    # BUILD-time vault yaw (+/- deg)
    item_jitter: float = tunable(0.020)     # per-item xy jitter (+/- m), each reset
    item_yaw_deg: float = tunable(180.0)    # per-item free yaw (+/- deg), each reset
    q0_lo: float = tunable(0.005)           # reset shuttle coordinate range (anywhere on
    q0_hi: float = tunable(0.100)           # the rail, home NOT guaranteed)

    # --- info: layout (world nominal, ground z = 0) ----------------------------------------------
    vault_pos: tuple = info((0.45, 0.0))    # vault footprint centre (knob side faces -x)
    pudding_slot: tuple = info((0.28, 0.17))
    ketchup_slot: tuple = info((0.28, -0.17))
    milk_slot: tuple = info((0.44, 0.25))
    # --- info: vault structure (housing local frame, origin at footprint centre, ground) ---------
    out_hx: float = info(0.113)             # outer half-extents / height
    out_hy: float = info(0.062)
    out_h: float = info(0.215)
    wall_t: float = info(0.010)
    x_a: float = info(-0.052)               # station A: loading window + shuttle HOME
    x_b: float = info(0.052)                # station B: drop hole
    window: float = info(0.072)             # roof loading window, square
    hole: float = info(0.080)               # gallery-floor drop hole, square
    gal_z0: float = info(0.115)             # gallery floor slab z-band
    gal_z1: float = info(0.125)
    roof_z0: float = info(0.205)            # roof slab z-band
    roof_z1: float = info(0.215)
    slot: float = info(0.024)               # handle slot in the -x wall, square
    rail_z: float = info(0.162)             # rail (and slot centre) height
    # --- info: shuttle ---------------------------------------------------------------------------
    pock_in: float = info(0.078)            # pocket interior span
    pock_wt: float = info(0.008)            # pocket wall thickness
    pock_wh: float = info(0.070)            # pocket wall height (z 0.127..0.197)
    travel: float = info(0.104)             # rail travel A -> B (= x_b - x_a)
    bar_len: float = info(0.138)
    bar_t: float = info(0.012)
    knob_t: float = info(0.010)
    knob_w: float = info(0.032)
    shuttle_mass: float = info(0.30)
    shuttle_damping: float = info(8.0)      # rail is frictionless (joint pair filtered;
    #                                         jointFriction is inert on non-articulation
    #                                         joints) — body damping is the brake
    # --- info: items -----------------------------------------------------------------------------
    pudding_size: tuple = info((0.045, 0.045, 0.050))
    pudding_mass: float = info(0.12)
    pudding_color: tuple = info((0.45, 0.26, 0.14))
    ketchup_size: tuple = info((0.045, 0.030, 0.060))
    ketchup_mass: float = info(0.10)
    ketchup_color: tuple = info((0.75, 0.10, 0.08))
    milk_size: tuple = info((0.050, 0.050, 0.050))
    milk_mass: float = info(0.10)
    milk_color: tuple = info((0.92, 0.92, 0.90))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.40)
    # rubric weights (0.20 + 0.15 + 0.30 + 0.10 = 0.75 = the non-success cap)
    w_load: float = info(0.20)
    w_conv: float = info(0.15)
    w_dep: float = info(0.30)
    w_home: float = info(0.10)

    def __post_init__(self) -> None:
        ix, iy = self.out_hx - self.wall_t, self.out_hy - self.wall_t
        pud = self.pudding_size
        pud_min, pud_h = min(pud[0], pud[1]), pud[2]
        # window inside the pocket footprint at home: a clean window pass lands INSIDE
        assert self.window < self.pock_in, "window must under-span the pocket interior"
        assert self.window / 2 <= self.pock_in / 2, "window inside the pocket at home"
        # pudding fits the window with clearance, and cannot jam in the hole
        assert pud_min <= self.window - 0.020, "pudding must pass the window cleanly"
        assert math.hypot(pud_min, pud_h) < self.hole - 0.010, \
            "pudding must pass the hole even tumbling"
        # at station B the whole pocket interior is over the hole: deposit guaranteed
        assert self.hole >= self.pock_in, "hole must cover the pocket interior at B"
        # NO straight drop path outside->chamber: window and hole projections separated
        assert (self.x_a + self.window / 2) + 0.015 <= (self.x_b - self.hole / 2), \
            "roof window and floor hole must not overlap (sealing)"
        # the wall slot passes the bar, and nothing else
        assert self.bar_t <= self.slot - 0.010, "bar must ride the slot freely"
        for s in (pud, self.ketchup_size, self.milk_size):
            assert min(s) > self.slot + 0.005, "every object must out-span the slot"
        # pocket sealing: roof gap and floor skim gap out-span nothing
        wall_top = self.rail_z + self.pock_wh / 2
        assert 0.004 <= self.roof_z0 - wall_top <= 0.012, "roof-to-pocket-wall gap"
        skim = (self.rail_z - self.pock_wh / 2) - self.gal_z1
        assert 0.0005 <= skim <= 0.006, "pocket must skim the gallery floor"
        # headroom: pudding stands in the gallery pocket and in the chamber
        assert self.roof_z0 - self.gal_z1 > pud_h + 0.020, "gallery headroom"
        assert self.gal_z0 - self.wall_t > pud_h + 0.030, "chamber headroom"
        # chamber band sits far below the sill (mid-transit cannot fake a deposit rest)
        assert self.cham_z_hi < self.gal_z0 - 0.030, "chamber z band below the sill"
        # shuttle rides the interior with clearance at both ends and both sides
        po = self.pock_in / 2 + self.pock_wt
        assert iy - po >= 0.004, "pocket-to-side-wall clearance"
        assert (self.x_a - po) - (-ix) >= 0.003, "pocket-to-A-wall clearance"
        assert ix - (self.x_b + po) >= 0.003, "pocket-to-B-wall clearance"
        assert abs((self.x_b - self.x_a) - self.travel) < 1e-6, "travel = A->B span"
        # the knob stays graspable OUTSIDE the vault at every q
        cap_in = -(self.bar_len + po)                   # knob inner face, shuttle frame
        worst = self.x_a + self.travel + cap_in         # housing-local at q = travel
        assert worst <= -self.out_hx - 0.015, "knob must protrude >= 15 mm at all q"
        # scatter slots clear the vault (bounding circle + build jitter) and each other
        bound = math.hypot(self.out_hx, self.out_hy) + self.vault_jitter
        knob_reach = self.out_hx + self.bar_len + self.knob_t + 0.02 + self.vault_jitter
        for slot, size in ((self.pudding_slot, pud), (self.ketchup_slot, self.ketchup_size),
                           (self.milk_slot, self.milk_size)):
            half_diag = math.hypot(size[0], size[1]) / 2
            d = math.hypot(slot[0] - self.vault_pos[0], slot[1] - self.vault_pos[1])
            assert d - self.item_jitter > bound + half_diag + 0.01, "slot clears the vault"
            # clear of the knob sweep: outside its reach OR far off the -x axis strip
            in_strip = abs(slot[1]) - self.item_jitter - half_diag < 0.115
            ahead = slot[0] + self.item_jitter + half_diag > self.vault_pos[0] - knob_reach
            assert not (in_strip and ahead and slot[0] < self.vault_pos[0]) or d > knob_reach, \
                "slot clears the knob sweep"
        for a, b in ((self.pudding_slot, self.ketchup_slot),
                     (self.pudding_slot, self.milk_slot),
                     (self.ketchup_slot, self.milk_slot)):
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            assert d - 2 * self.item_jitter > 0.09, "item slots clear each other"
        assert abs((self.w_load + self.w_conv + self.w_dep + self.w_home) - 0.75) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shuttle_vault")
class ShuttleVaultScene(BaseScene):
    cfg: ShuttleVaultSceneCfg

    def __init__(self, cfg: ShuttleVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or ShuttleVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """BUILD-time vault pose randomization: the housing is body0 of a spawn-authored
        joint, so it must NEVER be teleported (kinematic joint anchors stay world-fixed).
        Its pose is drawn HERE — `BaseEnv` seeds the RNGs before calling assets(), so
        `build(seed=...)` makes the draw reproducible — and the shuttle spawns at HOME
        (q=0) in that frame. Per-episode variation of the free items and of the shuttle
        coordinate happens in reset()."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        r = torch.rand(3)
        vx = c.vault_pos[0] + float(r[0] * 2 - 1) * c.vault_jitter
        vy = c.vault_pos[1] + float(r[1] * 2 - 1) * c.vault_jitter
        vyaw = math.radians(float(r[2] * 2 - 1) * c.vault_yaw_deg)
        self._vault_build = (vx, vy, vyaw)
        cosw, sinw = math.cos(vyaw), math.sin(vyaw)
        quat = (math.cos(vyaw / 2), 0.0, 0.0, math.sin(vyaw / 2))
        # shuttle spawn at HOME: housing frame (x_a, 0, rail_z) -> env frame
        sx = vx + cosw * c.x_a
        sy = vy + sinw * c.x_a

        cls = _spawner_classes()
        housing_spawn = cls["housing"](
            out_hx=c.out_hx, out_hy=c.out_hy, out_h=c.out_h, wall_t=c.wall_t,
            x_a=c.x_a, x_b=c.x_b, window=c.window, hole=c.hole,
            gal_z0=c.gal_z0, gal_z1=c.gal_z1, roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            slot=c.slot, rail_z=c.rail_z, contact_offset=c.contact_offset)
        shuttle_spawn = cls["shuttle"](
            pock_in=c.pock_in, pock_wt=c.pock_wt, pock_wh=c.pock_wh,
            bar_len=c.bar_len, bar_t=c.bar_t, knob_t=c.knob_t, knob_w=c.knob_w,
            x_a=c.x_a, rail_z=c.rail_z, travel=c.travel, mass=c.shuttle_mass,
            lin_damping=c.shuttle_damping, contact_offset=c.contact_offset)

        item_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.45, dynamic_friction=0.40, restitution=0.0),
        )

        out: dict[str, Any] = {
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
            # ORDER MATTERS: the housing must exist when the shuttle's rail joint is
            # authored (Body0Rel targets the sibling {ENV_REGEX_NS}/Housing).
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=housing_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(vx, vy, 0.0), rot=quat),
            ),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle",
                spawn=shuttle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.rail_z), rot=quat),
            ),
        }
        for name, size, mass, color, slot in (
                ("pudding", c.pudding_size, c.pudding_mass, c.pudding_color, c.pudding_slot),
                ("ketchup", c.ketchup_size, c.ketchup_mass, c.ketchup_color, c.ketchup_slot),
                ("milk", c.milk_size, c.milk_mass, c.milk_color, c.milk_slot)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    **item_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(slot[0], slot[1], size[2] / 2 + 0.003)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.housing: RigidObject = env.iscene["housing"]
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.pudding: RigidObject = env.iscene["pudding"]
        self.ketchup: RigidObject = env.iscene["ketchup"]
        self.milk: RigidObject = env.iscene["milk"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._conveyed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._deposited = torch.zeros(n, dtype=torch.bool, device=dev)
        self._homed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the three free items scattered on their slots (xy jitter +
        free yaw), the shuttle teleported ALONG ITS RAIL to a random coordinate
        q0 ~ U[q0_lo, q0_hi] (home not guaranteed), latches cleared. The vault itself
        is kinematic body0 of the rail joint and is never moved after build."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.item_yaw_deg)

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, x, y, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for body, slot, size in ((self.pudding, c.pudding_slot, c.pudding_size),
                                 (self.ketchup, c.ketchup_slot, c.ketchup_size),
                                 (self.milk, c.milk_slot, c.milk_size)):
            write(body, slot[0] + rnd(c.item_jitter), slot[1] + rnd(c.item_jitter),
                  torch.full((m,), size[2] / 2 + 0.003, device=dev), _qz(rnd(yaw_amp)))

        # shuttle: random coordinate along the rail, in the (fixed) housing frame
        q0 = torch.rand(m, device=dev) * (c.q0_hi - c.q0_lo) + c.q0_lo
        hp = self.housing.data.root_pos_w[env_ids]
        hq = self.housing.data.root_quat_w[env_ids]
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.x_a + q0
        loc[:, 2] = c.rail_z
        pos = hp + _qapply(hq, loc)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pos
        st[:, 3:7] = hq
        self.shuttle.write_root_state_to_sim(st, env_ids)

        self._loaded[env_ids] = False
        self._conveyed[env_ids] = False
        self._deposited[env_ids] = False
        self._homed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "pudding": self.pudding.data.root_state_w[env_ids].clone(),
            "ketchup": self.ketchup.data.root_state_w[env_ids].clone(),
            "milk": self.milk.data.root_state_w[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "conveyed": self._conveyed[env_ids].clone(),
            "deposited": self._deposited[env_ids].clone(),
            "homed": self._homed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        self.pudding.write_root_state_to_sim(state["pudding"], env_ids)
        self.ketchup.write_root_state_to_sim(state["ketchup"], env_ids)
        self.milk.write_root_state_to_sim(state["milk"], env_ids)
        self._loaded[env_ids] = state["loaded"]
        self._conveyed[env_ids] = state["conveyed"]
        self._deposited[env_ids] = state["deposited"]
        self._homed[env_ids] = state["homed"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel-blue VAULT ({2 * c.out_hx * 1000:.0f} x {2 * c.out_hy * 1000:.0f} mm, "
            f"{c.out_h * 1000:.0f} mm tall) is fixed to the floor; its position and heading "
            f"vary per build. It is sealed except for three openings. (1) A square LOADING "
            f"WINDOW ({c.window * 1000:.0f} mm) in the roof at the end NEAREST the yellow "
            f"knob. (2) Hidden inside, a square DROP HOLE ({c.hole * 1000:.0f} mm) in the "
            f"mid-level gallery floor at the FAR end — horizontally offset from the window, "
            f"with solid roof between, so nothing dropped from outside can reach the lower "
            f"chamber directly. (3) A small slot in the near end wall, passing only the "
            f"{c.bar_t * 1000:.0f} mm handle bar of an internal SHUTTLE CAR to a yellow "
            f"T-KNOB outside. The shuttle is an open-top, open-bottom pocket collar riding "
            f"a straight rail in the gallery: push the knob IN and the pocket travels to "
            f"the far end (over the drop hole); pull it OUT and the pocket returns HOME "
            f"under the window. Its start position on the rail varies per episode.\n"
            f"On the floor nearby lie a brown CHOCOLATE PUDDING block "
            f"({c.pudding_size[0] * 1000:.0f} mm square, {c.pudding_size[2] * 1000:.0f} mm "
            f"tall), a red ketchup box and a white milk cube; positions and headings vary.\n"
            f"Goal: get the chocolate pudding INTO the vault's lower chamber and leave the "
            f"loader ready: pudding resting inside the chamber (it can only get there by "
            f"falling through the drop hole), the shuttle returned HOME (knob pulled fully "
            f"out, pocket under the window, within {c.home_tol * 1000:.0f} mm), the ketchup "
            f"and milk left OUTSIDE the vault, and everything at rest. A pudding parked on "
            f"the roof, inside the pocket, on the gallery floor, or anywhere outside the "
            f"chamber does not count; neither does a shuttle left pushed in."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Park the shuttle under the roof window with the yellow knob, drop the "
            "chocolate pudding through the window into the shuttle pocket, push the knob "
            "to carry it over the internal drop hole so it falls into the vault's lower "
            "chamber, then pull the shuttle back home. Leave the ketchup and milk outside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _housing_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the housing (vault) body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.housing.data.root_quat_w),
                       pos_w - self.housing.data.root_pos_w)

    def _shuttle_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the shuttle body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.shuttle.data.root_quat_w),
                       pos_w - self.shuttle.data.root_pos_w)

    def shuttle_q(self) -> torch.Tensor:
        """(N,) rail coordinate: 0 = HOME (station A, under the window),
        `travel` = station B (over the drop hole)."""
        loc = self._housing_local(self.shuttle.data.root_pos_w)
        return loc[:, 0] - self.cfg.x_a

    def home(self) -> torch.Tensor:
        """(N,) bool: shuttle at the loading station (q <= home_tol)."""
        return self.shuttle_q() <= self.cfg.home_tol

    def in_gallery(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the gallery airspace (housing frame)."""
        c = self.cfg
        loc = self._housing_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < c.out_hx - c.wall_t) \
            & (loc[:, 1].abs() < c.out_hy - c.wall_t) \
            & (loc[:, 2] > c.gal_z_lo) & (loc[:, 2] < c.gal_z_hi)

    def in_pocket(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the shuttle pocket footprint (shuttle
        frame) — combined with `in_gallery` this is a LOADED pudding."""
        loc = self._shuttle_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < self.cfg.pocket_xy_tol) \
            & (loc[:, 1].abs() < self.cfg.pocket_xy_tol)

    def in_chamber(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the lower chamber (housing frame, height
        band far BELOW the gallery-floor sill: only a body that fell through the
        drop hole and rests on the chamber floor satisfies it)."""
        c = self.cfg
        loc = self._housing_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < c.cham_x_tol) & (loc[:, 1].abs() < c.cham_y_tol) \
            & (loc[:, 2] > c.cham_z_lo) & (loc[:, 2] < c.cham_z_hi)

    def in_vault(self, body) -> torch.Tensor:
        """(N,) bool: body centre anywhere inside the vault envelope (distractor
        exclusion zone: chamber AND gallery AND both apertures)."""
        c = self.cfg
        loc = self._housing_local(body.data.root_pos_w)
        return (loc[:, 0].abs() < c.out_hx) & (loc[:, 1].abs() < c.out_hy) \
            & (loc[:, 2] > 0.004) & (loc[:, 2] < c.roof_z1 - 0.002)

    def distractors_clear(self) -> torch.Tensor:
        """(N,) bool: neither the ketchup box nor the milk cube is inside the vault."""
        return ~self.in_vault(self.ketchup) & ~self.in_vault(self.milk)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every judged body |lin vel| below `settle_speed`."""
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in (self.shuttle, self.pudding, self.ketchup, self.milk)],
                        dim=1)
        return (v < self.cfg.settle_speed).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.shuttle, self.pudding, self.ketchup, self.milk)],
                        dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        live_load = self.in_gallery(self.pudding) & self.in_pocket(self.pudding) & fin
        self._loaded |= live_load
        px = self._housing_local(self.pudding.data.root_pos_w)[:, 0]
        self._conveyed |= live_load & (px > self.cfg.conv_x_min)
        self._deposited |= self.in_chamber(self.pudding) & fin
        self._homed |= self._deposited & self.home() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: pudding resting inside the lower chamber, both distractors
        outside the vault, shuttle home, everything settled and finite. All clauses
        are live physical outcomes."""
        self._update_latches()
        return self.in_chamber(self.pudding) & self.distractors_clear() & self.home() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*loaded + 0.15*conveyed + 0.30*deposited +
        0.10*homed (all latched; ~0 for doing nothing — an idle vault latches
        nothing), capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._loaded.float() + c.w_conv * self._conveyed.float()
                + c.w_dep * self._deposited.float()
                + c.w_home * self._homed.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="shuttle_vault", robot="null"))
