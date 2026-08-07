"""BendGalleryScene — the cup cannot be picked up: it must be THREADED out of a roofed
L-gallery (piano-movers style) before it can be placed.

Derived from rlbench/pick_up_cup ("pick up the cup": a free cup stands on a table, the
robot grasps it and lifts it — one grasp, one vertical lift, in open space). Here the cup
is the head of a rigid long-handled DIPPER (400 mm overall) whose handle starts deep in a
dead-end ROOFED tunnel of a kinematic gallery. A straight lift is mechanically blocked by
the roof; the dipper is longer than the diagonal of the only open-top pocket (the corner
PLAZA), so there is NO reachable pose inside the gallery from which the whole dipper can
rise freely. The seed's strategy is physically void: the object must first be EXTRACTED by
confined planar maneuvering — slid out of the dead-end tunnel into the plaza, rotated ~90
degrees through the corner (a motion that only fits because the plaza relieves the bend;
the numeric config-space check passes with >= 16 mm clearance), threaded head-first down a
second roofed tunnel, and delivered into the open YARD — and only then can it be placed:
success is the cup standing UPRIGHT on the PAD marking with the ENTIRE dipper out of the
gallery. Gallery pose (xy + yaw), handle insertion depth, and pad pose all jitter.

Assets are fully procedural:
  - gallery: KINEMATIC compound (walls 20 mm thick, 120 mm tall + roof slabs whose
    underside is at 105 mm) forming, in the gallery frame:
      tunnel1  x in [-0.24, 0], y in [0.08, 0.26]  (dead-ended at x = -0.24, ROOFED)
      plaza    x in [0, 0.26],  y in [0, 0.26]     (OPEN TOP — the only working pocket)
      tunnel2  x in [0.08, 0.26], y in [-0.15, 0]  (ROOFED, opens to the yard at y = -0.15)
      yard     y < -0.15                            (open floor; the pad lies here)
  - dipper: DYNAMIC compound rigid body, 400 mm long: a 310 x 50 x 50 mm handle bar and,
    at the head, an open-top CUP (octagon of eight 8 mm wall plates, outer diameter
    ~96 mm, 80 mm tall, plus a bottom plate). Root origin = rod centre, +x toward the
    cup, resting height 25 mm. Body-frame +z up == cup upright.
  - pad: a KINEMATIC 160 x 160 mm grey square marking flush on the yard floor
    (collision-free — purely a painted zone).

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.25 * cleared — the handle tail EVER fully out of the dead-end tunnel (tail past the
                   tunnel1 mouth; geometrically this can only happen MID-TURN, once the
                   head has already swung toward the exit — a straight rod can never
                   clear, its head would overrun the plaza first)          (latched)
  0.20 * entered — the cup EVER inside the exit tunnel (the corner rotation completed
                   and threading begun)                                    (latched)
  0.25 * emerged — the cup EVER out past the tunnel2 exit into the yard    (latched)
  1.0 iff success() — the ENTIRE dipper out of the gallery (every sample point past the
                   exit line), the cup centred on the pad (xy within `pad_tol`), upright
                   (body +z within `upright_tol_deg` of world up), settled and finite.
                   Non-success capped at 0.75.
All success clauses are live physical outcomes; the latches only preserve credit for
stages genuinely passed through. A null policy latches nothing (score ~0), and the
seed-family naive move — lift the cup straight up — is blocked by the roof itself.

Honesty geometry (asserted in `__post_init__`):
  - the dipper is LONGER than the plaza diagonal (+20 mm): no pose inside the gallery
    leaves the whole dipper unroofed, so it can never be simply lifted out;
  - yet no more than diagonal +45 mm: the corner pass validated by the config-space
    check (BFS over (x, y, yaw), footprint inflated 16 mm, PASS) stays in envelope;
  - cup and handle clear the tunnel width and the roof height with real margins;
  - at every insertion depth the cup head is exposed in the open plaza (graspable) and
    stays clear of the far plaza wall;
  - the pad is placed so that a cup-on-pad pose with the tail still inside the gallery
    EXISTS (the full-extraction clause is load-bearing, not vacuous) and the pad square
    itself lies wholly beyond the exit line.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def _box(stage, path: str, *, center, size, color, contact_offset: float, yaw_deg: float = 0.0):
    """One collidable box child: translate (+ optional z-orient) + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg != 0.0:
        half = math.radians(yaw_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
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


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on the compound root: explicit mass, mild damping so
    the dipper settles promptly, no sleeping while velocities are judged,
    depenetration cap, iterated solver."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.15)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _rigid_kinematic(root) -> None:
    """Kinematic rigid-body armor on the compound root (teleportable at reset)."""
    from pxr import UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(50.0)


def _spawn_gallery(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC gallery at `prim_path`. Root origin = the plaza's south-west
    interior corner at floor level (the frame all layout numbers live in). Children:
    seven wall slabs + two roof slabs (children of one body never self-collide)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_kinematic(root)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    p, w, t1, t2, t = c.plaza, c.tunnel_w, c.t1_len, c.t2_len, c.wall_t
    hw, rlo, rt = c.wall_h, c.roof_lo, c.roof_t
    co = c.contact_offset
    ymouth = p - w                     # tunnel1 south interior wall line (y = 0.08)
    walls = [
        # (name, xmin, xmax, ymin, ymax) — all height hw from the floor
        ("north", -t1 - t, p + t, p, p + t),
        ("cap", -t1 - t, -t1, ymouth - t, p),
        ("t1_south", -t1 - t, 0.0, ymouth - t, ymouth),
        ("west", -t, 0.0, -t, ymouth),
        ("south", -t, p - w - t, -t, 0.0),
        ("t2_west", p - w - t, p - w, -t2, 0.0),
        ("east", p, p + t, -t2, p),
    ]
    kids = []
    for name, x0, x1, y0, y1 in walls:
        kids.append(_box(stage, f"{prim_path}/wall_{name}",
                         center=((x0 + x1) / 2, (y0 + y1) / 2, hw / 2),
                         size=(x1 - x0, y1 - y0, hw), color=c.wall_color,
                         contact_offset=co))
    roofs = [
        ("t1", -t1 - t, 0.0, ymouth - t, p + t),
        ("t2", p - w - t, p + t, -t2, 0.0),
    ]
    for name, x0, x1, y0, y1 in roofs:
        kids.append(_box(stage, f"{prim_path}/roof_{name}",
                         center=((x0 + x1) / 2, (y0 + y1) / 2, rlo + rt / 2),
                         size=(x1 - x0, y1 - y0, rt), color=c.roof_color,
                         contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_dipper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC dipper at `prim_path`. Root origin = rod centre on the rod
    axis at handle mid-height; +x toward the cup head; resting height = handle_h/2.
    Children: handle bar + cup bottom plate + eight octagon wall plates."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass)
    mat = _phys_material(stage, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    half_l = c.length / 2                       # 0.20
    cup_d = 2 * c.cup_r                          # nominal cup diameter
    cup_cx = half_l - c.cup_r                    # cup centre on the axis (x = +0.155)
    zb = -c.handle_h / 2                         # floor line in the body frame (-0.025)
    kids = [
        _box(stage, f"{prim_path}/handle",
             center=(-(cup_d) / 2, 0.0, 0.0),
             size=(c.length - cup_d, c.handle_w, c.handle_h),
             color=c.handle_color, contact_offset=c.contact_offset),
        _box(stage, f"{prim_path}/cup_bottom",
             center=(cup_cx, 0.0, zb + c.cup_bot_t / 2),
             size=(c.cup_bot_w, c.cup_bot_w, c.cup_bot_t),
             color=c.cup_color, contact_offset=c.contact_offset),
    ]
    rmid = c.cup_r - c.cup_wall_t / 2
    seg_w = 2 * rmid * math.tan(math.pi / 8) * 1.02   # slight overlap, no gaps
    wall_h = c.cup_h - c.cup_bot_t
    for k in range(8):
        ang = k * 45.0
        a = math.radians(ang)
        kids.append(_box(
            stage, f"{prim_path}/cup_wall{k}",
            center=(cup_cx + rmid * math.cos(a), rmid * math.sin(a),
                    zb + c.cup_bot_t + wall_h / 2),
            size=(c.cup_wall_t, seg_w, wall_h),
            color=c.cup_color, contact_offset=c.contact_offset, yaw_deg=ang))
    for kid in kids:
        _bind_material(kid, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gallery" not in _SPAWNER_CACHE:

        @configclass
        class GallerySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gallery)
            plaza: float = 0.26
            tunnel_w: float = 0.18
            t1_len: float = 0.24
            t2_len: float = 0.15
            wall_t: float = 0.02
            wall_h: float = 0.12
            roof_lo: float = 0.105
            roof_t: float = 0.02
            mu_static: float = 0.20
            mu_dynamic: float = 0.15
            wall_color: tuple = (0.47, 0.52, 0.58)
            roof_color: tuple = (0.33, 0.36, 0.42)
            contact_offset: float = 0.002

        @configclass
        class DipperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_dipper)
            length: float = 0.40
            handle_w: float = 0.05
            handle_h: float = 0.05
            cup_r: float = 0.045
            cup_h: float = 0.08
            cup_wall_t: float = 0.008
            cup_bot_w: float = 0.062
            cup_bot_t: float = 0.010
            mass: float = 0.35
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            handle_color: tuple = (0.72, 0.55, 0.30)
            cup_color: tuple = (0.80, 0.15, 0.12)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["gallery"] = GallerySpawnerCfg
        _SPAWNER_CACHE["dipper"] = DipperSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BendGallerySceneCfg(BaseCfg):
    """Config for `BendGalleryScene`. The extraction constraint is enforced by the
    gallery's own roofs and by the dipper being longer than the plaza diagonal; the
    rubric only reads out states the geometry makes meaningful (see honesty asserts)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    pad_tol: float = tunable(0.050)          # |cup centre - pad centre| xy, world (m)
    upright_tol_deg: float = tunable(10.0)   # body +z within this of world up
    out_margin: float = tunable(0.010)       # exit line = y_local < -(t2_len + out_margin)
    settle_lin: float = tunable(0.030)       # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.20)        # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    gallery_jitter: float = tunable(0.025)   # gallery xy jitter (+/- m)
    gallery_yaw_deg: float = tunable(28.0)   # gallery yaw jitter (+/- deg)
    depth_lo: float = tunable(-0.225)        # handle tail x in the gallery frame, deepest
    depth_hi: float = tunable(-0.155)        # handle tail x, shallowest
    rod_lat_jitter: float = tunable(0.008)   # dipper lateral jitter in the tunnel (+/- m)
    rod_yaw_jitter_deg: float = tunable(1.5)  # dipper yaw jitter about the tunnel axis (+/- deg)
    pad_jitter: float = tunable(0.035)       # pad xy jitter in the gallery frame (+/- m)

    # --- info: gallery structure (gallery frame: plaza = [0,plaza]^2, floor z = 0) ---------------
    plaza: float = info(0.26)                # plaza side (open top) — diagonal 0.368 < rod 0.40
    tunnel_w: float = info(0.18)             # interior width of both tunnels
    t1_len: float = info(0.24)               # dead-end tunnel (west, top-aligned), ROOFED
    t2_len: float = info(0.15)               # exit tunnel (south, right-aligned), ROOFED
    wall_t: float = info(0.02)
    wall_h: float = info(0.12)
    roof_lo: float = info(0.105)             # roof slab underside height
    roof_t: float = info(0.02)
    gallery_pos: tuple = info((0.36, 0.04))  # world xy of the gallery frame origin
    gal_mu_static: float = info(0.20)        # gallery material (low: walls guide, not grab)
    gal_mu_dynamic: float = info(0.15)
    wall_color: tuple = info((0.47, 0.52, 0.58))
    roof_color: tuple = info((0.33, 0.36, 0.42))
    # --- info: dipper ----------------------------------------------------------------------------
    rod_len: float = info(0.40)              # overall length (tail tip -> cup rim)
    handle_w: float = info(0.05)
    handle_h: float = info(0.05)
    cup_r: float = info(0.045)               # nominal cup radius (octagon corners ~0.048)
    cup_h: float = info(0.08)
    cup_wall_t: float = info(0.008)
    cup_bot_w: float = info(0.062)
    cup_bot_t: float = info(0.010)
    rod_mass: float = info(0.35)
    rod_mu_static: float = info(0.35)
    rod_mu_dynamic: float = info(0.30)
    handle_color: tuple = info((0.72, 0.55, 0.30))
    cup_color: tuple = info((0.80, 0.15, 0.12))
    # --- info: pad (gallery frame; the yard is y < -t2_len) --------------------------------------
    pad_local: tuple = info((0.17, -0.33))   # pad centre in the gallery frame
    pad_half: float = info(0.080)
    pad_t: float = info(0.004)
    pad_color: tuple = info((0.30, 0.30, 0.33))
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.40)            # ground material (average-combined with bodies)
    # rubric weights (0.25 + 0.20 + 0.25 = 0.70 <= the non-success cap 0.75)
    w_cleared: float = info(0.25)
    w_entered: float = info(0.20)
    w_emerged: float = info(0.25)

    def __post_init__(self) -> None:
        diag = self.plaza * math.sqrt(2.0)
        cup_eff = 2 * math.hypot(self.cup_r + self.cup_wall_t / 2,
                                 self.cup_r * math.tan(math.pi / 8))  # octagon corner dia
        # never liftable inside: the rod does NOT fit in the only open-top pocket ...
        assert self.rod_len > diag + 0.02, "rod must overhang the plaza diagonal"
        # ... yet stays inside the corner-pass envelope validated by the config-space BFS
        # (P=0.26, W=0.18, T1=0.24, T2=0.15, L=0.40, handle 0.05, cup dia 0.09 inflated
        #  by 2 x 16 mm: PASS) — cup/handle must clear tunnel and roof with margin:
        assert self.rod_len < diag + 0.045, "rod too long for the validated corner pass"
        assert self.tunnel_w - cup_eff > 0.04, "cup must clear the tunnel width"
        assert self.tunnel_w - self.handle_w > 0.10, "handle must clear the tunnel width"
        # rod rests with its underside on the floor, so the cup top sits at cup_h
        assert self.roof_lo - self.cup_h > 0.015, "cup must pass under the roof"
        assert self.wall_h > self.roof_lo, "walls must reach the roof slabs"
        # the rod never fits inside tunnel1, and the cup head is exposed in the open
        # plaza at EVERY insertion depth, clear of the far plaza wall:
        assert self.rod_len > self.t1_len + 0.10, "rod must overhang the dead-end tunnel"
        assert self.depth_lo > -(self.t1_len - 0.012), "tail would start inside the cap wall"
        assert self.depth_lo + self.rod_len - 2 * self.cup_r > 0.01, \
            "cup head must be exposed in the plaza at max depth"
        assert self.depth_hi + self.rod_len < self.plaza - 0.01, \
            "cup head must stay clear of the far plaza wall"
        # exit line and pad: the pad lies wholly beyond the exit line, but close enough
        # that a cup-on-pad pose with the tail still inside the gallery EXISTS (so the
        # full-extraction clause of success() is load-bearing):
        exit_y = -(self.t2_len + self.out_margin)
        assert self.pad_local[1] + self.pad_half + self.pad_jitter < exit_y - 0.01, \
            "pad must lie wholly beyond the exit line"
        tail_from_cup = self.rod_len - self.cup_r  # 0.355
        assert self.pad_local[1] + self.pad_jitter + tail_from_cup > exit_y + 0.02, \
            "full-extraction clause would be vacuous at this pad distance"
        # the `cleared` latch is unreachable for a straight rod (head would overrun the
        # plaza first): clearing the tail REQUIRES the corner rotation to have begun
        assert 0.01 + self.rod_len > self.plaza - 0.01, \
            "`cleared` must be impossible for an axis-aligned rod"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bend_gallery")
class BendGalleryScene(BaseScene):
    cfg: BendGallerySceneCfg

    def __init__(self, cfg: BendGallerySceneCfg | None = None) -> None:
        super().__init__(cfg or BendGallerySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        gallery_spawn = cls["gallery"](
            plaza=c.plaza, tunnel_w=c.tunnel_w, t1_len=c.t1_len, t2_len=c.t2_len,
            wall_t=c.wall_t, wall_h=c.wall_h, roof_lo=c.roof_lo, roof_t=c.roof_t,
            mu_static=c.gal_mu_static, mu_dynamic=c.gal_mu_dynamic,
            wall_color=c.wall_color, roof_color=c.roof_color,
            contact_offset=c.contact_offset)
        dipper_spawn = cls["dipper"](
            length=c.rod_len, handle_w=c.handle_w, handle_h=c.handle_h,
            cup_r=c.cup_r, cup_h=c.cup_h, cup_wall_t=c.cup_wall_t,
            cup_bot_w=c.cup_bot_w, cup_bot_t=c.cup_bot_t, mass=c.rod_mass,
            mu_static=c.rod_mu_static, mu_dynamic=c.rod_mu_dynamic,
            handle_color=c.handle_color, cup_color=c.cup_color,
            contact_offset=c.contact_offset)

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
            "gallery": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gallery",
                spawn=gallery_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.gallery_pos[0], c.gallery_pos[1], 0.0)),
            ),
            "dipper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dipper",
                spawn=dipper_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.gallery_pos[0] - 0.02, c.gallery_pos[1] + c.plaza - c.tunnel_w / 2,
                         c.handle_h / 2 + 0.003)),
            ),
            # pad: kinematic, COLLISION-FREE (no collision_props -> no collider): a
            # painted zone the dipper slides over, repositionable at reset.
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.pad_half, 2 * c.pad_half, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.gallery_pos[0] + c.pad_local[0],
                         c.gallery_pos[1] + c.pad_local[1], c.pad_t / 2 - 0.003)),
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
        self.gallery: RigidObject = env.iscene["gallery"]
        self.dipper: RigidObject = env.iscene["dipper"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._cleared = torch.zeros(n, dtype=torch.bool, device=dev)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._emerged = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: gallery pose (xy jitter + yaw), dipper laid in tunnel1 at a
        random insertion depth (+ lateral/yaw jitter, all in the gallery frame), pad
        repositioned in the yard (gallery frame + jitter), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def rnd(k: float) -> torch.Tensor:
            return (torch.rand(m, device=dev) * 2 - 1) * k

        def write(body, x, y, z, q) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3:7] = q
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        gx = c.gallery_pos[0] + rnd(c.gallery_jitter)
        gy = c.gallery_pos[1] + rnd(c.gallery_jitter)
        gyaw = rnd(math.radians(c.gallery_yaw_deg))
        gq = _qz(gyaw)
        write(self.gallery, gx, gy, torch.zeros(m, device=dev), gq)

        # dipper: gallery-local pose -> world
        tail_x = c.depth_lo + torch.rand(m, device=dev) * (c.depth_hi - c.depth_lo)
        cx = tail_x + c.rod_len / 2
        cy = c.plaza - c.tunnel_w / 2 + rnd(c.rod_lat_jitter)
        local = torch.stack([cx, cy, torch.zeros(m, device=dev)], dim=-1)
        wpos = torch.stack([gx, gy, torch.zeros(m, device=dev)], dim=-1) + _qapply(gq, local)
        ryaw = gyaw + rnd(math.radians(c.rod_yaw_jitter_deg))
        write(self.dipper, wpos[:, 0], wpos[:, 1],
              torch.full((m,), c.handle_h / 2 + 0.003, device=dev), _qz(ryaw))

        # pad: gallery-local (jittered) -> world, free yaw (square marking)
        px = c.pad_local[0] + rnd(c.pad_jitter)
        py = c.pad_local[1] + rnd(c.pad_jitter)
        plocal = torch.stack([px, py, torch.zeros(m, device=dev)], dim=-1)
        ppos = torch.stack([gx, gy, torch.zeros(m, device=dev)], dim=-1) + _qapply(gq, plocal)
        write(self.pad, ppos[:, 0], ppos[:, 1],
              torch.full((m,), c.pad_t / 2 - 0.003, device=dev),
              _qz(rnd(math.pi)))

        self._cleared[env_ids] = False
        self._entered[env_ids] = False
        self._emerged[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gallery": self.gallery.data.root_state_w[env_ids].clone(),
            "dipper": self.dipper.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "cleared": self._cleared[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "emerged": self._emerged[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gallery.write_root_state_to_sim(state["gallery"], env_ids)
        self.dipper.write_root_state_to_sim(state["dipper"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self._cleared[env_ids] = state["cleared"]
        self._entered[env_ids] = state["entered"]
        self._emerged[env_ids] = state["emerged"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey-blue GALLERY stands on the floor: a dead-end tunnel and an exit "
            f"tunnel (both {c.tunnel_w * 1000:.0f} mm wide and ROOFED over at "
            f"{c.roof_lo * 1000:.0f} mm) meeting at right angles through an open-top "
            f"corner PLAZA ({c.plaza * 1000:.0f} mm square). Inside lies a DIPPER — a "
            f"red open cup ({2 * c.cup_r * 1000:.0f} mm across, {c.cup_h * 1000:.0f} mm "
            f"tall) rigidly fixed to a long wooden handle, {c.rod_len * 1000:.0f} mm "
            f"overall — with its handle deep in the dead-end tunnel and only the cup "
            f"head exposed in the plaza. How deep it starts, and the pose of the "
            f"gallery and of the grey PAD marking out in the open YARD beyond the exit "
            f"tunnel, vary per episode. The dipper is LONGER than the plaza diagonal "
            f"and both tunnels are roofed, so it can never be simply lifted out: it "
            f"must be slid out of the dead-end tunnel, rotated through the plaza "
            f"corner, and threaded head-first down the exit tunnel into the yard.\n"
            f"Goal: get the ENTIRE dipper out of the gallery and stand the cup UPRIGHT "
            f"on the grey pad (cup centre within {c.pad_tol * 1000:.0f} mm of the pad "
            f"centre), everything settled. A cup on the pad with any part of the "
            f"handle still inside the gallery does not count, nor does a tipped or "
            f"upside-down cup, or anything still moving. Only the final state is "
            f"judged — how you maneuver it is up to you."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the long-handled red cup out of the dead-end tunnel, rotate it "
            "through the open corner plaza, thread it out through the roofed exit "
            "tunnel, and stand the cup upright on the grey pad in the yard — the "
            "whole handle must end outside the gallery."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _gal_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the gallery frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.gallery.data.root_quat_w),
                       pos_w - self.gallery.data.root_pos_w)

    def _rod_pts_local(self) -> torch.Tensor:
        """(N,6,3) gallery-frame positions of six sample points along the rod axis:
        tail tip, -half, centre, +half, cup centre, head tip."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        s = torch.tensor([-c.rod_len / 2, -c.rod_len / 4, 0.0, c.rod_len / 4,
                          c.rod_len / 2 - c.cup_r, c.rod_len / 2], device=dev)
        offs = torch.zeros(len(s), 3, device=dev)
        offs[:, 0] = s
        q = self.dipper.data.root_quat_w.unsqueeze(1).expand(n, len(s), 4)
        pts_w = self.dipper.data.root_pos_w.unsqueeze(1) + _qapply(q, offs.unsqueeze(0).expand(n, len(s), 3))
        gq = _qinv(self.gallery.data.root_quat_w).unsqueeze(1).expand(n, len(s), 4)
        return _qapply(gq, pts_w - self.gallery.data.root_pos_w.unsqueeze(1))

    def tail_local(self) -> torch.Tensor:
        """(N,3) gallery-frame position of the handle tail tip."""
        return self._rod_pts_local()[:, 0]

    def cup_local(self) -> torch.Tensor:
        """(N,3) gallery-frame position of the cup centre."""
        return self._rod_pts_local()[:, 4]

    def cup_pos_w(self) -> torch.Tensor:
        """(N,3) world position of the cup centre."""
        c = self.cfg
        n = self.env.num_envs
        off = torch.tensor([c.rod_len / 2 - c.cup_r, 0.0, 0.0],
                           device=self.env.device).expand(n, 3)
        return self.dipper.data.root_pos_w + _qapply(self.dipper.data.root_quat_w, off)

    def heading_local(self) -> torch.Tensor:
        """(N,3) rod +x axis expressed in the gallery frame."""
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        ax_w = _qapply(self.dipper.data.root_quat_w, ex)
        return _qapply(_qinv(self.gallery.data.root_quat_w), ax_w)

    def rod_up_z(self) -> torch.Tensor:
        """(N,) world-z component of the dipper's body +z axis (+1 = cup upright)."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return _qapply(self.dipper.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def extracted(self) -> torch.Tensor:
        """(N,) bool: EVERY rod sample point is past the exit line (gallery-frame
        y < -(t2_len + out_margin)) — the whole dipper is out in the yard."""
        c = self.cfg
        pts = self._rod_pts_local()
        return (pts[:, :, 1] < -(c.t2_len + c.out_margin)).all(dim=1)

    def cup_on_pad(self) -> torch.Tensor:
        """(N,) bool: cup centre within `pad_tol` (xy, world) of the pad centre."""
        d = (self.cup_pos_w()[:, :2] - self.pad.data.root_pos_w[:, :2]).norm(dim=-1)
        return d < self.cfg.pad_tol

    def upright(self) -> torch.Tensor:
        """(N,) bool: body +z within `upright_tol_deg` of world up."""
        return self.rod_up_z() > math.cos(math.radians(self.cfg.upright_tol_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: dipper linear AND angular velocity below the settle thresholds."""
        return (self.dipper.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (self.dipper.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.dipper.data.root_pos_w, self.gallery.data.root_pos_w,
                         self.pad.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        pts = self._rod_pts_local()
        tail = pts[:, 0]
        cup = pts[:, 4]
        low = self.dipper.data.root_pos_w[:, 2] - self.env_origins[:, 2] < 0.10
        # cleared: tail fully out of the dead-end tunnel (past the tunnel1 mouth) —
        # geometrically only reachable mid-turn (asserted in __post_init__)
        self._cleared |= fin & low & (tail[:, 0] > 0.01)
        # entered: cup centre inside the exit tunnel (turn completed, threading begun)
        in_t2 = (cup[:, 0] > c.plaza - c.tunnel_w + 0.005) & (cup[:, 0] < c.plaza - 0.005) \
            & (cup[:, 1] < -0.02) & (cup[:, 1] > -(c.t2_len + c.out_margin))
        self._entered |= fin & low & in_t2
        # emerged: cup centre out past the exit line into the yard
        self._emerged |= fin & low & (cup[:, 1] < -(c.t2_len + c.out_margin))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ENTIRE dipper out of the gallery (every sample point past the
        exit line), cup centred on the pad, upright, settled and finite. All clauses
        are live physical outcomes."""
        self._update_latches()
        return self.extracted() & self.cup_on_pad() & self.upright() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*cleared + 0.20*entered + 0.25*emerged (all
        latched; ~0 for doing nothing — an untouched dipper latches nothing), capped
        at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_cleared * self._cleared.float() + c.w_entered * self._entered.float()
                + c.w_emerged * self._emerged.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="bend_gallery", robot="null"))
