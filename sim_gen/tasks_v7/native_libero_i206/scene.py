"""TiltMazeScene — steer a captive ball through a switchback maze by tilting the tray,
until it drops into the sunken green pocket; the tray must return level on its own.

Derived from libero/native_libero (vendored LIBERO MJCF scenes: single-object
pick-and-place / fixture articulation, an OSC arm carries ONE object to a region and a
static BDDL on/in predicate judges the end pose). Here NOTHING is ever grasped or
carried and no end-pose can be produced by transport: the goal object is a CAPTIVE
ball sealed under a slatted cage roof inside a tilting maze tray (a hand-held
"ball-in-a-box" toy scaled up). The only interface the world offers is the tray's ONE
degree of freedom — a revolute hinge on a yoke stand — driven by pressing the paddle
tabs at the tray ends. Reaching the goal requires a closed-loop, two-stroke PLAN whose
order is physically forced by the maze topology: tilt BLUE-end-down so the ball runs
the north lane east and the diagonal baffle deflects it around the divider into the
south lane; then tilt RED-end-down so it runs the south lane west and sinks into the
recessed pocket; then release, and the keel under the tray must swing the deck back
level. Tilting red-first parks the ball harmlessly at the start wall; the pocket is
unreachable except through the bay because the divider separates the lanes and the
roof denies any approach from above.

Assets are fully procedural (one compound spawner per fixture, boxes + spheres):
  - stand: heavy DYNAMIC yoke (base slab + two posts straddling the tray) resting on
    the ground. Dynamic, not kinematic, so the hinge anchor follows reset teleports.
  - tray:  DYNAMIC compound hinged to the stand (revolute, axis = tray body y,
    travel +/- `tilt_limit_deg`). Children: deck plates with a rectangular pocket
    hole, recessed green pocket floor, perimeter walls, center divider (stops short of
    the east end -> the bay), diagonal baffle across the bay, 5 roof slats (gaps
    smaller than the ball — the ball can NEVER leave), blue / red press paddles
    outside the end walls, and a keel strut + bob below the hinge whose authored CoM
    returns the tray to level when released.
  - ball:  yellow 26 mm sphere, captive inside the maze.
  - spare: white 26 mm sphere resting in a small kinematic cradle on the floor beside
    the stand — a decoy: dropping/placing the WRONG ball anywhere achieves nothing,
    and the spare must be LEFT IN ITS CRADLE.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve trajectory):
  0.30 * bay      — the ball ever reached the east bay (past the divider tip) (latched)
  0.20 * crossed  — the ball ever entered the SOUTH lane band (only reachable
                    around the baffle)                                        (latched)
  0.25 * returned — ever in the south lane west of `return_x` (rolled most of
                    the way back toward the pocket)                           (latched)
  1.0 iff success() — ball SUNK in the pocket (tray-frame xy window AND centre
                    below deck level — a ball resting ON the deck or ON the roof
                    fails the sink test), tray back within `level_tol_deg` of
                    level, ball and tray settled, spare still in its cradle,
                    everything finite. Non-success capped at 0.75.
A null policy scores ~0: the ball stays at the north-lane start wall and never
latches anything.

Honesty geometry (asserted in `__post_init__`):
  - the ball fits every passage it must take, and NO roof gap passes the ball;
  - the pocket hole admits the ball and its lip retains it at the hinge travel limit;
  - the pocket window tolerances lie inside what the pocket walls physically enforce,
    and a ball on the deck fails the sink threshold by its full drop;
  - the keel's restoring moment beats the worst ball-induced heeling moment well
    inside `level_tol_deg`;
  - the tray (walls + paddles) keeps an axis-invariant clearance to the yoke posts at
    every hinge angle, and the cradle sits outside the tray's swept reach.

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


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _box(stage, path: str, *, center, size, color, contact_offset: float, yaw: float = 0.0):
    """One collidable box child: translate (+ optional z-yaw) + scale, displayColor."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw != 0.0:
        h = float(yaw) / 2.0
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, 0.0, math.sin(h))))
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


def _rigid_dynamic(root, *, mass: float, com, inertia, lin_damp: float, ang_damp: float) -> None:
    """Dynamic rigid-body armor: EXPLICIT mass + CoM + diagonal inertia (the pod
    leaves the CoM at the body origin and derives unknown inertia otherwise — the
    keel return and the hinge servo both need authored values), damping, no sleeping,
    depenetration cap, iterated solver (vel iters 4: rolling-creep floor)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Heavy DYNAMIC yoke stand: base slab on the ground + two posts straddling the
    tray in y. Dynamic so the spawn-authored hinge anchor follows reset teleports;
    heavy + grippy so presses and hinge reactions cannot move it."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, mass=c.mass, com=(0.0, 0.0, 0.02),
                   inertia=(0.06, 0.06, 0.12), lin_damp=0.5, ang_damp=0.5)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.80, 0.75)
    co = c.contact_offset
    kids = [
        _box(stage, f"{prim_path}/base", center=(0.0, 0.0, c.base_t / 2),
             size=(c.base_x, c.base_y, c.base_t), color=c.color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/post_{s}",
                         center=(0.0, sgn * c.post_y, c.base_t + c.post_h / 2),
                         size=(c.post_w, c.post_w, c.post_h),
                         color=c.post_color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The DYNAMIC maze tray. Body frame: origin ON the hinge axis at the deck-top
    plane centre; x toward the bay (blue) end, y along the hinge, z up. Children
    never self-collide, so plates may butt/overlap freely. Also authors the revolute
    hinge to the sibling stand (joint pair => tray/stand collision auto-filtered;
    real clearance is geometric and asserted in cfg)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, mass=c.mass, com=(0.0, 0.0, c.com_z),
                   inertia=(0.006, 0.020, 0.024), lin_damp=0.05, ang_damp=c.ang_damp)
    mat = _phys_material(stage, f"{prim_path}/physmat", c.mu_static, c.mu_dynamic)
    co = c.contact_offset
    hx, hy = c.inner_hx, c.inner_hy          # inner cavity half-extents
    wt, wh, td = c.wall_t, c.wall_h, c.deck_t
    pd = c.pocket_depth
    px_w, px_e = -hx, -hx + c.pocket_len     # pocket hole x span (west lane-S end)
    py_s, py_n = -hy, -c.div_hy              # pocket hole y span (south lane)
    kids = []

    def box(name, center, size, color, yaw=0.0):
        kids.append(_box(stage, f"{prim_path}/{name}", center=center, size=size,
                         color=color, contact_offset=co, yaw=yaw))

    # deck plates (top at z=0) leaving the pocket hole open
    box("deck_n", ((0.0), (py_n + hy) / 2 + 0.0, -td / 2),
        (2 * hx, hy - py_n, td), c.deck_color)
    box("deck_se", ((px_e + hx) / 2, (py_s + py_n) / 2, -td / 2),
        (hx - px_e, py_n - py_s, td), c.deck_color)
    # recessed pocket floor (top at z = -pd)
    box("pocket_floor", ((px_w + px_e) / 2, (py_s + py_n) / 2, -pd - td / 2),
        (c.pocket_len, py_n - py_s, td), c.pocket_color)
    # perimeter walls (down to the pocket floor bottom, up to wall_h)
    wz_lo, wz_hi = -pd - td, wh
    wall_sz_z = wz_hi - wz_lo
    wall_cz = (wz_lo + wz_hi) / 2
    for sgn in (1.0, -1.0):
        s = "e" if sgn > 0 else "w"
        box(f"wall_{s}", (sgn * (hx + wt / 2), 0.0, wall_cz),
            (wt, 2 * (hy + wt), wall_sz_z), c.wall_color)
        s = "n" if sgn > 0 else "s"
        box(f"wall_{s}", (0.0, sgn * (hy + wt / 2), wall_cz),
            (2 * hx, wt, wall_sz_z), c.wall_color)
    # center divider: west wall -> div_tip_x (leaves the east bay open)
    box("divider", ((-hx + c.div_tip_x) / 2, 0.0, wh / 2),
        (c.div_tip_x + hx, 2 * c.div_hy, wh), c.trim_color)
    # diagonal baffle across the bay: from the north wall down-east to the east wall
    bx0, by0 = c.baffle_x0, hy + 0.002
    bx1, by1 = hx + 0.002, c.baffle_y1
    blen = math.hypot(bx1 - bx0, by1 - by0)
    byaw = math.atan2(by1 - by0, bx1 - bx0)
    box("baffle", ((bx0 + bx1) / 2, (by0 + by1) / 2, wh / 2),
        (blen + 0.006, 2 * c.div_hy, wh), c.trim_color, yaw=byaw)
    # cage roof: slats along x, gaps < ball diameter (the ball is CAPTIVE)
    for i, yc in enumerate((-2, -1, 0, 1, 2)):
        box(f"slat_{i}", (0.0, yc * c.slat_pitch, (c.roof_lo + c.roof_hi) / 2),
            (2 * (hx + wt), c.slat_w, c.roof_hi - c.roof_lo), c.slat_color)
    # press paddles outside the end walls (blue = bay end +x, red = pocket end -x)
    for sgn, color, name in ((1.0, c.blue, "paddle_blue"), (-1.0, c.red, "paddle_red")):
        box(name, (sgn * (hx + wt + c.paddle_len / 2), 0.0, -td / 2),
            (c.paddle_len, c.paddle_w, td), color)
    # keel: strut + bob below the hinge (mass placement is authored on the root; the
    # keel geometry makes the return visually honest and carries no contacts in play)
    box("keel_strut", (0.0, 0.0, -pd - td - c.keel_strut_h / 2),
        (0.02, 0.02, c.keel_strut_h), c.trim_color)
    box("keel_bob", (0.0, 0.0, -pd - td - c.keel_strut_h - c.keel_bob_h / 2),
        (0.06, 0.05, c.keel_bob_h), c.trim_color)
    for k in kids:
        _bind_material(k, mat)

    # revolute hinge to the sibling stand: axis = tray body y through the tray origin
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-c.tilt_limit_deg))
    j.CreateUpperLimitAttr(float(c.tilt_limit_deg))
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Small KINEMATIC cradle dish for the spare ball (never jointed, teleport-safe)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    mat = _phys_material(stage, f"{prim_path}/physmat", 0.60, 0.55)
    c = cfg
    co = c.contact_offset
    ih = c.inner_half
    kids = [
        _box(stage, f"{prim_path}/plate", center=(0.0, 0.0, c.plate_t / 2),
             size=(2 * (ih + c.wall_t), 2 * (ih + c.wall_t), c.plate_t),
             color=c.color, contact_offset=co),
    ]
    for sgn in (1.0, -1.0):
        s = "p" if sgn > 0 else "n"
        kids.append(_box(stage, f"{prim_path}/wx{s}",
                         center=(sgn * (ih + c.wall_t / 2), 0.0, c.plate_t + c.wall_h / 2),
                         size=(c.wall_t, 2 * (ih + c.wall_t), c.wall_h),
                         color=c.color, contact_offset=co))
        kids.append(_box(stage, f"{prim_path}/wy{s}",
                         center=(0.0, sgn * (ih + c.wall_t / 2), c.plate_t + c.wall_h / 2),
                         size=(2 * ih, c.wall_t, c.wall_h),
                         color=c.color, contact_offset=co))
    for k in kids:
        _bind_material(k, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            mass: float = 8.0
            base_x: float = 0.30
            base_y: float = 0.30
            base_t: float = 0.03
            post_w: float = 0.05
            post_h: float = 0.125
            post_y: float = 0.125
            color: tuple = (0.45, 0.40, 0.35)
            post_color: tuple = (0.38, 0.34, 0.30)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            mass: float = 1.2
            com_z: float = -0.055
            ang_damp: float = 4.0
            inner_hx: float = 0.18
            inner_hy: float = 0.075
            wall_t: float = 0.012
            wall_h: float = 0.05
            deck_t: float = 0.012
            pocket_len: float = 0.06
            pocket_depth: float = 0.012
            div_hy: float = 0.006
            div_tip_x: float = 0.10
            baffle_x0: float = 0.098
            baffle_y1: float = -0.022
            roof_lo: float = 0.038
            roof_hi: float = 0.048
            slat_w: float = 0.014
            slat_pitch: float = 0.03
            paddle_len: float = 0.05
            paddle_w: float = 0.05
            keel_strut_h: float = 0.055
            keel_bob_h: float = 0.04
            hinge_h: float = 0.14
            tilt_limit_deg: float = 12.0
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            deck_color: tuple = (0.72, 0.72, 0.75)
            wall_color: tuple = (0.35, 0.38, 0.45)
            trim_color: tuple = (0.24, 0.26, 0.32)
            slat_color: tuple = (0.55, 0.58, 0.62)
            pocket_color: tuple = (0.10, 0.65, 0.20)
            blue: tuple = (0.15, 0.35, 0.85)
            red: tuple = (0.85, 0.15, 0.12)
            contact_offset: float = 0.002

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            inner_half: float = 0.033
            wall_t: float = 0.012
            wall_h: float = 0.030
            plate_t: float = 0.008
            color: tuple = (0.30, 0.25, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, tray=TraySpawnerCfg,
                              cradle=CradleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class TiltMazeSceneCfg(BaseCfg):
    """Config for `TiltMazeScene`. The pocket window is enforced by the pocket walls
    themselves, the sink threshold by the pocket depth, and the level tolerance by
    the keel's authored righting moment (see the honesty asserts)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    pocket_x_tol: float = tunable(0.022)     # |ball_x - pocket centre x| in the tray frame
    pocket_y_tol: float = tunable(0.026)     # |ball_y - pocket centre y| in the tray frame
    sink_z: float = tunable(0.007)           # ball centre BELOW this tray-frame height = sunk
    level_tol_deg: float = tunable(3.5)      # |hinge tilt| at success
    settle_speed: float = tunable(0.12)      # max ball |lin vel| when judging (m/s)
    tray_calm: float = tunable(0.5)          # max tray |ang vel| when judging (rad/s)
    bay_x: float = tunable(0.105)            # latch: ball past the divider tip (bay)
    lane_s_y: float = tunable(-0.019)        # latch: ball centre inside the SOUTH lane band
    return_x: float = tunable(-0.06)         # latch: south lane, rolled back west of this
    cradle_tol: float = tunable(0.030)       # spare ball xy distance from cradle centre

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    asm_jitter: float = tunable(0.05)        # assembly (stand+tray+cradle) xy jitter (+/- m)
    yaw_deg: float = tunable(180.0)          # assembly free yaw (+/- deg)
    ball_x_lo: float = tunable(-0.158)       # ball start zone, tray-frame x (north lane, west)
    ball_x_hi: float = tunable(-0.105)
    ball_y_jit: float = tunable(0.015)       # ball start y jitter about the north lane centre

    # --- info: world layout (ground z = 0) ------------------------------------------------------
    asm_pos: tuple = info((0.45, 0.0))       # nominal stand centre
    cradle_off: tuple = info((0.0, -0.34))   # cradle centre in the STAND frame
    hinge_h: float = info(0.14)              # hinge height above the stand origin (base bottom)
    # --- info: tray structure (tray body frame, origin ON the hinge axis at deck top) -----------
    inner_hx: float = info(0.18)
    inner_hy: float = info(0.075)
    wall_t: float = info(0.012)
    wall_h: float = info(0.05)
    deck_t: float = info(0.012)
    pocket_len: float = info(0.06)           # pocket hole spans x in [-inner_hx, -inner_hx+len]
    pocket_depth: float = info(0.012)
    div_hy: float = info(0.006)              # divider half-thickness
    div_tip_x: float = info(0.10)            # divider ends here; bay = east of this
    baffle_x0: float = info(0.098)           # baffle start on the north wall
    baffle_y1: float = info(-0.022)          # baffle end on the east wall
    roof_lo: float = info(0.038)
    roof_hi: float = info(0.048)
    slat_w: float = info(0.014)
    slat_pitch: float = info(0.03)
    tilt_limit_deg: float = info(12.0)
    tray_mass: float = info(1.2)
    tray_com_z: float = info(-0.055)         # authored CoM below the hinge -> keel return
    # --- info: balls ----------------------------------------------------------------------------
    ball_r: float = info(0.013)
    ball_mass: float = info(0.012)
    ball_color: tuple = info((0.95, 0.85, 0.10))
    spare_color: tuple = info((0.92, 0.92, 0.92))
    # --- info: misc -----------------------------------------------------------------------------
    lane_n_y: float = info(0.0405)           # north lane centreline
    contact_offset: float = info(0.002)
    ground_mu: float = info(0.60)
    # rubric weights (0.30 + 0.20 + 0.25 = 0.75 = the non-success cap)
    w_bay: float = info(0.30)
    w_cross: float = info(0.20)
    w_return: float = info(0.25)

    def __post_init__(self) -> None:
        r, d = self.ball_r, 2 * self.ball_r
        lane_w = self.inner_hy - self.div_hy          # each lane's clear width
        # passages: lanes, bay, roof headroom all pass the ball
        assert lane_w > d + 0.010, "lane too narrow for the ball"
        assert self.inner_hx - self.div_tip_x > d + 0.010, "bay too narrow for the ball"
        assert self.roof_lo > d + 0.010, "no headroom under the roof slats"
        # the roof is a CAGE: no slat gap passes the ball (edge slat to wall included)
        gap = self.slat_pitch - self.slat_w
        edge_gap = self.inner_hy - (2 * self.slat_pitch + self.slat_w / 2)
        assert max(gap, edge_gap) < d - 0.006, "a roof gap could pass the ball"
        # pocket admits the ball, and its lip retains it even at the hinge travel limit
        assert self.pocket_len > d + 0.010 and 2 * (self.inner_hy - self.div_hy) > d, \
            "pocket hole too small"
        h = self.pocket_depth
        assert h < r, "lip taller than ball centre"
        esc = math.degrees(math.atan2(math.sqrt(2 * r * h - h * h), r - h))
        assert esc > 2 * self.tilt_limit_deg, "pocket lip cannot retain the ball"
        # pocket window tolerances sit inside what the pocket walls enforce; a ball on
        # the deck fails the sink test by its full drop
        px_c = -self.inner_hx + self.pocket_len / 2
        assert self.pocket_x_tol >= self.pocket_len / 2 - r + 0.003, \
            "pocket_x_tol rejects balls the pocket physically holds"
        assert self.pocket_x_tol < self.pocket_len / 2 + r, "pocket_x_tol leaks past the hole"
        y_half = (self.inner_hy - self.div_hy) / 2
        assert self.pocket_y_tol >= y_half - r + 0.003 and self.pocket_y_tol < y_half + r, \
            "pocket_y_tol inconsistent with the hole"
        z_in, z_on = r - self.pocket_depth, r     # sunk vs on-deck ball centre heights
        assert z_in + 0.003 < self.sink_z < z_on - 0.004, \
            "sink_z must separate sunk from on-deck by a clear margin"
        # keel: righting moment beats the worst ball heel well inside level_tol
        heel = math.degrees(math.asin(
            (self.ball_mass * (self.inner_hx + self.wall_t))
            / (self.tray_mass * abs(self.tray_com_z))))
        assert heel < 0.7 * self.level_tol_deg, "keel too weak for level_tol_deg"
        assert self.level_tol_deg < self.tilt_limit_deg / 2, "level_tol must mean level"
        # latch lines are reachable only via the intended route
        assert self.bay_x > self.div_tip_x and self.bay_x < self.inner_hx - r, \
            "bay_x must lie inside the bay"
        assert self.lane_s_y < -self.div_hy - r + 0.001, "lane_s_y not strictly south"
        assert -self.inner_hx + self.pocket_len < self.return_x < -0.02, \
            "return_x must lie between the pocket and midfield"
        # ball start zone: north lane, west half, clear of walls and divider tip
        assert self.ball_x_lo > -self.inner_hx + r + 0.003, "start zone in the west wall"
        assert self.ball_x_hi < 0.0, "start zone must stay in the west half"
        assert self.ball_y_jit + r + 0.002 < (self.inner_hy - self.div_hy) / 2, \
            "ball start jitter escapes the north lane"
        # yoke clearance is axis-invariant (rotation about y preserves y extents):
        # posts inner face vs tray outer wall face, at every hinge angle
        post_inner = 0.125 - 0.05 / 2
        assert post_inner > self.inner_hy + self.wall_t + 0.008, "tray can strike the posts"
        # cradle sits outside the tray's swept reach (tray tip incl. paddle)
        reach = self.inner_hx + self.wall_t + 0.05
        off = math.hypot(*self.cradle_off)
        assert off - self.asm_jitter * 0 > reach + 0.07, "cradle under the tray sweep"
        # the spare-in-cradle window is inside the cradle walls
        assert self.cradle_tol < 0.033 + 0.012, "cradle_tol leaks outside the cradle"

    @property
    def pocket_center(self) -> tuple:
        return (-self.inner_hx + self.pocket_len / 2, -(self.inner_hy + self.div_hy) / 2)


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("tilt_maze")
class TiltMazeScene(BaseScene):
    cfg: TiltMazeSceneCfg

    def __init__(self, cfg: TiltMazeSceneCfg | None = None) -> None:
        super().__init__(cfg or TiltMazeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        ball_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.05,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.40, dynamic_friction=0.35, restitution=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
        )
        ax, ay = c.asm_pos
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=cls["stand"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.001)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=cls["tray"](
                    mass=c.tray_mass, com_z=c.tray_com_z,
                    inner_hx=c.inner_hx, inner_hy=c.inner_hy, wall_t=c.wall_t,
                    wall_h=c.wall_h, deck_t=c.deck_t, pocket_len=c.pocket_len,
                    pocket_depth=c.pocket_depth, div_hy=c.div_hy,
                    div_tip_x=c.div_tip_x, baffle_x0=c.baffle_x0,
                    baffle_y1=c.baffle_y1, roof_lo=c.roof_lo, roof_hi=c.roof_hi,
                    slat_w=c.slat_w, slat_pitch=c.slat_pitch, hinge_h=c.hinge_h,
                    tilt_limit_deg=c.tilt_limit_deg, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.001 + c.hinge_h)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                    **ball_props),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax - 0.13, ay + c.lane_n_y, 0.001 + c.hinge_h + c.ball_r + 0.002)),
            ),
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle",
                spawn=cls["cradle"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.cradle_off[0], ay + c.cradle_off[1], 0.0)),
            ),
            "spare": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spare",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.spare_color),
                    **ball_props),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.cradle_off[0], ay + c.cradle_off[1], 0.008 + c.ball_r + 0.002)),
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
                # external hinge torques must be applied every TGS iteration or the
                # press plant is silently under-driven
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.stand: RigidObject = env.iscene["stand"]
        self.tray: RigidObject = env.iscene["tray"]
        self.ball: RigidObject = env.iscene["ball"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.spare: RigidObject = env.iscene["spare"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches (partial credit survives transients; success is judged live)
        self._bay = torch.zeros(n, dtype=torch.bool, device=dev)
        self._crossed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._returned = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: teleport the WHOLE hinged assembly rigidly (stand + tray at
        the exact spawn relative pose, same yaw — the hinge anchors are body-relative
        because both bodies are dynamic), cradle + spare with it, ball to a jittered
        start in the north lane's west half; latches cleared."""
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

        ax = c.asm_pos[0] + rnd(c.asm_jitter)
        ay = c.asm_pos[1] + rnd(c.asm_jitter)
        yaw = rnd(math.radians(c.yaw_deg))
        q = _qz(yaw)
        z0 = torch.full((m,), 0.001, device=dev)
        write(self.stand, ax, ay, z0, q)
        write(self.tray, ax, ay, z0 + c.hinge_h, q)
        # cradle + spare ride in the stand frame
        off = torch.tensor(c.cradle_off, device=dev).expand(m, 2)
        off_w = _qapply(q, torch.cat([off, torch.zeros(m, 1, device=dev)], dim=-1))
        write(self.cradle, ax + off_w[:, 0], ay + off_w[:, 1],
              torch.zeros(m, device=dev), q)
        write(self.spare, ax + off_w[:, 0], ay + off_w[:, 1],
              torch.full((m,), 0.008 + c.ball_r + 0.002, device=dev), q)
        # ball: tray-frame start in the north lane west half (tray is level at reset)
        bx = c.ball_x_lo + torch.rand(m, device=dev) * (c.ball_x_hi - c.ball_x_lo)
        by = c.lane_n_y + rnd(c.ball_y_jit)
        loc = torch.stack([bx, by, torch.full((m,), c.ball_r + 0.002, device=dev)], dim=-1)
        loc_w = _qapply(q, loc)
        write(self.ball, ax + loc_w[:, 0], ay + loc_w[:, 1],
              c.hinge_h + 0.001 + loc_w[:, 2], q)

        self._bay[env_ids] = False
        self._crossed[env_ids] = False
        self._returned[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "cradle": self.cradle.data.root_state_w[env_ids].clone(),
            "spare": self.spare.data.root_state_w[env_ids].clone(),
            "bay": self._bay[env_ids].clone(),
            "crossed": self._crossed[env_ids].clone(),
            "returned": self._returned[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self.cradle.write_root_state_to_sim(state["cradle"], env_ids)
        self.spare.write_root_state_to_sim(state["spare"], env_ids)
        self._bay[env_ids] = state["bay"]
        self._crossed[env_ids] = state["crossed"]
        self._returned[env_ids] = state["returned"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A tabletop TILT MAZE stands on a wooden yoke: a rectangular tray "
            f"({2 * (c.inner_hx + c.wall_t) * 1000:.0f} x "
            f"{2 * (c.inner_hy + c.wall_t) * 1000:.0f} mm) hangs between two posts on a "
            f"single hinge and can rock about that one axis by up to "
            f"{c.tilt_limit_deg:.0f} deg each way; a keel under the tray returns it to "
            f"level when nothing pushes it. Flat press TABS stick out past the end "
            f"walls: a BLUE tab at one end, a RED tab at the other — pressing a tab "
            f"down tilts the tray that way. Inside, under a slatted cage roof (the "
            f"gaps are too narrow for anything to pass — the contents are sealed in), "
            f"a wall divides the tray into two lanes joined only at the BAY under the "
            f"blue tab's end, where a diagonal baffle guides a rolling ball from the "
            f"north lane around into the south lane. A YELLOW ball "
            f"({2 * c.ball_r * 1000:.0f} mm) starts in the north lane near the red "
            f"end; its start spot, and the whole stand's position and heading, vary "
            f"every episode. In the south lane's far corner (red end) the deck drops "
            f"into a sunken GREEN POCKET {c.pocket_depth * 1000:.0f} mm deep. A WHITE "
            f"spare ball rests in a small dark cradle dish on the floor beside the "
            f"stand.\n"
            f"Goal: get the YELLOW ball into the green pocket and let the tray come to "
            f"rest level. Tilt the tray blue-end-down so the ball rolls up the north "
            f"lane and the baffle carries it around into the south lane, then tilt "
            f"red-end-down so it rolls back and drops into the pocket, then let go. "
            f"Success requires the yellow ball SUNK in the pocket (its centre below "
            f"deck level, within the pocket window), the tray settled within "
            f"{c.level_tol_deg:.1f} deg of level, ball and tray at rest, and the white "
            f"spare ball still in its cradle. A ball resting on the deck beside the "
            f"pocket, on top of the roof, parked in the bay, the wrong (white) ball "
            f"anywhere, a tray still tilted or moving — none of these count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Press the blue tab down until the yellow ball rolls around the baffle "
            "into the far lane, then press the red tab down until it drops into the "
            "sunken green pocket, then release and let the tray settle level. Leave "
            "the white spare ball in its cradle."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _tray_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the tray body frame, (N,3) -> (N,3)."""
        return _qapply(_qinv(self.tray.data.root_quat_w),
                       pos_w - self.tray.data.root_pos_w)

    def tilt(self) -> torch.Tensor:
        """(N,) hinge tilt in radians, >0 = blue (+x) end down. Yaw-invariant."""
        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        return torch.asin((-_qapply(self.tray.data.root_quat_w, ex)[:, 2]).clamp(-1.0, 1.0))

    def ball_local(self) -> torch.Tensor:
        """(N,3) ball centre in the tray frame."""
        return self._tray_local(self.ball.data.root_pos_w)

    def in_pocket(self) -> torch.Tensor:
        """(N,) bool: yellow ball SUNK in the green pocket (tray-frame xy window and
        centre below `sink_z` — on-deck and on-roof balls fail by construction)."""
        c = self.cfg
        loc = self.ball_local()
        px, py = c.pocket_center
        return ((loc[:, 0] - px).abs() < c.pocket_x_tol) \
            & ((loc[:, 1] - py).abs() < c.pocket_y_tol) \
            & (loc[:, 2] < c.sink_z) & (loc[:, 2] > -0.02)

    def level(self) -> torch.Tensor:
        """(N,) bool: hinge within `level_tol_deg` of level."""
        return self.tilt().abs() < math.radians(self.cfg.level_tol_deg)

    def spare_home(self) -> torch.Tensor:
        """(N,) bool: white spare ball still in its cradle dish."""
        c = self.cfg
        d = (self.spare.data.root_pos_w[:, :2]
             - self.cradle.data.root_pos_w[:, :2]).norm(dim=-1)
        z_rel = self.spare.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (d < c.cradle_tol) & (z_rel < 0.05)

    def settled(self) -> torch.Tensor:
        """(N,) bool: ball slow and tray rotationally calm (thresholds sit above the
        pod's phantom-velocity readback floor; position windows do the real work)."""
        c = self.cfg
        return (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.tray.data.root_ang_vel_w.norm(dim=-1) < c.tray_calm)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w
                         for b in (self.stand, self.tray, self.ball, self.spare)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        loc = self.ball_local()
        # inside = within the tray CAVITY (xy bounds AND deck..roof z band): a ball on
        # the roof or fallen outside the tray must never latch route credit
        inside = (loc[:, 2] > -0.02) & (loc[:, 2] < c.wall_h) \
            & (loc[:, 0].abs() < c.inner_hx) & (loc[:, 1].abs() < c.inner_hy) \
            & self._finite()
        lane_s = inside & (loc[:, 1] < c.lane_s_y)
        self._bay |= inside & (loc[:, 0] > c.bay_x)
        self._crossed |= lane_s
        self._returned |= lane_s & (loc[:, 0] < c.return_x)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: yellow ball sunk in the pocket, tray level, everything settled,
        spare in its cradle, all finite. All clauses are live physical outcomes."""
        self._update_latches()
        return self.in_pocket() & self.level() & self.settled() \
            & self.spare_home() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*bay + 0.20*crossed + 0.25*returned (all
        latched; ~0 for doing nothing — the ball never leaves the start wall on its
        own), capped at 0.75 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_bay * self._bay.float() + c.w_cross * self._crossed.float()
                + c.w_return * self._returned.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="tilt_maze", robot="null"))
