"""BalanceServeScene — seat the loaded white bowl on the plate END of a pivoting
balance board, and make the board settle LEVEL by counterweighting the marked
weighing half at the correct radius.

Derived from libero_90/kitchen_scene7 "put the white bowl on the plate", but the
plate is no longer a static support surface: it is bolted to the +x end of a
0.44 m plank riding a low-friction Y-axis PIVOT with a gentle centering spring.
Success requires the bowl (plus every copper cube it carries — 0..3 are present,
count what you see) seated on the plate AND the board at rest LEVEL (|pitch| <=
3.5 deg). The seed's own end state — bowl set on the plate, nothing else done —
is expressible here and REJECTED by physics: the un-counterweighted board tips
until its plate end grounds on the deck stop at ~5.7 deg, outside the level band
(smoke-verified). To level the board the solver must reason about TORQUE: read
the load (bowl 150 g + 100 g per present cube), and place the 500 g steel weight
on the tick-marked weighing half at r = 0.18 * m_load / 0.5 — a per-episode
answer spanning 54..162 mm depending on the cube count. The pale 40 g dummy
block is torque-insufficient at every radius (max 0.075 N*m vs the 0.13 N*m
spring window around a >= 0.26 N*m imbalance).

Anti-cheat geometry / clauses (each smoke-verified):
  - The deck under the plank doubles as the tip stop: 22 mm below the level
    underside. No object fits under the plank (min dims: cube 30, dummy 45,
    weight 50, bowl 78 mm) — but a 30 mm cube STANDING ON the deck pokes 8 mm
    into the sweep and could prop the board level; therefore success also
    requires EVERY cube accounted for: present cubes inside the bowl, absent
    cubes untouched in the far spare rack. The dummy/weight/bowl standing on
    the deck prop the board to >= 6.0 / 7.3 / 8.6 deg — all outside the band.
  - The bowl must be seated on the plate (beam-frame gate) — parking it at the
    pivot to zero its torque fails the seat clause.
  - Success is judged LIVE on the physical tilt: removing the counterweight
    after the fact drops success (latched partial credit remains).

Assets are fully procedural (compound-spawner pattern):
  - stand: KINEMATIC charcoal pedestal — deck slab 0.50 x 0.16 x 0.081 under
    the whole plank span (the tip stop) + a visual-only pillar. FIXED pose (the
    spawn-authored joint anchor is world-fixed — randomization lives on the
    movables instead).
  - beam: DYNAMIC plank 0.44 x 0.12 x 0.014 with a white plate disc (r 55 mm)
    + retaining rim at local +0.18 m, and a dark-red tick-marked weighing strip
    (visual-only markers) on the -x half. Explicit MassAPI mass/CoM/inertia;
    spawn-authored RevoluteJoint (axis Y, +/-12 deg) to the stand with a
    spring-return angular drive (k = 2.2 N*m/rad, d = 0.5 N*m*s/rad).
  - bowl: DYNAMIC white octagonal cup (inner r 42 mm, 78 mm tall — deep enough to
    carry the 2+1 pile of three cubes, 150 g).
  - cubes: 3 x 30 mm copper cubes, 100 g each; 0..3 PRESENT per episode
    (subset sampling), absentees parked in a ground rack out of reach.
  - weight: DYNAMIC dark steel block 50 x 50 x 70 mm, 500 g.
  - dummy: DYNAMIC pale block 45 x 45 x 55 mm, 40 g (a decoy).

Per-episode randomization (readback-verifiable): WHICH SIDE the weight and the
dummy start on (they swap stations), xy jitter + yaw on both blocks and on the
bowl, and the present-cube count 0..3.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.25 * place — the weight ever settled on the weighing half (latched)
  0.30 * seat  — the bowl ever seated on the plate with every cube accounted
                 for, bowl+cubes settled, any tilt (latched)
  1.0 iff success() — live: bowl seated, cubes accounted, |tilt| <= 3.5 deg,
                 everything settled and finite.  Non-success caps at 0.55.

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


def _friction_material(stage, path: str, mu_s: float, mu_d: float):
    """A physics material prim (custom-spawner colliders otherwise get ~0.5 friction)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_box(stage, path: str, *, center, size, color, contact_offset=None,
             material=None, orient=None):
    """One box child. `contact_offset=None` -> VISUAL-ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(box.GetPrim(), contact_offset, material)
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, contact_offset,
             material=None):
    """One z-axis cylinder child: translate, displayColor, collider."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(cyl.GetPrim(), contact_offset, material)
    return cyl.GetPrim()


def _rb(root, *, kinematic: bool = False, lin_damp: float = 0.05,
        ang_damp: float = 0.05):
    """RigidBodyAPI + PhysX body armor on a compound root."""
    from pxr import PhysxSchema, UsdPhysics

    api = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        api.CreateKinematicEnabledAttr(True)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)  # TGS vel iters max 4
    return px


def _qz_tuple(yaw: float) -> tuple:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The pedestal: KINEMATIC. Origin at the footprint center on the ground.
    Deck slab (the tip stop, its top `deck_top` below the level plank underside
    by the 22 mm anti-prop gap) + a VISUAL-ONLY pivot pillar (no collider, so it
    can never touch the swinging plank)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, kinematic=True)
    mat = _friction_material(stage, f"{prim_path}/mat", 0.7, 0.6)
    _add_box(stage, f"{prim_path}/deck",
             center=(0.0, 0.0, c.deck_top / 2),
             size=(c.deck_len, c.deck_wid, c.deck_top), color=c.color,
             contact_offset=c.contact_offset, material=mat)
    _add_box(stage, f"{prim_path}/pillar",  # visual only
             center=(0.0, 0.0, (c.deck_top + c.pillar_top) / 2),
             size=(0.05, 0.05, c.pillar_top - c.deck_top),
             color=(0.30, 0.30, 0.33))
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The balance board: DYNAMIC compound, origin AT THE PIVOT (plank mid-plane).
    Plank + white plate disc/rim at local +plate_x + visual-only weighing strip
    and tick marks on the -x half. Explicit MassAPI mass/CoM/diagonal inertia
    (CoM at the pivot: the empty board is balanced by construction). Carries the
    spawn-authored Y-axis RevoluteJoint to the sibling Stand with a spring-return
    angular drive (USD angular drive units are per-DEGREE — converted here)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, ang_damp=0.02)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(c.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in c.inertia]))
    mat = _friction_material(stage, f"{prim_path}/mat", 0.7, 0.6)
    co = c.contact_offset
    ht = c.plank_t / 2  # plank top at local +ht
    _add_box(stage, f"{prim_path}/plank", center=(0.0, 0.0, 0.0),
             size=(c.plank_l, c.plank_w, c.plank_t), color=c.plank_color,
             contact_offset=co, material=mat)
    _add_cyl(stage, f"{prim_path}/plate",
             center=(c.plate_x, 0.0, ht + c.plate_t / 2),
             radius=c.plate_r, height=c.plate_t, color=c.plate_color,
             contact_offset=co, material=mat)
    # retaining rim: 8 boxes around the plate (inner face apothem rim_inner)
    n_side = 8
    rmid = c.rim_inner + c.rim_t / 2
    side_l = 2 * rmid * math.tan(math.pi / n_side) + 0.004
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/rim_{i}",
                 center=(c.plate_x + rmid * math.cos(a), rmid * math.sin(a),
                         ht + c.rim_h / 2),
                 size=(c.rim_t, side_l, c.rim_h), color=c.rim_color,
                 contact_offset=co, material=mat, orient=_qz_tuple(a))
    # visual-only weighing-half strip + tick marks (no colliders: flush marker
    # colliders would wall/park sliding contacts)
    _add_box(stage, f"{prim_path}/strip",
             center=(-0.12, 0.0, ht + 0.0002),
             size=(0.20, c.plank_w - 0.01, 0.0004), color=(0.45, 0.10, 0.10))
    for i in range(5):
        x = -(0.06 + 0.03 * i)
        _add_box(stage, f"{prim_path}/tick_{i}",
                 center=(x, 0.0, ht + 0.0005),
                 size=(0.002, c.plank_w - 0.01, 0.0004), color=(0.95, 0.95, 0.95))
    # the pivot: Y-axis hinge to the sibling Stand, spring-return drive to 0
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    # PhysX filters collisions between jointed bodies by DEFAULT — re-enable so the
    # deck under the plank is a real tip stop (the board grounds at ~5.7 deg).
    j.CreateCollisionEnabledAttr(True)
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.tilt_limit_deg))
    j.CreateUpperLimitAttr(float(c.tilt_limit_deg))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drv.CreateTypeAttr("force")
    drv.CreateTargetPositionAttr(0.0)
    drv.CreateStiffnessAttr(float(c.spring_k) * math.pi / 180.0)  # N*m/deg
    drv.CreateDampingAttr(float(c.spring_d) * math.pi / 180.0)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The bowl: DYNAMIC white octagonal cup. Local origin at the bottom center
    of the floor disc, +z up. Explicit MassAPI mass (CoM at the origin — low,
    stabilizing) + friction material."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rb(root, lin_damp=0.2, ang_damp=0.2)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.mass))
    mat = _friction_material(stage, f"{prim_path}/mat", 0.7, 0.6)
    co = c.contact_offset
    _add_cyl(stage, f"{prim_path}/floor", center=(0.0, 0.0, c.floor_t / 2),
             radius=c.inner_r + c.wall_t, height=c.floor_t, color=c.color,
             contact_offset=co, material=mat)
    n_side = 8
    rmid = c.inner_r + c.wall_t / 2
    side_l = 2 * rmid * math.tan(math.pi / n_side) + 0.004
    zc = c.floor_t + c.wall_h / 2
    for i in range(n_side):
        a = i * 2 * math.pi / n_side
        _add_box(stage, f"{prim_path}/wall_{i}",
                 center=(rmid * math.cos(a), rmid * math.sin(a), zc),
                 size=(c.wall_t, side_l, c.wall_h), color=c.color,
                 contact_offset=co, material=mat, orient=_qz_tuple(a))
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            deck_len: float = 0.50
            deck_wid: float = 0.16
            deck_top: float = 0.081
            pillar_top: float = 0.095
            color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            plank_l: float = 0.44
            plank_w: float = 0.12
            plank_t: float = 0.014
            plate_x: float = 0.18
            plate_r: float = 0.055
            plate_t: float = 0.008
            rim_inner: float = 0.056
            rim_t: float = 0.006
            rim_h: float = 0.012
            mass: float = 0.35
            inertia: tuple = (0.0008, 0.006, 0.0066)
            pivot_z: float = 0.110
            tilt_limit_deg: float = 12.0
            spring_k: float = 2.2   # N*m/rad (converted to per-deg at author time)
            spring_d: float = 0.5   # N*m*s/rad
            plank_color: tuple = (0.55, 0.38, 0.20)
            plate_color: tuple = (0.95, 0.95, 0.92)
            rim_color: tuple = (0.82, 0.82, 0.86)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.042
            wall_t: float = 0.006
            wall_h: float = 0.068
            floor_t: float = 0.010
            mass: float = 0.15
            color: tuple = (0.95, 0.95, 0.92)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, beam=BeamSpawnerCfg,
                              bowl=BowlSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class BalanceServeSceneCfg(BaseCfg):
    """Config for `BalanceServeScene`.

    Balance arithmetic (the task's spine): load torque = (bowl 0.15 + 0.10 per
    present cube) * g * 0.18; weight torque = 0.5 * g * r. Balance radii
    r = 0.36 * m_load = {54, 90, 126, 162} mm for 0..3 cubes. The spring
    (2.2 N*m/rad) makes |tilt| <= 3.5 deg a +/-27 mm placement window around the
    correct radius — one cube of miscount (36 mm) lands OUTSIDE it (4.6 deg).
    With no/insufficient counterweight the board grounds on the deck stop at
    asin(0.022/0.22) = 5.7 deg. The dummy block's best torque (0.075 N*m at
    r = 0.19) leaves >= 0.19 N*m residual -> >= 5.0 deg. Objects standing on the
    deck prop the plank at >= asin((min_dim - 0.022)/0.22): dummy 6.0 deg,
    weight 7.3, bowl 8.6 — only a 30 mm cube could prop it level (2.1 deg),
    which is why success accounts for EVERY cube."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    level_tol_deg: float = tunable(3.5)   # |board pitch| at success; deck stop is 5.7
    seat_xy_tol: float = tunable(0.030)   # bowl origin within this of the plate center
    seat_z_lo: float = tunable(0.008)     # (beam frame; resting origin z = 15 mm)
    seat_z_hi: float = tunable(0.035)
    upright_max_deg: float = tunable(10.0)   # bowl +z within this of world up when seated
    cube_xy_tol: float = tunable(0.040)   # cube center inside the bowl (bowl frame)
    cube_z_lo: float = tunable(0.003)
    cube_z_hi: float = tunable(0.100)
    depot_tol: float = tunable(0.15)      # absent cube still within this of the rack
    half_x_lo: float = tunable(-0.215)    # weight-on-weighing-half gate (beam frame)
    half_x_hi: float = tunable(-0.030)
    half_y_tol: float = tunable(0.055)
    half_z_lo: float = tunable(0.015)     # (resting weight center z = 42 mm)
    half_z_hi: float = tunable(0.100)
    settle_speed: float = tunable(0.06)   # max |lin vel| of bowl/cubes/weight (m/s)
    beam_still_w: float = tunable(0.25)   # max beam |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    side_swap: bool = tunable(True)       # weight/dummy swap start stations
    block_jitter: float = tunable(0.03)   # weight/dummy xy jitter (+/- m)
    block_yaw_deg: float = tunable(180.0)
    bowl_jitter: float = tunable(0.04)    # bowl xy jitter (+/- m)
    bowl_yaw_deg: float = tunable(180.0)
    subset_sample: bool = tunable(True)   # per-episode cube count 0..3

    # --- info: layout (env-local, ground z = 0; the STAND IS FIXED: the spawn-
    # authored joint anchor is world-fixed, so randomization lives on movables) --
    stand_pos: tuple = info((0.42, 0.0))
    station_x: float = info(0.28)         # weight/dummy stations: (station_x, +/-station_y)
    station_y: float = info(0.22)
    bowl_pos: tuple = info((0.10, 0.0))   # clear of the deck face at x = 0.17
    depot_pos: tuple = info((1.15, 1.15))  # spare-cube rack (out of arm reach)
    # --- info: stand / board geometry -----------------------------------------------------------
    pivot_z: float = info(0.110)
    deck_top: float = info(0.081)         # plank underside 0.103 -> 22 mm anti-prop gap
    deck_len: float = info(0.50)
    deck_wid: float = info(0.16)
    plank_l: float = info(0.44)
    plank_w: float = info(0.12)
    plank_t: float = info(0.014)
    plate_x: float = info(0.18)           # the LOAD ARM (plate center, beam frame)
    plate_r: float = info(0.055)
    plate_t: float = info(0.008)          # bowl resting origin: beam z = 0.007+0.008
    rim_inner: float = info(0.056)
    rim_h: float = info(0.012)
    beam_mass: float = info(0.35)
    spring_k: float = info(2.2)           # N*m/rad centering spring
    spring_d: float = info(0.5)
    tilt_limit_deg: float = info(12.0)
    deck_stop_deg: float = info(5.7)      # asin(0.022 / 0.22)
    # --- info: movables -------------------------------------------------------------------------
    bowl_inner_r: float = info(0.042)
    bowl_wall_t: float = info(0.006)
    bowl_wall_h: float = info(0.068)   # deep enough to contain the 2+1 cube pile
    bowl_floor_t: float = info(0.010)
    bowl_h: float = info(0.078)
    bowl_mass: float = info(0.15)
    cube_size: float = info(0.030)
    cube_mass: float = info(0.100)
    n_cubes: int = info(3)
    cube_color: tuple = info((0.72, 0.35, 0.15))
    weight_size: tuple = info((0.05, 0.05, 0.07))
    weight_mass: float = info(0.500)
    weight_color: tuple = info((0.25, 0.26, 0.30))
    dummy_size: tuple = info((0.045, 0.045, 0.055))
    dummy_mass: float = info(0.040)
    dummy_color: tuple = info((0.78, 0.79, 0.82))
    contact_offset: float = info(0.002)
    # --- info: rubric weights (0.25 + 0.30 = 0.55 = the non-success cap) ------------------------
    w_place: float = info(0.25)
    w_seat: float = info(0.30)


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


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("balance_serve")
class BalanceServeScene(BaseScene):
    cfg: BalanceServeSceneCfg

    CUBE_NAMES = ("cube_0", "cube_1", "cube_2")

    def __init__(self, cfg: BalanceServeSceneCfg | None = None) -> None:
        super().__init__(cfg or BalanceServeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        stand_spawn = cls["stand"](
            deck_len=c.deck_len, deck_wid=c.deck_wid, deck_top=c.deck_top,
            contact_offset=c.contact_offset,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        beam_spawn = cls["beam"](
            plank_l=c.plank_l, plank_w=c.plank_w, plank_t=c.plank_t,
            plate_x=c.plate_x, plate_r=c.plate_r, plate_t=c.plate_t,
            rim_inner=c.rim_inner, rim_h=c.rim_h, mass=c.beam_mass,
            pivot_z=c.pivot_z, tilt_limit_deg=c.tilt_limit_deg,
            spring_k=c.spring_k, spring_d=c.spring_d,
            contact_offset=c.contact_offset)
        bowl_spawn = cls["bowl"](
            inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, wall_h=c.bowl_wall_h,
            floor_t=c.bowl_floor_t, mass=c.bowl_mass,
            contact_offset=c.contact_offset)

        def block(size: tuple, mass: float, color: tuple) -> Any:
            return sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.05,
                    sleep_threshold=0.0, stabilization_threshold=0.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # NOTE: dict order matters — Stand must exist when Beam's joint is authored.
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], c.pivot_z)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bowl_pos[0], c.bowl_pos[1], 0.001)),
            ),
            "weight": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Weight",
                spawn=block(c.weight_size, c.weight_mass, c.weight_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.station_x, c.station_y, c.weight_size[2] / 2 + 0.001)),
            ),
            "dummy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dummy",
                spawn=block(c.dummy_size, c.dummy_mass, c.dummy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.station_x, -c.station_y, c.dummy_size[2] / 2 + 0.001)),
            ),
        }
        for i, name in enumerate(self.CUBE_NAMES):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + str(i),
                spawn=block((c.cube_size,) * 3, c.cube_mass, c.cube_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.06 * i, c.depot_pos[1],
                         c.cube_size / 2 + 0.001)),
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
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.weight: RigidObject = env.iscene["weight"]
        self.dummy: RigidObject = env.iscene["dummy"]
        self.cubes: dict[str, RigidObject] = {n: env.iscene[n] for n in self.CUBE_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # present[e, i]: cube i participates in episode e (sampled at reset)
        self.present = torch.zeros(n, len(self.CUBE_NAMES), dtype=torch.bool, device=dev)
        self.weight_side = torch.ones(n, dtype=torch.long, device=dev)  # +1 / -1 (y sign)
        # latches (partial credit survives transients; success is judged live)
        self._place = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: burn RNG draws (first post-seed draw is degenerate),
        pick the weight/dummy side split, place both blocks (jitter + yaw), place
        the bowl (jitter + yaw), sample the present cube subset and seat present
        cubes INSIDE the bowl in stable seats (two abreast + one bridging — a 2+1
        pile); park absentees in the rack; re-home the board; clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(8, device=dev)  # burn: first post-seed draws are degenerate

        def jit(k: float) -> torch.Tensor:
            return (torch.rand(m, 2, device=dev) * 2 - 1) * k

        def yaw(deg: float) -> torch.Tensor:
            return _qz((torch.rand(m, device=dev) * 2 - 1) * math.radians(deg))

        # --- stand (kinematic, FIXED) + board re-homed level at the pivot ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_pos[0]
        st[:, 1] = c.stand_pos[1]
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_pos[0]
        st[:, 1] = c.stand_pos[1]
        st[:, 2] = c.pivot_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- which side does the WEIGHT start on ---
        if c.side_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, dtype=torch.long, device=dev),
                               -torch.ones(m, dtype=torch.long, device=dev))
        else:
            side = torch.ones(m, dtype=torch.long, device=dev)
        self.weight_side[env_ids] = side

        # --- weight / dummy: stations + jitter + yaw ---
        for body, sgn, h in ((self.weight, side.float(), c.weight_size[2]),
                             (self.dummy, -side.float(), c.dummy_size[2])):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.station_x
            st[:, 1] = sgn * c.station_y
            st[:, :2] += jit(c.block_jitter)
            st[:, 2] = h / 2 + 0.001
            st[:, 3:7] = yaw(c.block_yaw_deg)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- bowl (dynamic): start pose + jitter + yaw ---
        q_bowl = yaw(c.bowl_yaw_deg)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.bowl_pos[0]
        bp[:, 1] = c.bowl_pos[1]
        bp[:, :2] += jit(c.bowl_jitter)
        bp[:, 2] = 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_bowl
        self.bowl.write_root_state_to_sim(st, env_ids)

        # --- cubes: sample subset, seat present cubes inside the bowl ---
        nc = len(self.CUBE_NAMES)
        if c.subset_sample:
            k = torch.randint(0, nc + 1, (m,), device=dev)
        else:
            k = torch.full((m,), nc, dtype=torch.long, device=dev)
        rank = torch.rand(m, nc, device=dev).argsort(dim=1).argsort(dim=1)
        pres = rank < k.unsqueeze(1)
        self.present[env_ids] = pres
        # Present cubes fill STABLE SEATS by presence rank (bowl frame, cube faces
        # aligned to the bowl): two abreast on the floor, the third bridging their
        # seam — a deterministic 2+1 pile (top face ~70 mm < the 78 mm rim). A
        # staggered near-coaxial tower topples into rim-height leaners that spill
        # during any carry.
        zc = c.bowl_floor_t + c.cube_size / 2
        seats = torch.tensor([[-0.017, 0.0, zc + 0.002],
                              [+0.017, 0.0, zc + 0.002],
                              [0.0, 0.0, zc + c.cube_size + 0.004]], device=dev)
        q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).expand(m, 4)
        for i, name in enumerate(self.CUBE_NAMES):
            loc = seats[rank[:, i].clamp(max=nc - 1)]
            in_bowl_pos = bp + quat_apply(q_bowl, loc)
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.depot_pos[0] + 0.06 * i
            park[:, 1] = c.depot_pos[1]
            park[:, 2] = c.cube_size / 2 + 0.001
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres[:, i].unsqueeze(1),
                                              in_bowl_pos, park)
            st[:, 3:7] = torch.where(pres[:, i].unsqueeze(1), q_bowl, q_id)
            self.cubes[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._place[env_ids] = False
        self._seat[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "weight": self.weight.data.root_state_w[env_ids].clone(),
            "dummy": self.dummy.data.root_state_w[env_ids].clone(),
            "cubes": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.cubes.items()},
            "present": self.present[env_ids].clone(),
            "weight_side": self.weight_side[env_ids].clone(),
            "place": self._place[env_ids].clone(),
            "seat": self._seat[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.weight.write_root_state_to_sim(state["weight"], env_ids)
        self.dummy.write_root_state_to_sim(state["dummy"], env_ids)
        for n, b in self.cubes.items():
            b.write_root_state_to_sim(state["cubes"][n], env_ids)
        self.present[env_ids] = state["present"]
        self.weight_side[env_ids] = state["weight_side"]
        self._place[env_ids] = state["place"]
        self._seat[env_ids] = state["seat"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BALANCE BOARD stands on a charcoal pedestal: a "
            f"{c.plank_l * 1000:.0f} mm wooden plank riding a free center PIVOT with a "
            f"gentle centering spring ({c.spring_k:.1f} N*m/rad). Bolted to one end "
            f"(+{c.plate_x * 1000:.0f} mm from the pivot) is a round WHITE PLATE with a "
            f"low retaining rim. The other half is the WEIGHING HALF: a dark-red strip "
            f"with white tick marks every 30 mm, from 60 mm to 180 mm out from the "
            f"pivot. The board can tip about +/-{c.deck_stop_deg:.0f} deg before its "
            f"underside grounds on the pedestal deck.\n"
            f"On the floor nearby: a WHITE octagonal BOWL "
            f"({c.bowl_mass * 1000:.0f} g, ~{2 * 0.052 * 1000:.0f} mm across) that may "
            f"contain up to three COPPER CUBES ({c.cube_size * 1000:.0f} mm, "
            f"{c.cube_mass * 1000:.0f} g EACH) — the count varies every episode; look "
            f"into the bowl and count them. Off to the sides stand two blocks: the "
            f"dark STEEL WEIGHT ({c.weight_mass * 1000:.0f} g, "
            f"{c.weight_size[0] * 1000:.0f}x{c.weight_size[1] * 1000:.0f}x"
            f"{c.weight_size[2] * 1000:.0f} mm) and a PALE DUMMY block "
            f"({c.dummy_mass * 1000:.0f} g, slightly smaller) — which side each starts "
            f"on is randomized; tell them apart by color and size. Far away is a rack "
            f"of spare cubes: LEAVE IT ALONE — every cube must end where it belongs "
            f"(in the bowl if it started there, in the rack otherwise).\n"
            f"Goal: serve the bowl LEVEL. Seat the loaded bowl on the white plate "
            f"(cubes still inside), and counterweight the board so it settles LEVEL "
            f"(within {c.level_tol_deg:.1f} deg). Balance the torques: the load pulls "
            f"at {c.plate_x * 1000:.0f} mm, so the steel weight belongs on the weighing "
            f"half at radius = {c.plate_x:.2f} * load_mass / {c.weight_mass:.1f} "
            f"(use the tick marks). A board simply loaded with the bowl and left alone "
            f"tips onto the deck stop — that is failure. Success: bowl seated on the "
            f"plate, every cube accounted for, board at rest level, nothing moving."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the white bowl with its copper cubes on the plate end of the "
            "balance board, then put the steel weight on the marked weighing half at "
            "the right distance so the board settles level. Count the cubes to pick "
            "the distance; the pale dummy block is too light to work."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> `body`'s frame. Accepts (N,3) or (N,P,3); same shape out."""
        from isaaclab.utils.math import quat_apply_inverse

        bp = body.data.root_pos_w
        bq = body.data.root_quat_w
        if pos_w.dim() == 3:
            n, p = pos_w.shape[0], pos_w.shape[1]
            rel = (pos_w - bp[:, None, :]).reshape(n * p, 3)
            q = bq[:, None, :].expand(n, p, 4).reshape(n * p, 4)
            return quat_apply_inverse(q, rel).reshape(n, p, 3)
        return quat_apply_inverse(bq, pos_w - bp)

    def tilt(self) -> torch.Tensor:
        """(N,) rad: board pitch — asin of the world-z of the beam's +x axis.
        POSITIVE = plate end UP (weight side heavy)."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        return torch.asin(quat_apply(self.beam.data.root_quat_w, ex)[:, 2]
                          .clamp(-1.0, 1.0))

    def bowl_up(self) -> torch.Tensor:
        """(N, 3): the bowl's local +z axis in world coordinates."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0],
                          device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.bowl.data.root_quat_w, ez)

    def _cube_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,3,3), |lin_vel| (N,3)) for all cubes, name order."""
        pos = torch.stack([b.data.root_pos_w for b in self.cubes.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.cubes.values()], dim=1)
        return pos, vel

    def bowl_seated(self) -> torch.Tensor:
        """(N,) bool: bowl origin on the plate (beam frame: xy within `seat_xy_tol`
        of (plate_x, 0), z in the seat band) and the bowl near-upright."""
        c = self.cfg
        loc = self._local(self.beam, self.bowl.data.root_pos_w)
        tgt = torch.tensor([c.plate_x, 0.0], device=loc.device)
        on_plate = ((loc[:, :2] - tgt).norm(dim=-1) < c.seat_xy_tol) \
            & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)
        upright = self.bowl_up()[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(c.upright_max_deg))
        return on_plate & upright

    def cubes_in_bowl(self) -> torch.Tensor:
        """(N, 3) bool: cube center inside the bowl's interior volume (bowl frame)."""
        c = self.cfg
        pos, _v = self._cube_tensors()
        loc = self._local(self.bowl, pos)
        return (loc[:, :, :2].norm(dim=-1) < c.cube_xy_tol) \
            & (loc[:, :, 2] > c.cube_z_lo) & (loc[:, :, 2] < c.cube_z_hi)

    def cubes_in_depot(self) -> torch.Tensor:
        """(N, 3) bool: cube still at the spare rack on the ground."""
        c = self.cfg
        pos, _v = self._cube_tensors()
        rel = pos - self.env_origins[:, None, :]
        dep = torch.tensor([c.depot_pos[0] + 0.06, c.depot_pos[1]], device=pos.device)
        return ((rel[:, :, :2] - dep).norm(dim=-1) < c.depot_tol) \
            & (rel[:, :, 2] < 0.08)

    def cubes_ok(self) -> torch.Tensor:
        """(N,) bool: EVERY cube accounted for — present cubes inside the bowl,
        absent cubes untouched in the rack. This clause (not fiat) is what makes a
        cube-on-the-deck prop of the board a failure."""
        inb = self.cubes_in_bowl()
        ind = self.cubes_in_depot()
        return torch.where(self.present, inb, ind).all(dim=1)

    def weight_on_half(self) -> torch.Tensor:
        """(N,) bool: the steel weight resting on the WEIGHING HALF (beam frame)."""
        c = self.cfg
        loc = self._local(self.beam, self.weight.data.root_pos_w)
        return (loc[:, 0] > c.half_x_lo) & (loc[:, 0] < c.half_x_hi) \
            & (loc[:, 1].abs() < c.half_y_tol) \
            & (loc[:, 2] > c.half_z_lo) & (loc[:, 2] < c.half_z_hi)

    def level(self) -> torch.Tensor:
        """(N,) bool: |board pitch| within the level band."""
        return self.tilt().abs() <= math.radians(self.cfg.level_tol_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: bowl, present cubes, weight below `settle_speed`; beam
        angular velocity below `beam_still_w`."""
        c = self.cfg
        _p, cvel = self._cube_tensors()
        cubes_still = ((cvel < c.settle_speed) | ~self.present).all(dim=1)
        return (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.weight.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & cubes_still \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_still_w)

    def _update_latches(self) -> None:
        c = self.cfg
        wstill = self.weight.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._place |= self.weight_on_half() & wstill
        _p, cvel = self._cube_tensors()
        cubes_still = ((cvel < c.settle_speed) | ~self.present).all(dim=1)
        bstill = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        self._seat |= self.bowl_seated() & self.cubes_ok() & cubes_still & bstill

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, live: bowl seated on the plate, every cube accounted for,
        the board at rest LEVEL, everything settled and finite. All clauses are
        physical outcomes — remove the counterweight and success drops."""
        self._update_latches()
        pos, _v = self._cube_tensors()
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.bowl.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.beam.data.root_quat_w).all(dim=-1)
        return self.bowl_seated() & self.cubes_ok() & self.level() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*place + 0.30*seat (latched; 0 for the null
        policy), capped at 0.55 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_place * self._place.float()
                + c.w_seat * self._seat.float()).clamp(max=0.55)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="balance_serve", robot="null"))
