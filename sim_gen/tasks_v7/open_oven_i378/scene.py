"""SpitRoastScene — thread a spit rod through a meat block, then hang the loaded spit in a
two-post roasting rack (sim_gen task `open_oven_i378`).

Derived from the RLBench `open_oven` seed but STRATEGICALLY DIFFERENT (see TASK.md): the
seed's plan is "grasp the door handle, pull the hinged oven door open through its arc" —
one grasp, one guided pull along a built articulation, a binary open/closed goal. Here
there is NO articulation and nothing opens: the solver must MANUFACTURE a compound object
and establish a gravity-borne suspension. A steel SPIT ROD (12 mm dia, 410 mm, ball knob
on one end) lies on the floor; a MEAT BLOCK (90 mm cube with a 26 mm square through-bore)
rests on a loading cradle beside the rack. The plan skeleton is:

  (1) THREAD: drive the rod's bare tip through the block's bore — a contact-dynamics
      insertion (26 mm bore vs 12 mm rod), pushing until the block's face presses the
      cradle's backstop and the tip protrudes; the 32 mm knob cannot pass the bore, so the
      threading direction is forced and the knob becomes the carry handle;
  (2) CARRY: lift the loaded spit — the block hangs freely on the rod (its bore top rides
      the rod), a two-body load the seed never has;
  (3) SEAT: lower BOTH rod ends into the rack's two slotted brackets (15 mm slots with 45
      degree funnel plates) so the meat hangs suspended between the posts, hands-off.

Execution order is geometrically forced: the block cannot be threaded onto an already
racked rod, because the posts + slot cheeks + funnel plates block the 90 mm block from
travelling along a seated rod into the between-posts region (and success requires the
block to hang BETWEEN the posts). Threading must happen off the rack, at the cradle.

Assets are fully procedural (compound spawners; children of one body never collide):
  - rack: STATIC colliders — 2 posts (top 0.194 m), base beam, per post 2 slot cheeks
    (gap 15 mm, top 0.210 m) and 2 funnel plates at 45 deg (mouth half-width 64 mm,
    top edge 0.267 m). Rod seat line: y=0, z=0.200.
  - station: KINEMATIC cradle (0.10 cube) + backstop plate (top 0.128 m, below the
    resting block's bore bottom 0.132 m so the rod clears the stop while the block
    presses it). Teleported per episode: left/right of the rack, yawed, jittered.
  - roast: DYNAMIC 90 mm block built from 4 slabs leaving a 26 mm square x-bore.
  - spit: DYNAMIC rod (cylinder r 6 mm, len 410 mm, axis x) + knob sphere r 16 mm at
    +x end. Knob dia 32 mm > bore 26 mm: it cannot pass.

External wrench buffers (`spit_force/spit_torque/roast_force/roast_torque`) are consumed
by `post_step` (the scene OWNS the wrench slots); writers are solve.py's teleport-contract
servo (transport by teleport, every load-bearing interaction through contact) and smoke's
probes.

Rubric (0..1; latched stage credit anchored in the demonstrated solve trajectory):
  0.15  entered  — rod tip ever inside the bore (>= 2 cm of rod within the bore, on axis)
  0.40  threaded — rod ever fully through: protrudes >= 2 cm past BOTH bore faces, on axis
  0.65  carried  — ... while the block hangs high (centre above 0.170 m, off any support)
  1.0 iff success(): threaded AND both rod ends resting in the two seat slots (|y| <= 6 mm,
      |z - 0.200| <= 8 mm at each notch plane) AND the block hanging between the posts
      (centre above 0.170 m, |x - rack_x| <= 45 mm) AND everything settled and finite.
  Non-success is capped at 0.65. Laying the bare rod in the rack, or perching the block
  on top of things, scores ~0 (smoke-verified).

Per-episode randomization (readback-verifiable): station side (left/right of the rack),
station yaw (flip 180 deg + uniform), station xy jitter, block yaw on the cradle, spit
floor pose (rejection-sampled clear of rack + station, free yaw).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- USD authoring helpers ---------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author the canonical [translate, orient, scale] op triple
    (XformPrimView requires it)."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    t = translation if translation is not None else (0.0, 0.0, 0.0)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in t]))
    w, x, y, z = (float(v) for v in (orientation if orientation is not None
                                     else (1.0, 0.0, 0.0, 0.0)))
    xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _phys_material(stage, path: str, mu_s: float, mu_d: float, restitution: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(float(restitution))
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_box(stage, path: str, *, center, size, color, quat=(1.0, 0.0, 0.0, 0.0),
             collide: Callable | None = None, mat=None):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    w, x, y, z = (float(v) for v in quat)
    xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
        if mat is not None:
            _bind_material(box.GetPrim(), mat)
    return box.GetPrim()


def _dynamic_body(root, *, lin_damp: float, ang_damp: float):
    """Author the dynamic-rigid-body APIs a custom spawner must supply itself (custom
    spawn funcs apply NO cfg schemas)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    return px


def _author_mass(root, *, mass: float, com, diag):
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in diag]))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


# ----- rack spawner (static: posts + beam + slot cheeks + funnel plates) -------------------------
def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Roasting rack: STATIC colliders only (no rigid body). Local frame: origin on the
    ground midway between the posts; posts along x at +-post_dx; seat line y=0,
    z = post_h + rod_r."""
    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/rackMat", cfg.mu, cfg.mu, 0.0)
    c = cfg
    steel = (0.38, 0.40, 0.44)
    light = (0.58, 0.60, 0.66)
    s45, c45 = math.sin(math.pi / 4), math.cos(math.pi / 4)
    hq = math.pi / 8  # half of 45 deg

    _add_box(stage, f"{prim_path}/beam",
             center=(0.0, 0.0, c.beam_h / 2.0),
             size=(2.0 * c.post_dx + c.post_sx, c.post_sy, c.beam_h),
             color=steel, collide=collide, mat=mat)
    for k, xk in enumerate((-c.post_dx, c.post_dx)):
        _add_box(stage, f"{prim_path}/post_{k}",
                 center=(xk, 0.0, c.post_h / 2.0),
                 size=(c.post_sx, c.post_sy, c.post_h),
                 color=steel, collide=collide, mat=mat)
        for j, s in enumerate((-1.0, 1.0)):
            _add_box(stage, f"{prim_path}/cheek_{k}_{j}",
                     center=(xk, s * (c.slot_half + c.cheek_w / 2.0),
                             c.post_h + c.cheek_h / 2.0),
                     size=(c.post_sx, c.cheek_w, c.cheek_h),
                     color=light, collide=collide, mat=mat)
            # 45-deg funnel plate: guide (inner-top) face's lower edge EXACTLY on the
            # cheek's inner top corner (y = s*slot_half, z = cheek_top) — no flat ledge
            # can catch the rod; the face funnels it into the slot.
            cy = s * (c.slot_half + c45 * c.fun_len / 2.0 + s45 * c.fun_t / 2.0)
            cz = c.post_h + c.cheek_h + s45 * c.fun_len / 2.0 - c45 * c.fun_t / 2.0
            _add_box(stage, f"{prim_path}/funnel_{k}_{j}",
                     center=(xk, cy, cz),
                     size=(c.post_sx, c.fun_len, c.fun_t),
                     color=light,
                     quat=(math.cos(hq), s * math.sin(hq), 0.0, 0.0),
                     collide=collide, mat=mat)
    return root


# ----- station spawner (kinematic: cradle + backstop) --------------------------------------------
def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Loading station: KINEMATIC compound (teleported per episode). Local frame: origin
    on the ground under the cradle centre; backstop plate on the -x side. The block is
    threaded along -x, pressing its face on the backstop."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    _author_mass(root, mass=5.0, com=(0.0, 0.0, 0.05), diag=(0.02, 0.02, 0.02))
    collide = _make_collide(cfg.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/woodMat", cfg.mu, cfg.mu, 0.0)
    c = cfg
    _add_box(stage, f"{prim_path}/cradle",
             center=(0.0, 0.0, c.cradle_s / 2.0),
             size=(c.cradle_s, c.cradle_s, c.cradle_s),
             color=(0.55, 0.38, 0.20), collide=collide, mat=mat)
    _add_box(stage, f"{prim_path}/backstop",
             center=(c.stop_x, 0.0, c.cradle_s + c.stop_h / 2.0),
             size=(c.stop_t, c.cradle_s, c.stop_h),
             color=(0.40, 0.26, 0.13), collide=collide, mat=mat)
    return root


# ----- roast spawner (dynamic: 4 slabs leaving a square x-bore) ----------------------------------
def _spawn_roast(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Meat block: DYNAMIC compound. Body frame at the cube centre; the 2*bore_half
    square bore runs along body x. Four slabs (top/bottom/left/right) tile the cube
    minus the bore; children of one body never collide with each other."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, lin_damp=0.05, ang_damp=0.05)
    c = cfg
    s, b = c.roast_s, c.bore_half
    wall = (s / 2.0 - b)  # slab thickness covering bore..outer face
    # solid-cube inertia approximation (bore removes ~8% of the mass moment)
    ii = c.mass * s * s / 6.0
    _author_mass(root, mass=c.mass, com=(0.0, 0.0, 0.0), diag=(ii, ii, ii))
    collide = _make_collide(c.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/meatMat", c.mu, c.mu - 0.05, 0.0)
    meat, dark = (0.55, 0.20, 0.12), (0.44, 0.15, 0.09)
    zc = b + wall / 2.0
    _add_box(stage, f"{prim_path}/top", center=(0.0, 0.0, zc),
             size=(s, s, wall), color=meat, collide=collide, mat=mat)
    _add_box(stage, f"{prim_path}/bottom", center=(0.0, 0.0, -zc),
             size=(s, s, wall), color=meat, collide=collide, mat=mat)
    _add_box(stage, f"{prim_path}/left", center=(0.0, -zc, 0.0),
             size=(s, wall, 2.0 * b), color=dark, collide=collide, mat=mat)
    _add_box(stage, f"{prim_path}/right", center=(0.0, zc, 0.0),
             size=(s, wall, 2.0 * b), color=dark, collide=collide, mat=mat)
    return root


# ----- spit spawner (dynamic: rod cylinder + knob sphere) ----------------------------------------
def _spawn_spit(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Spit rod: DYNAMIC compound. Body frame at the rod centre, axis along body x; bare
    TIP at x=-rod_half, ball KNOB (cannot pass the bore) centred at x=+rod_half.
    Transverse inertia is physical (slender rod); the axial moment is authored well
    above the physical ~7e-6 so the axis-spin mode stays integrable under the servo's
    transverse damping."""
    from pxr import Gf, UsdGeom

    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, lin_damp=0.1, ang_damp=0.2)
    c = cfg
    m, h = c.mass, c.rod_half
    iyy = m * (2.0 * h) ** 2 / 12.0 + 0.1 * m * h ** 2  # rod + knob point mass at +h
    _author_mass(root, mass=m, com=(c.com_x, 0.0, 0.0), diag=(1.0e-3, iyy, iyy))
    collide = _make_collide(c.contact_offset)
    mat = _phys_material(stage, f"{prim_path}/steelMat", c.mu, c.mu, 0.0)

    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/rod")
    cyl.CreateRadiusAttr(float(c.rod_r))
    cyl.CreateHeightAttr(float(2.0 * h))
    cyl.CreateAxisAttr("X")
    cyl.CreateExtentAttr([Gf.Vec3f(-h, -c.rod_r, -c.rod_r), Gf.Vec3f(h, c.rod_r, c.rod_r)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.75, 0.78)])
    collide(cyl.GetPrim())
    _bind_material(cyl.GetPrim(), mat)

    knob = UsdGeom.Sphere.Define(stage, f"{prim_path}/knob")
    knob.CreateRadiusAttr(float(c.knob_r))
    UsdGeom.Xformable(knob.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(h), 0.0, 0.0))
    knob.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.18)])
    collide(knob.GetPrim())
    _bind_material(knob.GetPrim(), mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(SpawnerCfg):
            func: Callable = clone(_spawn_rack)
            post_dx: float = 0.13
            post_sx: float = 0.04
            post_sy: float = 0.06
            post_h: float = 0.194
            beam_h: float = 0.03
            slot_half: float = 0.0075
            cheek_w: float = 0.012
            cheek_h: float = 0.016
            fun_len: float = 0.08
            fun_t: float = 0.008
            mu: float = 0.30
            contact_offset: float = 0.002

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            cradle_s: float = 0.10
            stop_x: float = -0.055
            stop_t: float = 0.012
            stop_h: float = 0.028
            mu: float = 0.60
            contact_offset: float = 0.002

        @configclass
        class RoastSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_roast)
            roast_s: float = 0.09
            bore_half: float = 0.013
            mass: float = 0.70
            mu: float = 0.50
            contact_offset: float = 0.002

        @configclass
        class SpitSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_spit)
            rod_r: float = 0.006
            rod_half: float = 0.205
            knob_r: float = 0.016
            mass: float = 0.40
            com_x: float = 0.015
            mu: float = 0.15
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, station=StationSpawnerCfg,
                              roast=RoastSpawnerCfg, spit=SpitSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class SpitRoastSceneCfg(BaseCfg):
    """Config for `SpitRoastScene`. Derived geometry lives in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers; the geometric claims the task
    rests on are asserted there (honesty by construction)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    bore_lat_tol: float = tunable(0.011)   # rod axis within this of the bore axis (m) at the faces
    entry_min: float = tunable(0.020)      # >= this much rod inside the bore = "entered" (m)
    protrude_min: float = tunable(0.020)   # rod past BOTH bore faces by this = "threaded" (m)
    seat_x_margin: float = tunable(0.020)  # rod x-extent must cover each notch plane by this (m)
    seat_y_tol: float = tunable(0.006)     # |rod y| at the notch plane (m); slot itself holds 1.5mm
    seat_z_tol: float = tunable(0.008)     # |rod z - z_seat| at the notch plane (m)
    hang_z_min: float = tunable(0.170)     # block centre above this = hanging (rest: 0.193)
    roast_x_tol: float = tunable(0.045)    # block centre within this of rack_x = between the posts
    # (strictly INSIDE the physical jam limit post_dx - post_sx/2 - roast_s/2 = 0.065,
    # so a block shoved along the rod against a post EXCEEDS the band: success stays a
    # live, revocable state — smoke check 12 depends on this margin)
    settle_lin: float = tunable(0.05)      # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(0.50)      # max |ang vel| when judging success (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    station_jit: float = tunable(0.03)     # station xy jitter (+- m)
    station_yaw_deg: float = tunable(35.0)  # station yaw jitter about 0/180 (+- deg)
    roast_yaw_deg: float = tunable(4.0)    # block yaw jitter relative to the station (+- deg)
    spit_x_lo: float = tunable(0.08)       # spit spawn: centre x range (m)
    spit_x_hi: float = tunable(0.28)
    spit_y_lo: float = tunable(0.18)       # spit spawn: |centre y| range, opposite the station
    spit_y_hi: float = tunable(0.34)

    # --- info: rack ------------------------------------------------------------------------------
    rack_x: float = info(0.45)             # rack centre (y = 0, on the ground)
    post_dx: float = info(0.13)            # notch planes at rack_x +- post_dx
    post_sx: float = info(0.04)
    post_sy: float = info(0.06)
    post_h: float = info(0.194)            # post top = rod seating surface
    beam_h: float = info(0.03)
    slot_half: float = info(0.0075)        # slot inner half-gap (15 mm slot)
    cheek_w: float = info(0.012)
    cheek_h: float = info(0.016)           # cheek top = post_h + cheek_h = 0.210
    fun_len: float = info(0.08)            # 45-deg funnel plate slope length
    fun_t: float = info(0.008)
    rack_mu: float = info(0.30)
    # --- info: loading station -------------------------------------------------------------------
    cradle_s: float = info(0.10)           # cradle cube; top face = loading height
    stop_x: float = info(-0.055)           # backstop plate centre x (station frame)
    stop_t: float = info(0.012)
    stop_h: float = info(0.028)            # backstop top = cradle_s + stop_h = 0.128
    station_y: float = info(0.30)          # station centre at y = +-station_y (+ jitter)
    station_mu: float = info(0.60)
    # --- info: roast block -----------------------------------------------------------------------
    roast_s: float = info(0.09)
    bore_half: float = info(0.013)         # 26 mm square through-bore along body x
    roast_mass: float = info(0.70)
    roast_mu: float = info(0.50)
    # --- info: spit rod --------------------------------------------------------------------------
    rod_r: float = info(0.006)
    rod_half: float = info(0.205)          # tip at body -x, knob centre at body +x
    knob_r: float = info(0.016)            # 32 mm knob > 26 mm bore: cannot pass
    rod_mass: float = info(0.40)
    rod_com_x: float = info(0.015)
    rod_mu: float = info(0.15)
    contact_offset: float = info(0.002)
    # --- info: rubric stage weights (ordered; non-success cap = w_carried) -----------------------
    w_entered: float = info(0.15)
    w_threaded: float = info(0.40)
    w_carried: float = info(0.65)

    # Derived (filled in __post_init__).
    cheek_top: float = field(default=None, init=False)
    z_seat: float = field(default=None, init=False)     # seated rod centre height
    z_hang: float = field(default=None, init=False)     # hanging block centre height
    mouth_half: float = field(default=None, init=False)  # funnel mouth inner half-width
    funnel_top: float = field(default=None, init=False)  # funnel plate top edge height
    roast_rest_z: float = field(default=None, init=False)  # block centre resting on the cradle

    def __post_init__(self) -> None:
        c45 = math.cos(math.pi / 4)
        self.cheek_top = self.post_h + self.cheek_h
        self.z_seat = self.post_h + self.rod_r
        self.z_hang = self.z_seat - (self.bore_half - self.rod_r)
        self.mouth_half = self.slot_half + c45 * self.fun_len
        self.funnel_top = self.cheek_top + c45 * self.fun_len
        self.roast_rest_z = self.cradle_s + self.roast_s / 2.0

        # ----- honesty-by-construction asserts (the claims the task rests on) -----
        assert 2.0 * self.knob_r > 2.0 * self.bore_half + 0.004, \
            "the knob must NOT pass the bore (forces the threading direction)"
        assert self.bore_half - self.rod_r >= 0.005, \
            "the bore must admit the rod with real play (contact insertion, not a press fit)"
        assert self.bore_lat_tol >= (self.bore_half - self.rod_r) + 0.003, \
            "exceeding bore_lat_tol must require wall penetration (tol > physical max play)"
        assert 2.0 * self.rod_r + 0.002 < 2.0 * self.slot_half, \
            "the seat slot must admit the rod"
        assert self.z_seat + self.seat_z_tol < self.cheek_top, \
            "a rod seated within tolerance stays laterally captive below the cheek tops"
        assert self.seat_y_tol < self.slot_half, "seat_y_tol inside the slot half-gap"
        assert self.cradle_s + self.stop_h < self.roast_rest_z - self.bore_half - 0.002, \
            "backstop top below the resting block's bore bottom (rod clears the stop)"
        assert self.hang_z_min > self.roast_rest_z + 0.01, \
            "a block still resting on the cradle must NOT count as hanging"
        assert self.z_hang > self.hang_z_min + 0.015, \
            "the properly hanging block clears hang_z_min with margin"
        assert self.post_dx - self.post_sx / 2.0 > self.roast_s / 2.0 + 0.015, \
            "the hanging block fits between the posts with margin"
        assert self.roast_x_tol <= (self.post_dx - self.post_sx / 2.0
                                    - self.roast_s / 2.0 - 0.015), \
            "a block within roast_x_tol hangs strictly between the posts, AND a block " \
            "shoved along the rod to the post jam limit exceeds the band (revocable)"
        assert self.mouth_half > 4.0 * self.rod_r, \
            "the funnel mouth forgives seat placement by several rod diameters"
        # order forcing: a block threaded onto a seated rod's outboard overhang cannot
        # travel along the rod into the between-posts region — the POST blocks it in z
        # (block bottom far below the post top) and the CHEEK assembly blocks it in y
        # (block much wider than the cheeks' outer span) — and success demands it hang
        # between the posts (roast_x_tol).
        assert self.z_hang - self.roast_s / 2.0 < self.post_h - 0.02, \
            "the hanging block overlaps the post in z: it cannot ride over a post"
        assert self.roast_s / 2.0 > self.slot_half + self.cheek_w + 0.01, \
            "the hanging block overlaps the cheek assembly in y: it cannot pass a notch"
        assert self.station_y - self.station_jit > (self.cradle_s * 0.71 + self.post_sy / 2.0
                                                    + 0.05), \
            "the station never overlaps the rack"
        assert self.spit_x_hi < self.rack_x - self.post_dx - self.post_sx / 2.0 - 0.01, \
            "spit spawn x-range keeps its CENTRE clear of the rack (reset() additionally"\
            " rejection-samples both rod ENDPOINTS against the rack + station footprints)"
        assert 0.0 < self.w_entered < self.w_threaded < self.w_carried < 1.0

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) -------------------
    def notch_x(self, k: int) -> float:
        """World x of notch plane k (0 = -x post, 1 = +x post)."""
        return self.rack_x + (-self.post_dx if k == 0 else self.post_dx)


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("spit_roast")
class SpitRoastScene(BaseScene):
    cfg: SpitRoastSceneCfg

    def __init__(self, cfg: SpitRoastSceneCfg | None = None) -> None:
        super().__init__(cfg or SpitRoastSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, static rack, kinematic loading station, dynamic roast + spit."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rack": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=sp["rack"](
                    post_dx=c.post_dx, post_sx=c.post_sx, post_sy=c.post_sy,
                    post_h=c.post_h, beam_h=c.beam_h, slot_half=c.slot_half,
                    cheek_w=c.cheek_w, cheek_h=c.cheek_h, fun_len=c.fun_len,
                    fun_t=c.fun_t, mu=c.rack_mu, contact_offset=c.contact_offset),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.rack_x, 0.0, 0.0)),
            ),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station",
                spawn=sp["station"](
                    cradle_s=c.cradle_s, stop_x=c.stop_x, stop_t=c.stop_t,
                    stop_h=c.stop_h, mu=c.station_mu, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rack_x, c.station_y, 0.0)),
            ),
            "roast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Roast",
                spawn=sp["roast"](
                    roast_s=c.roast_s, bore_half=c.bore_half, mass=c.roast_mass,
                    mu=c.roast_mu, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x, c.station_y, c.roast_rest_z + 0.003)),
            ),
            "spit": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Spit",
                spawn=sp["spit"](
                    rod_r=c.rod_r, rod_half=c.rod_half, knob_r=c.knob_r,
                    mass=c.rod_mass, com_x=c.rod_com_x, mu=c.rod_mu,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.12, -c.station_y, 0.02)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "enable_external_forces_every_iteration": True,
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
        n = env.num_envs
        dev = env.device
        self.station: RigidObject = env.iscene["station"]
        self.roast: RigidObject = env.iscene["roast"]
        self.spit: RigidObject = env.iscene["spit"]
        self.env_origins = env.iscene.env_origins
        # Episode randomization state (smoke readback verifies these vary).
        self.side = torch.zeros(n, device=dev)          # +1: station at +y; -1: at -y
        self.station_yaw = torch.zeros(n, device=dev)   # rad (includes the 180-deg flip)
        self.station_xy = torch.zeros(n, 2, device=dev)  # env-local station centre
        # Latched rubric stage credit (in [0, w_carried]).
        self.best_stage = torch.zeros(n, device=dev)
        # External wrench buffers — post_step consumes + OWNS the bodies' wrench slots;
        # never call set_external_force_and_torque on roast/spit directly.
        self.spit_force = torch.zeros(n, 3, device=dev)    # world force at the spit CoM
        self.spit_torque = torch.zeros(n, 3, device=dev)   # world torque on the spit
        self.roast_force = torch.zeros(n, 3, device=dev)   # world force at the roast CoM
        self.roast_torque = torch.zeros(n, 3, device=dev)  # world torque on the roast

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: park the station left/right of the rack (yawed + jittered), rest
        the block on the cradle (bore roughly along the station axis), lay the spit on the
        floor on the OPPOSITE side (rejection-sampled clear of rack + station, free yaw)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(8, device=dev)  # burn: first post-seed draws are near-degenerate

        st = torch.zeros(m, 13, device=dev)
        st_roast = torch.zeros(m, 13, device=dev)
        st_spit = torch.zeros(m, 13, device=dev)
        for row in range(m):
            e = env_ids[row]
            side = 1.0 if float(torch.rand(1)) < 0.5 else -1.0
            flip = math.pi if float(torch.rand(1)) < 0.5 else 0.0
            yaw = flip + math.radians(c.station_yaw_deg) * (2.0 * float(torch.rand(1)) - 1.0)
            sx = c.rack_x + c.station_jit * (2.0 * float(torch.rand(1)) - 1.0)
            sy = side * c.station_y + c.station_jit * (2.0 * float(torch.rand(1)) - 1.0)
            self.side[e] = side
            self.station_yaw[e] = yaw
            self.station_xy[e, 0], self.station_xy[e, 1] = sx, sy
            st[row, 0:3] = origin[row] + torch.tensor([sx, sy, 0.0], device=dev)
            st[row, 3] = math.cos(yaw / 2.0)
            st[row, 6] = math.sin(yaw / 2.0)
            # block on the cradle: 3 mm drop, small yaw relative to the station
            ryaw = yaw + math.radians(c.roast_yaw_deg) * (2.0 * float(torch.rand(1)) - 1.0)
            st_roast[row, 0:3] = origin[row] + torch.tensor(
                [sx, sy, c.roast_rest_z + 0.003], device=dev)
            st_roast[row, 3] = math.cos(ryaw / 2.0)
            st_roast[row, 6] = math.sin(ryaw / 2.0)
            # spit on the floor, opposite side; endpoints clear of rack + station
            px, py, pyaw = 0.12, -side * c.station_y, math.pi / 2.0  # deterministic fallback
            for _try in range(40):
                tx = c.spit_x_lo + (c.spit_x_hi - c.spit_x_lo) * float(torch.rand(1))
                ty = -side * (c.spit_y_lo + (c.spit_y_hi - c.spit_y_lo) * float(torch.rand(1)))
                tyaw = 2.0 * math.pi * float(torch.rand(1))
                ok = True
                for sgn in (-1.0, 1.0):
                    ex = tx + sgn * c.rod_half * math.cos(tyaw)
                    ey = ty + sgn * c.rod_half * math.sin(tyaw)
                    if (c.rack_x - c.post_dx - 0.09 < ex < c.rack_x + c.post_dx + 0.09
                            and -0.09 < ey < 0.09):
                        ok = False  # rack footprint
                    if (ex - sx) ** 2 + (ey - sy) ** 2 < 0.13 ** 2:
                        ok = False  # station disc
                if ok:
                    px, py, pyaw = tx, ty, tyaw
                    break
            st_spit[row, 0:3] = origin[row] + torch.tensor([px, py, 0.018], device=dev)
            st_spit[row, 3] = math.cos(pyaw / 2.0)
            st_spit[row, 6] = math.sin(pyaw / 2.0)
        self.station.write_root_state_to_sim(st, env_ids)
        self.roast.write_root_state_to_sim(st_roast, env_ids)
        self.spit.write_root_state_to_sim(st_spit, env_ids)

        self.best_stage[env_ids] = 0.0
        self.spit_force[env_ids] = 0.0
        self.spit_torque[env_ids] = 0.0
        self.roast_force[env_ids] = 0.0
        self.roast_torque[env_ids] = 0.0

    # ----- readings / rubric ---------------------------------------------------------------------
    def _axes(self):
        """Spit endpoints in env-local world coords: (tip e0, knob-centre e1), plus the
        roast pose. All (N, 3) / (N, 4)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        p_s = self.spit.data.root_pos_w - self.env_origins
        q_s = self.spit.data.root_quat_w
        xhat = quat_apply(q_s, torch.tensor([1.0, 0.0, 0.0],
                                            device=p_s.device).expand(p_s.shape[0], 3))
        e0 = p_s - c.rod_half * xhat
        e1 = p_s + c.rod_half * xhat
        p_r = self.roast.data.root_pos_w - self.env_origins
        q_r = self.roast.data.root_quat_w
        return e0, e1, p_r, q_r

    def _bore_status(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(entered, threaded): rod-vs-bore, judged in the ROAST body frame (the bore runs
        along roast body x, centred, half-length roast_s/2, half-aperture bore_half)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        e0, e1, p_r, q_r = self._axes()
        l0 = quat_apply_inverse(q_r, e0 - p_r)
        l1 = quat_apply_inverse(q_r, e1 - p_r)
        H = c.roast_s / 2.0
        dx = l1[:, 0] - l0[:, 0]
        safe = dx.abs() > 0.15  # rod roughly aligned with the bore axis
        dxg = torch.where(dx.abs() < 1e-6, torch.full_like(dx, 1e-6), dx)

        def lat_at(a: torch.Tensor) -> torch.Tensor:
            t = ((a - l0[:, 0]) / dxg).clamp(0.0, 1.0).unsqueeze(1)
            p = l0 + t * (l1 - l0)
            return torch.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)

        xmin = torch.minimum(l0[:, 0], l1[:, 0])
        xmax = torch.maximum(l0[:, 0], l1[:, 0])
        ov_lo = torch.maximum(xmin, torch.full_like(xmin, -H))
        ov_hi = torch.minimum(xmax, torch.full_like(xmax, H))
        entered = (safe & (ov_hi - ov_lo >= c.entry_min)
                   & (lat_at(ov_lo) <= c.bore_lat_tol) & (lat_at(ov_hi) <= c.bore_lat_tol))
        hh = torch.full_like(xmin, H)
        threaded = (safe & (xmin <= -H - c.protrude_min) & (xmax >= H + c.protrude_min)
                    & (lat_at(-hh) <= c.bore_lat_tol) & (lat_at(hh) <= c.bore_lat_tol))
        return entered, threaded

    def seated(self, k: int) -> torch.Tensor:
        """(N,) bool: the rod lies across notch k — its x-extent covers the notch plane
        with margin, roughly x-aligned, and its axis point AT that plane sits in the slot
        (|y| <= seat_y_tol, |z - z_seat| <= seat_z_tol)."""
        c = self.cfg
        e0, e1, _, _ = self._axes()
        xk = c.notch_x(k)
        dx = e1[:, 0] - e0[:, 0]
        dxg = torch.where(dx.abs() < 1e-6, torch.full_like(dx, 1e-6), dx)
        t = ((xk - e0[:, 0]) / dxg).clamp(0.0, 1.0).unsqueeze(1)
        p = e0 + t * (e1 - e0)
        cover = ((torch.minimum(e0[:, 0], e1[:, 0]) <= xk - c.seat_x_margin)
                 & (torch.maximum(e0[:, 0], e1[:, 0]) >= xk + c.seat_x_margin))
        return (cover & (dx.abs() > 0.25)
                & (p[:, 1].abs() <= c.seat_y_tol)
                & ((p[:, 2] - c.z_seat).abs() <= c.seat_z_tol))

    def entered(self) -> torch.Tensor:
        return self._bore_status()[0]

    def threaded(self) -> torch.Tensor:
        return self._bore_status()[1]

    def carried(self) -> torch.Tensor:
        """(N,) bool: threaded AND the block riding high (above any support surface)."""
        c = self.cfg
        _, thr = self._bore_status()
        p_r = self.roast.data.root_pos_w - self.env_origins
        return thr & (p_r[:, 2] > c.hang_z_min)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for b in (self.spit, self.roast):
            ok = (ok & (b.data.root_lin_vel_w.norm(dim=1) < c.settle_lin)
                  & (b.data.root_ang_vel_w.norm(dim=1) < c.settle_ang))
        return ok

    def success(self) -> torch.Tensor:
        """(N,) bool, current physical state: rod threaded through the block, both ends
        resting in the two seat slots, block hanging between the posts, all settled."""
        c = self.cfg
        _, thr = self._bore_status()
        p_r = self.roast.data.root_pos_w - self.env_origins
        fin = torch.ones_like(thr)
        for b in (self.spit, self.roast):
            fin = fin & torch.isfinite(b.data.root_state_w).all(dim=1)
        return (thr & self.seated(0) & self.seated(1)
                & (p_r[:, 2] > c.hang_z_min)
                & ((p_r[:, 0] - c.rack_x).abs() <= c.roast_x_tol)
                & self.settled() & fin)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 1.0 iff success() holds NOW; otherwise the latched best
        stage credit (entered 0.15 / threaded 0.40 / carried 0.65). Null policy ~0."""
        return torch.where(self.success(), torch.ones_like(self.best_stage), self.best_stage)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Apply the external wrench buffers (the scene owns both bodies' wrench slots),
        then latch rubric stage credit."""
        c = self.cfg
        n = self.env.num_envs
        # The wrench buffers are WORLD-frame by contract. On this pod's IsaacLab the
        # `is_global=True` path applies a stale-reference rotation (a tilted hold turns
        # the force cap into a runaway thruster), so pre-encode per step into the BODY
        # frame with the live quat and use the default (body-frame) call.
        from isaaclab.utils.math import quat_apply_inverse

        q_s = self.spit.data.root_quat_w
        q_r = self.roast.data.root_quat_w
        self.spit.set_external_force_and_torque(
            quat_apply_inverse(q_s, self.spit_force).view(n, 1, 3),
            quat_apply_inverse(q_s, self.spit_torque).view(n, 1, 3))
        self.roast.set_external_force_and_torque(
            quat_apply_inverse(q_r, self.roast_force).view(n, 1, 3),
            quat_apply_inverse(q_r, self.roast_torque).view(n, 1, 3))

        ent, thr = self._bore_status()
        p_r = self.roast.data.root_pos_w - self.env_origins
        car = thr & (p_r[:, 2] > c.hang_z_min)
        stage = torch.where(
            car, torch.full_like(self.best_stage, c.w_carried),
            torch.where(thr, torch.full_like(self.best_stage, c.w_threaded),
                        torch.where(ent, torch.full_like(self.best_stage, c.w_entered),
                                    torch.zeros_like(self.best_stage))))
        # A diverged substep must not latch: torch.maximum propagates NaN and best_stage
        # is checkpointed via get_state/set_state. A garbage frame earns NO progress.
        stage = torch.nan_to_num(stage, nan=0.0, posinf=0.0, neginf=0.0)
        self.best_stage = torch.maximum(self.best_stage, stage)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"station": self.station, "roast": self.roast, "spit": self.spit}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "station_yaw", "station_xy", "best_stage",
                               "spit_force", "spit_torque", "roast_force", "roast_torque")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"station": self.station, "roast": self.roast, "spit": self.spit}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A roasting rack stands fixed on the ground at x={c.rack_x:.2f}, y=0: two steel "
            f"posts {2 * c.post_dx:.2f} m apart along x (notch planes at x={c.notch_x(0):.2f} "
            f"and x={c.notch_x(1):.2f}), each {c.post_h:.3f} m tall. On top of each post is a "
            f"rod seat: a {2 * c.slot_half * 1000:.0f} mm wide slot between two cheeks (cheek "
            f"tops {c.cheek_top:.3f} m), flared by two 45-degree funnel plates opening to a "
            f"{2 * c.mouth_half * 1000:.0f} mm mouth (plate top edges {c.funnel_top:.3f} m). "
            f"A rod lying in both slots rests at height {c.z_seat:.3f} m on the y=0 line.\n"
            f"A wooden LOADING STATION (a {c.cradle_s:.2f} m cradle cube with a backstop "
            f"plate rising to {c.cradle_s + c.stop_h:.3f} m on one end) stands on the ground "
            f"roughly {c.station_y:.2f} m to one side of the rack (+y or -y, yawed and "
            f"jittered — its pose is randomized every episode; read it from the scene). On "
            f"the cradle rests the MEAT BLOCK: a {c.roast_s * 100:.0f} cm cube of meat with a "
            f"straight {2 * c.bore_half * 1000:.0f} mm square channel bored horizontally "
            f"through its centre, aligned away from the backstop. The channel's bottom sits "
            f"just above the backstop top, so a rod sliding through the channel clears the "
            f"backstop while the block's face presses against it.\n"
            f"The SPIT lies on the floor on the opposite side of the rack (pose randomized): "
            f"a steel rod {2 * c.rod_r * 1000:.0f} mm across and {2 * c.rod_half:.2f} m long "
            f"with a bare TIP on one end and a {2 * c.knob_r * 1000:.0f} mm ball KNOB on the "
            f"other. The knob is wider than the channel and cannot pass through it.\n"
            f"Goal: thread the spit's bare tip through the block's channel until the rod "
            f"sticks out at least {c.protrude_min * 1000:.0f} mm past BOTH faces (push until "
            f"the block presses the backstop — the knob end stays outside), then lift the "
            f"loaded spit, carry it to the rack, and lay it into the two seats: at each "
            f"notch plane the rod must rest within {c.seat_y_tol * 1000:.0f} mm of y=0 and "
            f"{c.seat_z_tol * 1000:.0f} mm of height {c.z_seat:.3f} m, with the meat hanging "
            f"on the rod BETWEEN the posts (block centre within {c.roast_x_tol * 100:.1f} cm "
            f"of x={c.rack_x:.2f}, hanging above {c.hang_z_min:.3f} m), everything at rest. "
            f"Laying the bare rod in the rack scores nothing; perching the block on top of "
            f"the rod or rack scores nothing; the block cannot be threaded onto an already "
            f"racked rod (the posts and funnels block its path) — thread first, then hang."
        )

    def instruction(self) -> str:
        return (
            "Pick up the steel spit rod and slide its bare tip all the way through the "
            "square channel in the meat block on the loading cradle, until the block rests "
            "against the backstop and the tip sticks out the far side. Then lift the loaded "
            "spit by its knob end, carry it over to the roasting rack, and lower its two "
            "ends into the slotted brackets on top of the posts, so the meat hangs "
            "suspended on the rod between the posts."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.spit_roast" -----------------
register_env("simgen", lambda: EnvCfg(scene="spit_roast", robot="null"))
