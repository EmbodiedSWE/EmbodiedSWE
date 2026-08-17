"""LiddedShoeBoxScene — pack a PAIR of heel-counter shoes heel-to-toe under a plug lid
(sim_gen task `put_shoes_in_box_i143`).

Derived from rlbench/put_shoes_in_box, but STRATEGICALLY different: the seed is a bare
pick-and-drop — an always-OPEN box and two free shoes; carry each shoe over the mouth,
release, done. No closure, no fit constraint, no ordering: any pose inside the box
counts and the box never has to be operated. Here the box is a CLOSED container with a
removable PLUG LID (it starts seated, sealing the mouth — nothing can be put in until
the lid is taken OFF), and the interior is deliberately one size too small for naive
packing: each shoe carries a WIDE HEEL COUNTER (72 mm) on a NARROW SOLE (32 mm), and
the interior width (132 mm) admits both shoes lying flat ONLY in the heel-beside-toe
antiparallel nesting (needs 124 mm); side-by-side same-heading needs 144 mm and
physically jams, one shoe on top of the other stands 62 mm proud of the 50 mm plug
line, a side-lying or upright shoe is taller than the cavity. The lid is the physical
VERIFIER: its plug (4 mm per-side clearance) can only drop into the mouth and the
plate only rest flush on the rim if neither shoe pokes above the plug line — a
mis-packed box leaves the lid riding >= 8 mm proud. The solver therefore needs a
different PLAN (open the container first; REASON about shoe headings — flip one shoe
180 degrees; close and verify flush) and different CODE STRUCTURE (a box-frame
containment predicate, a seated-lid pose predicate, latched open/packed credit), not
a drop-in-the-open-box routine.

Execution order is enforced by GEOMETRY, not rubric fiat: while the lid is seated the
plate+plug seal the whole mouth (smoke presses a shoe onto the closed lid with 3x its
weight — it never enters). Between the shoes there is no required order.

success(): both shoes fully inside the cavity below the rim (three body points of
each shoe inside the interior box in the BOX FRAME), the lid seated flush (origin
within 6 mm of the box axis, plate height within 2 mm of the seated height, handle-up
within 5 degrees), everything settled and finite — all live physical outcomes.

score() (latched credit never evaporates): 0.15 * lid ever fully CLEAR of the mouth
+ 0.25 * each shoe ever inside-and-settled, capped at 0.65; exactly 1.0 iff
success() live. Doing nothing scores ~0 (the lid starts seated, the shoes outside).

Assets are fully procedural (no external files):
  - box: KINEMATIC compound — floor slab + four walls, tan; interior
    165 x 132 x 58 mm; the box frame origin is the interior floor-top center.
  - lid (dynamic, 250 g): steel-blue plate 205 x 172 x 12 mm with a centered plug
    157 x 124 x 8 mm underneath (per-side clearance 4 mm) and a graspable handle
    ridge 56 x 16 x 24 mm on top.
  - shoe (x2, dynamic, 150 g each): crimson sole 150 x 32 x 18 mm with a wide heel
    counter 50 x 72 x 26 mm at the rear (total height 44 mm; the plug line sits at
    50 mm — 6 mm clearance over a flat shoe, negative clearance over every
    mis-packed one).
All spawners author mass/CoM/inertia + friction material + contact offsets (1.5 mm)
in-func (custom spawn funcs apply no cfg schemas); bind() asserts the masses by
readback.

Per-episode randomization (readback-verified in smoke): box center xy jitter + FREE
yaw (the packing axis rotates — it must be read, not memorized), seated-lid micro
jitter, shoe ground spawns in a reach annulus with free yaw, keep-out resampled
(outside the box footprint, pairwise separated) with a deterministic fallback.
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


# ----- custom compound spawners (box / lid / shoe) ----------------------------------------------
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


def _box_prim(stage, path: str, size, center, color, contact_offset: float,
              material=None) -> None:
    """Author one axis-aligned box child prim (translate -> scale; authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_armor(root, mass: float, com, inertia, kinematic: bool) -> None:
    """RigidBody + explicit MassAPI (mass, CoM, diagonal inertia) + PhysX damping/solver
    armor, authored in-func (custom spawn funcs apply no cfg schemas)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    if not kinematic:
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px.CreateLinearDampingAttr(0.05)
        px.CreateAngularDampingAttr(0.05)
        px.CreateMaxDepenetrationVelocityAttr(0.5)
        px.CreateSolverPositionIterationCountAttr(16)
        px.CreateSolverVelocityIterationCountAttr(1)
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_box(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC box compound. Local frame: origin at the INTERIOR FLOOR-TOP CENTER,
    +z up; interior spans |x| < L/2, |y| < W/2, 0 < z < H."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_armor(root, 5.0, (0.0, 0.0, 0.0), (0.02, 0.02, 0.03), kinematic=True)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    L, W, H, tw, tf = cfg.in_l, cfg.in_w, cfg.in_h, cfg.wall_t, cfg.floor_t
    co = cfg.contact_offset
    _box_prim(stage, f"{prim_path}/floor", (L + 2 * tw, W + 2 * tw, tf),
              (0.0, 0.0, -tf / 2), cfg.box_color, co, mat)
    _box_prim(stage, f"{prim_path}/wall_yn", (L + 2 * tw, tw, H),
              (0.0, -(W + tw) / 2, H / 2), cfg.box_color, co, mat)
    _box_prim(stage, f"{prim_path}/wall_yp", (L + 2 * tw, tw, H),
              (0.0, (W + tw) / 2, H / 2), cfg.box_color, co, mat)
    _box_prim(stage, f"{prim_path}/wall_xn", (tw, W, H),
              (-(L + tw) / 2, 0.0, H / 2), cfg.box_color2, co, mat)
    _box_prim(stage, f"{prim_path}/wall_xp", (tw, W, H),
              ((L + tw) / 2, 0.0, H / 2), cfg.box_color2, co, mat)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic plug lid. Local frame: origin at the PLATE CENTER; plug below, handle up."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    mL = cfg.lid_mass
    ix = mL / 12 * (cfg.plate_w ** 2 + cfg.plate_t ** 2)
    iy = mL / 12 * (cfg.plate_l ** 2 + cfg.plate_t ** 2)
    iz = mL / 12 * (cfg.plate_l ** 2 + cfg.plate_w ** 2)
    _rigid_armor(root, mL, (0.0, 0.0, -0.001), (ix, iy, iz), kinematic=False)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box_prim(stage, f"{prim_path}/plate", (cfg.plate_l, cfg.plate_w, cfg.plate_t),
              (0.0, 0.0, 0.0), cfg.lid_color, co, mat)
    _box_prim(stage, f"{prim_path}/plug", (cfg.plug_l, cfg.plug_w, cfg.plug_t),
              (0.0, 0.0, -(cfg.plate_t + cfg.plug_t) / 2), cfg.lid_color, co, mat)
    _box_prim(stage, f"{prim_path}/handle",
              (cfg.handle_l, cfg.handle_w, cfg.handle_h),
              (0.0, 0.0, (cfg.plate_t + cfg.handle_h) / 2), cfg.handle_color, co, mat)
    return root


def _spawn_shoe(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic shoe. Local frame: origin at the SOLE CENTER (mid-thickness); toe at +x,
    the wide heel counter at -x on top of the sole rear."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    ms = cfg.shoe_mass
    h_tot = cfg.sole_t + cfg.heel_h
    ix = ms / 12 * (cfg.heel_w ** 2 + h_tot ** 2)
    iy = ms / 12 * (cfg.sole_l ** 2 + h_tot ** 2)
    iz = ms / 12 * (cfg.sole_l ** 2 + cfg.heel_w ** 2)
    _rigid_armor(root, ms, (-0.022, 0.0, 0.010), (ix, iy, iz), kinematic=False)
    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box_prim(stage, f"{prim_path}/sole", (cfg.sole_l, cfg.sole_w, cfg.sole_t),
              (0.0, 0.0, 0.0), cfg.shoe_color, co, mat)
    _box_prim(stage, f"{prim_path}/heel",
              (cfg.heel_l, cfg.heel_w, cfg.heel_h),
              (-(cfg.sole_l - cfg.heel_l) / 2, 0.0, (cfg.sole_t + cfg.heel_h) / 2),
              cfg.heel_color, co, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "box" not in _SPAWNER_CACHE:

        @configclass
        class BoxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_box)
            in_l: float = 0.165
            in_w: float = 0.132
            in_h: float = 0.058
            wall_t: float = 0.012
            floor_t: float = 0.012
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            box_color: tuple = (0.72, 0.55, 0.34)
            box_color2: tuple = (0.62, 0.45, 0.26)
            contact_offset: float = 0.0015

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            plate_l: float = 0.205
            plate_w: float = 0.172
            plate_t: float = 0.012
            plug_l: float = 0.157
            plug_w: float = 0.124
            plug_t: float = 0.008
            handle_l: float = 0.056
            handle_w: float = 0.016
            handle_h: float = 0.024
            lid_mass: float = 0.25
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            lid_color: tuple = (0.30, 0.42, 0.58)
            handle_color: tuple = (0.15, 0.20, 0.28)
            contact_offset: float = 0.0015

        @configclass
        class ShoeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shoe)
            sole_l: float = 0.150
            sole_w: float = 0.032
            sole_t: float = 0.018
            heel_l: float = 0.050
            heel_w: float = 0.072
            heel_h: float = 0.026
            shoe_mass: float = 0.15
            mu_static: float = 0.6
            mu_dynamic: float = 0.5
            shoe_color: tuple = (0.72, 0.10, 0.12)
            heel_color: tuple = (0.55, 0.06, 0.08)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(box=BoxSpawnerCfg, lid=LidSpawnerCfg, shoe=ShoeSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LiddedShoeBoxSceneCfg(BaseCfg):
    """Config for `LiddedShoeBoxScene`. The packing-interference honesty (only the flat
    antiparallel nesting fits under the plug line) is asserted in __post_init__."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    in_margin: float = tunable(0.002)     # shoe body points inside the interior minus this (m)
    in_z_min: float = tunable(-0.004)     # shoe points above the floor minus this (no tunneling)
    lid_xy_tol: float = tunable(0.006)    # seated lid origin within this of the box axis (m)
    lid_z_tol: float = tunable(0.002)     # seated plate height within this of nominal (m)
    lid_tilt_max_deg: float = tunable(5.0)  # lid handle-up within this of world-up (deg)
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.6)      # max |ang vel| when judging (rad/s)
    clear_x: float = tunable(0.19)        # lid-CLEAR latch: |x_loc| beyond this (plate off mouth)
    clear_y: float = tunable(0.16)        # ... or |y_loc| beyond this

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    box_jitter: float = tunable(0.05)     # box center xy jitter (+/- m)
    box_yaw_deg: float = tunable(180.0)   # box yaw, uniform +/- deg (FREE yaw: read the axis)
    lid_seat_jitter: float = tunable(0.002)   # seated-lid xy micro jitter (within plug slack)
    lid_yaw_jitter_deg: float = tunable(1.5)  # seated-lid yaw micro jitter
    spawn_r_min: float = tunable(0.20)    # shoe ground spawns: reach annulus (m)
    spawn_r_max: float = tunable(0.55)
    spawn_sep: float = tunable(0.14)      # pairwise shoe separation (m)

    # --- info: layout ---------------------------------------------------------------------------
    box_pos: tuple = info((0.34, 0.0))    # box center on the ground (nominal)
    # --- info: box geometry (local frame: origin at interior floor-top center) ------------------
    in_l: float = info(0.165)             # interior length (x)
    in_w: float = info(0.132)             # interior width (y)
    in_h: float = info(0.058)             # interior height (rim above the floor)
    wall_t: float = info(0.012)
    floor_t: float = info(0.012)
    # --- info: lid ------------------------------------------------------------------------------
    plate_l: float = info(0.205)
    plate_w: float = info(0.172)
    plate_t: float = info(0.012)
    plug_l: float = info(0.157)
    plug_w: float = info(0.124)
    plug_t: float = info(0.008)
    handle_l: float = info(0.056)
    handle_w: float = info(0.016)
    handle_h: float = info(0.024)
    lid_mass: float = info(0.25)
    # --- info: shoe -----------------------------------------------------------------------------
    sole_l: float = info(0.150)
    sole_w: float = info(0.032)
    sole_t: float = info(0.018)
    heel_l: float = info(0.050)
    heel_w: float = info(0.072)
    heel_h: float = info(0.026)
    shoe_mass: float = info(0.15)
    # --- info: physics / rubric weights ---------------------------------------------------------
    mu_static: float = info(0.6)
    mu_dynamic: float = info(0.5)
    contact_offset: float = info(0.0015)
    w_open: float = info(0.15)            # lid ever fully clear of the mouth (latched)
    w_shoe: float = info(0.25)            # each shoe ever inside-and-settled (latched)

    # Derived in __post_init__.
    box_z: float = 0.0                    # box root height (floor slab resting on ground)
    seat_z: float = 0.0                   # seated lid origin height in the BOX frame
    plug_bot: float = 0.0                 # plug underside height in the BOX frame when seated
    shoe_h: float = 0.0                   # total shoe height lying flat

    def __post_init__(self) -> None:
        self.box_z = self.floor_t + 0.0005
        self.seat_z = self.in_h + self.plate_t / 2
        self.plug_bot = self.in_h - self.plug_t
        self.shoe_h = self.sole_t + self.heel_h
        # --- packing honesty: ONLY the flat antiparallel nesting fits under the plug ---
        # antiparallel fits with clearance: heel_w + (heel_w/2 + sole_w/2) + slack <= in_w
        need_anti = self.heel_w + (self.heel_w + self.sole_w) / 2
        assert self.in_w >= need_anti + 0.006, "antiparallel nesting too tight"
        # same-heading side-by-side does NOT fit (heel next to heel)
        assert 2 * self.heel_w >= self.in_w + 0.010, "parallel packing must jam"
        # end-to-end lengthwise does not fit
        assert 2 * self.sole_l > self.in_l + 0.05, "lengthwise packing must jam"
        # a flat shoe clears the plug line; every mis-packed shoe does not
        assert self.plug_bot >= self.shoe_h + 0.004, "flat shoe must clear the plug"
        assert self.sole_t + self.shoe_h >= self.plug_bot + 0.008, "stacked must jam the lid"
        assert self.heel_w >= self.plug_bot + 0.010, "side-lying must jam the lid"
        # a rim-perched lid (plug not engaged) is clearly out of the z tolerance
        assert self.plug_t + self.plate_t / 2 >= self.plate_t / 2 + self.lid_z_tol + 0.004
        # plug clearance is real but small (funnel for the seat)
        assert 0.003 <= (self.in_l - self.plug_l) / 2 <= 0.006
        assert 0.003 <= (self.in_w - self.plug_w) / 2 <= 0.006
        # single shoe passes the mouth easily; graspable by the 80 mm Franka jaw
        assert self.sole_l < self.in_l - 0.010 and self.heel_w < self.in_w - 0.030
        assert self.heel_l <= 0.078 and self.handle_w <= 0.078 and self.sole_w <= 0.078
        # lid-CLEAR latch really means the whole plate is off the mouth
        assert self.clear_x - self.plate_l / 2 > self.in_l / 2
        assert self.clear_y - self.plate_w / 2 > self.in_w / 2


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


def _qx(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 1] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("shoe_box_lid")
class LiddedShoeBoxScene(BaseScene):
    cfg: LiddedShoeBoxSceneCfg

    SHOE_NAMES = ("shoe_a", "shoe_b")

    def __init__(self, cfg: LiddedShoeBoxSceneCfg | None = None) -> None:
        super().__init__(cfg or LiddedShoeBoxSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        box_spawn = sp["box"](
            in_l=c.in_l, in_w=c.in_w, in_h=c.in_h, wall_t=c.wall_t, floor_t=c.floor_t,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic, contact_offset=c.contact_offset)
        lid_spawn = sp["lid"](
            plate_l=c.plate_l, plate_w=c.plate_w, plate_t=c.plate_t,
            plug_l=c.plug_l, plug_w=c.plug_w, plug_t=c.plug_t,
            handle_l=c.handle_l, handle_w=c.handle_w, handle_h=c.handle_h,
            lid_mass=c.lid_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box",
                spawn=box_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.box_pos[0], c.box_pos[1], c.box_z)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.box_pos[0], c.box_pos[1], c.box_z + c.seat_z + 0.001)),
            ),
        }
        for i, name in enumerate(self.SHOE_NAMES):
            shoe_spawn = sp["shoe"](
                sole_l=c.sole_l, sole_w=c.sole_w, sole_t=c.sole_t,
                heel_l=c.heel_l, heel_w=c.heel_w, heel_h=c.heel_h,
                shoe_mass=c.shoe_mass, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
                contact_offset=c.contact_offset)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shoe_" + name,
                spawn=shoe_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.30 + 0.15 * i, -0.35, c.sole_t / 2 + 0.002)),
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
        c = self.cfg
        self.box: RigidObject = env.iscene["box"]
        self.lid: RigidObject = env.iscene["lid"]
        self.shoes: dict[str, RigidObject] = {n: env.iscene[n] for n in self.SHOE_NAMES}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # authored-mass readback (custom spawn funcs apply no cfg schemas — verify)
        for body, m_want in ((self.lid, c.lid_mass),
                             (self.shoes["shoe_a"], c.shoe_mass),
                             (self.shoes["shoe_b"], c.shoe_mass)):
            m_got = float(body.root_physx_view.get_masses().reshape(-1)[0])
            assert abs(m_got - m_want) < 0.02, f"authored mass lost: {m_got} vs {m_want}"
        # shoe body points judged for containment (toe end, heel end, heel top), local
        self._pts_local = torch.tensor(
            [[(c.sole_l / 2, 0.0, 0.0),
              (-c.sole_l / 2, 0.0, 0.0),
              (-(c.sole_l - c.heel_l) / 2, 0.0, c.sole_t / 2 + c.heel_h)]],
            device=dev).expand(n, 3, 3).contiguous()
        # latches (partial credit survives transients; success is judged live)
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._packed = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: box at jittered center with FREE yaw, lid SEATED on it (micro
        jitter), shoes scattered flat on the ground (annulus, keep-out resampled with a
        deterministic fallback), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- box: kinematic, center jitter + free yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.box_yaw_deg)
        q_box = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = c.box_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
        bp[:, 1] = c.box_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
        bp[:, 2] = c.box_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_box
        self.box.write_root_state_to_sim(st, env_ids)

        # --- lid: seated on the box (micro jitter inside the plug slack) ---
        from isaaclab.utils.math import quat_apply

        jit = torch.zeros(m, 3, device=dev)
        jit[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.lid_seat_jitter
        jit[:, 2] = c.seat_z + 0.001
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin + quat_apply(q_box, jit)
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.lid_yaw_jitter_deg)
        st[:, 3:7] = _qmul(q_box, _qz(dyaw))
        self.lid.write_root_state_to_sim(st, env_ids)

        # --- shoes: flat on the ground, free yaw, keep-out resampled ---
        cpsi, spsi = torch.cos(yaw), torch.sin(yaw)
        xy = torch.zeros(m, 2, 2, device=dev)
        bad = torch.ones(m, 2, dtype=torch.bool, device=dev)
        for _ in range(40):
            if not bad.any():
                break
            k = int(bad.sum())
            cand = (torch.rand(k, 2, device=dev) * 2 - 1) * (c.spawn_r_max + 0.02)
            xy[bad] = cand
            r = xy.norm(dim=-1)
            ok = (r > c.spawn_r_min) & (r < c.spawn_r_max)
            rel = xy - bp[:, None, 0:2]
            u = rel[..., 0] * cpsi[:, None] + rel[..., 1] * spsi[:, None]
            v = -rel[..., 0] * spsi[:, None] + rel[..., 1] * cpsi[:, None]
            ok &= ~((u.abs() < 0.20) & (v.abs() < 0.18))          # off the box+lid footprint
            d = (xy[:, 0, :] - xy[:, 1, :]).norm(dim=-1)
            ok &= (d > c.spawn_sep).unsqueeze(-1)                  # pairwise separation
            bad = ~ok
        if bad.any():  # deterministic fallback, guaranteed clear of the jittered box
            fb = torch.tensor([[-0.42, -0.15], [-0.08, -0.44]], device=dev)
            xy[bad] = fb.unsqueeze(0).expand(m, 2, 2)[bad]
        syaw = torch.rand(m, 2, device=dev) * 2 * math.pi
        for i, name in enumerate(self.SHOE_NAMES):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xy[:, i, 0]
            st[:, 1] = xy[:, i, 1]
            st[:, 2] = c.sole_t / 2 + 0.002
            st[:, 3:7] = _qz(syaw[:, i])
            st[:, 0:3] += origin
            self.shoes[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._opened[env_ids] = False
        self._packed[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "box": self.box.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "shoes": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.shoes.items()},
            "opened": self._opened[env_ids].clone(),
            "packed": self._packed[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        for n, b in self.shoes.items():
            b.write_root_state_to_sim(state["shoes"][n], env_ids)
        self._opened[env_ids] = state["opened"]
        self._packed[env_ids] = state["packed"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A TAN WOODEN BOX (interior {c.in_l * 1000:.0f} x {c.in_w * 1000:.0f} mm, "
            f"{c.in_h * 1000:.0f} mm deep) stands on the floor, CLOSED by a STEEL-BLUE "
            f"LID: a flat plate with a plug underneath that nests into the box mouth and "
            f"a dark HANDLE RIDGE on top. The box's position and heading change every "
            f"episode — read its axis from the scene. Scattered on the floor lie a pair "
            f"of CRIMSON SHOES: each is a narrow flat sole ({c.sole_l * 1000:.0f} mm long, "
            f"{c.sole_w * 1000:.0f} mm wide) carrying a WIDE DARK-RED HEEL COUNTER "
            f"({c.heel_w * 1000:.0f} mm wide, {c.heel_l * 1000:.0f} mm long) at its rear; "
            f"lying flat a shoe is {(c.sole_t + c.heel_h) * 1000:.0f} mm tall.\n"
            f"Goal: pack BOTH shoes inside the box and close it. First take the lid OFF "
            f"(grasp the handle ridge; set the lid aside — nothing fits through a closed "
            f"lid). The interior is only {c.in_w * 1000:.0f} mm wide: two heel counters "
            f"side by side ({2 * c.heel_w * 1000:.0f} mm) do NOT fit, so the shoes must "
            f"lie flat HEEL-TO-TOE — pointing in OPPOSITE directions along the box's long "
            f"axis, each heel counter beside the other shoe's narrow sole. A shoe on its "
            f"side, on end, leaning on the rim, or on top of the other stands taller than "
            f"the cavity and the lid will not close over it. Finally put the lid back "
            f"handle-UP and seat it FLUSH: the plug drops into the mouth and the plate "
            f"rests level on the rim. A lid riding high, tilted, offset off the mouth, or "
            f"upside-down does not count. Success: both shoes fully inside below the rim, "
            f"lid seated flush, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Take the blue lid off the tan box and set it aside. Lay both crimson shoes "
            "flat inside, heel-to-toe in opposite directions so both fit below the rim — "
            "side by side the wide heels do not fit. Then put the lid back handle-up and "
            "seat it flush on the rim; the box only closes if both shoes lie flat inside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _box_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> box frame (kinematic root, live-read)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.box.data.root_quat_w,
                                  pos_w - self.box.data.root_pos_w)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _status(self) -> dict[str, torch.Tensor]:
        """Live geometric predicates ((N,) or (N,2))."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        # shoes: three body points each, inside the interior volume below the rim
        inside_l, settle_l = [], []
        for i, name in enumerate(self.SHOE_NAMES):
            b = self.shoes[name]
            q = b.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
            pts_w = b.data.root_pos_w[:, None, :] + quat_apply(
                q, self._pts_local.reshape(n * 3, 3)).reshape(n, 3, 3)
            loc = self._box_local(pts_w.reshape(n * 3, 3)).reshape(n, 3, 3)
            ok = (loc[:, :, 0].abs() < c.in_l / 2 - c.in_margin) \
                & (loc[:, :, 1].abs() < c.in_w / 2 - c.in_margin) \
                & (loc[:, :, 2] < c.in_h) & (loc[:, :, 2] > c.in_z_min)
            inside_l.append(ok.all(dim=1))
            settle_l.append(self._settled(b))
        inside = torch.stack(inside_l, dim=1)
        sh_settled = torch.stack(settle_l, dim=1)

        # lid: box-frame pose
        lid_loc = self._box_local(self.lid.data.root_pos_w)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        lid_up = quat_apply(self.lid.data.root_quat_w, ez)
        upright = lid_up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.lid_tilt_max_deg))
        seated = (lid_loc[:, 0:2].norm(dim=-1) < c.lid_xy_tol) \
            & ((lid_loc[:, 2] - c.seat_z).abs() < c.lid_z_tol) \
            & upright & self._settled(self.lid)
        clear = (lid_loc[:, 0].abs() > c.clear_x) | (lid_loc[:, 1].abs() > c.clear_y)
        return {"inside": inside, "sh_settled": sh_settled, "seated": seated,
                "clear": clear, "lid_loc": lid_loc, "lid_up_z": lid_up[:, 2]}

    def _update_latches(self) -> None:
        s = self._status()
        self._opened |= s["clear"]
        self._packed |= s["inside"] & s["sh_settled"]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both shoes fully inside the cavity below the rim, the lid seated
        flush handle-up on the rim (plug engaged: centered within 6 mm, plate height
        within 2 mm, upright within 5 deg), everything settled and finite — all live
        physical outcomes. A protruding, mis-packed shoe physically prevents the lid
        clauses; nothing here is bookkeeping."""
        self._update_latches()
        s = self._status()
        pos = torch.stack([b.data.root_pos_w for b in self.shoes.values()]
                          + [self.lid.data.root_pos_w], dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return (s["inside"] & s["sh_settled"]).all(dim=1) & s["seated"] & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*lid ever clear of the mouth + 0.25*each shoe ever
        inside-and-settled (all latched; ~0 for doing nothing — the lid starts seated),
        capped at 0.65 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened.float()
                + c.w_shoe * self._packed.float().sum(dim=1)).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="shoe_box_lid", robot="null"))
