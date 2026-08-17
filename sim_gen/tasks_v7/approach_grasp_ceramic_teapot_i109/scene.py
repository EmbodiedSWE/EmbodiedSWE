"""MugRackHangScene — hang the WHITE mug by its closed handle on the GREEN-marked peg
of a three-peg wall rack (sim_gen task `approach_grasp_ceramic_teapot_i109`).

Derived from pick_place/approach_grasp_ceramic_teapot, but STRATEGICALLY different:
the seed is a prehensile APPROACH-AND-GRASP — a Franka approaches a ceramic teapot
among table clutter, closes its jaw when the gripper-object distance drops below
2 cm, holds it a few frames and lifts; success is a gripper-object relation and the
episode ends HOLDING the object. Here holding is worth nothing and no gripper
relation is ever read: the judged outcome is a SUSPENDED EQUILIBRIUM — the white
mug hanging free in the air, its closed handle window threaded over one specific
cantilevered peg, supported by peg contact and gravity alone after the hand lets
go. An episode that ends with the mug "acquired", carried, or set down anywhere
scores at most the small carry latches; only the released, settled hang scores.

The scene: a KINEMATIC wall rack (base plate + upright panel) carrying three
square steel PEGS that cantilever from the panel face, tilted 12 deg upward, at
three different heights and lateral stations. A small green MARKER block sits on
the panel's top edge directly above ONE peg's column — the target peg — and is
re-posed every episode (1-of-3, uniform). Two identical open-topped MUGS with
closed loop handles (window 26 x 39 mm, peg 14 mm square) start upright on the
floor: one WHITE (judged) and one BLACK (decoy). The solver must read the marker,
pick the white mug, thread its handle window over the green peg, and release: the
upward peg tilt and slick peg surface let gravity slide the hang to the peg root
where the mug leans against the panel and comes to rest.

success(): the green peg's axis passes THROUGH the white mug's handle window
(computed in the mug's frame: the peg centerline crosses the open window rectangle
within the peg's span), the mug is AIRBORNE (center high above every non-hang
support in the scene — floor, base plate, even one mug stacked on the other,
all sit >= 15 mm below the gate; hang states sit >= 35 mm above it, asserted in
__post_init__), and it has HUNG STILL for an uninterrupted streak of substeps
(velocity gates) — a live suspended equilibrium, never bookkept. Hanging the
black mug, hanging on a gray peg, hanging the mug by its RIM (peg inside the cup
body), standing the mug anywhere, or holding it in the air all fail exactly one
clause.

score() is latched (credit never evaporates): 0.10 carried-near + 0.10 lifted-near
+ 0.20 ever-threaded + 0.20 ever-hung-settled, cap 0.60; exactly 1.0 iff success()
live. Doing nothing scores 0 (both mugs start on the floor, far from every peg).

Assets are fully procedural (no external files):
  - rack (KINEMATIC compound): base plate + upright panel + three tilted square
    pegs (slick physics material — tan(12 deg) exceeds the peg friction, so a hung
    handle slides to the root; asserted).
  - marker (KINEMATIC cube): green target flag on the panel top edge, re-posed at
    reset; sits above the swept volume of any hung mug (asserted).
  - mugs (x2, DYNAMIC compounds, 0.25 kg): 64 mm square cup (4 walls + floor) with
    a closed 3-bar loop handle forming a 26 x 39 mm open window. Mass / CoM /
    diagonal inertia authored explicitly.

Per-episode randomization (readback-verified in smoke): target peg (1-of-3, drawn
via torch.rand comparison), rack yaw +/-25 deg + xy jitter, both mug floor spawns
(annulus, keep-outs, free yaw). Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
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


# ----- USD authoring helpers ---------------------------------------------------------------------
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


def _decorate(prim, color, contact_offset: float, material=None) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None) -> None:
    """Author one box child prim (translate -> orient -> scale; authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _decorate(seg.GetPrim(), color, contact_offset, material)


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC rack. Local frame: origin on the ground under the panel
    center; the panel face is at x = `face_x` (negative side); pegs cantilever
    toward local -x, tilted `peg_tilt_deg` upward, rooted on the face at
    (face_x, peg_ys[i], peg_zs[i])."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(25.0)

    grippy = _friction_material(stage, f"{prim_path}/mat_wood", cfg.mu_static, cfg.mu_dynamic)
    slick = _friction_material(stage, f"{prim_path}/mat_peg", cfg.peg_mu_static, cfg.peg_mu_dynamic)
    co = cfg.contact_offset

    bx, by, bz = cfg.base_size
    _box(stage, f"{prim_path}/base", (bx, by, bz), (0.0, 0.0, bz / 2),
         cfg.wood_color, co, material=grippy)
    px_, py, pz = cfg.panel_size
    _box(stage, f"{prim_path}/panel", (px_, py, pz), (0.0, 0.0, bz + pz / 2),
         cfg.wood_color, co, material=grippy)

    # pegs: long axis rotated from +x to (-cos t, 0, +sin t) == rotation about +y
    # by (180 + tilt) deg
    t = math.radians(cfg.peg_tilt_deg)
    half = math.radians(180.0 + cfg.peg_tilt_deg) / 2.0
    orient = (math.cos(half), 0.0, math.sin(half), 0.0)
    d = (-math.cos(t), 0.0, math.sin(t))
    for i, (yy, zz) in enumerate(zip(cfg.peg_ys, cfg.peg_zs)):
        cx = cfg.face_x + d[0] * cfg.peg_len / 2
        cz = zz + d[2] * cfg.peg_len / 2
        _box(stage, f"{prim_path}/peg_{i}", (cfg.peg_len, cfg.peg_w, cfg.peg_w),
             (cx, yy, cz), cfg.peg_color, co, material=slick, orient=orient)
    return root


def _spawn_mug(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC mug. Local frame: origin at the BODY center (cup midpoint,
    half-height); the handle loop sticks out in +x, its open window lying in the
    local x-z plane. Mass / CoM / inertia authored explicitly."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mug_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in cfg.mug_com]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.mug_inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.15)
    px.CreateAngularDampingAttr(0.6)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    w, h, wt, ft = cfg.body_w, cfg.body_h, cfg.wall_t, cfg.floor_t
    hw = w / 2
    # cup: floor + 4 walls (open top)
    _box(stage, f"{prim_path}/floor", (w, w, ft), (0.0, 0.0, -(h - ft) / 2),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/wall_e", (wt, w, h), (hw - wt / 2, 0.0, 0.0),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/wall_w", (wt, w, h), (-(hw - wt / 2), 0.0, 0.0),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/wall_n", (w - 2 * wt, wt, h), (0.0, hw - wt / 2, 0.0),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/wall_s", (w - 2 * wt, wt, h), (0.0, -(hw - wt / 2), 0.0),
         cfg.color, co, material=mat)
    # closed loop handle: top bar + bottom bar + outer post, window in between
    bt, bw = cfg.bar_t, cfg.bar_w
    x0, x1 = cfg.win_x0, cfg.win_x1          # window x-range (wall face .. post inner face)
    z0, z1 = cfg.win_z0, cfg.win_z1          # window z-range (bar inner faces)
    bar_len = (x1 + bt) - x0                  # bars run from the wall face over the post
    bar_cx = x0 + bar_len / 2
    _box(stage, f"{prim_path}/bar_top", (bar_len, bw, bt), (bar_cx, 0.0, z1 + bt / 2),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/bar_bot", (bar_len, bw, bt), (bar_cx, 0.0, z0 - bt / 2),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/bar_out", (bt, bw, (z1 - z0) + 2 * bt),
         (x1 + bt / 2, 0.0, (z0 + z1) / 2), cfg.color, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            base_size: tuple = (0.16, 0.30, 0.030)
            panel_size: tuple = (0.030, 0.26, 0.35)
            face_x: float = -0.015
            peg_ys: tuple = (-0.08, 0.0, 0.08)
            peg_zs: tuple = (0.24, 0.33, 0.285)
            peg_len: float = 0.085
            peg_w: float = 0.014
            peg_tilt_deg: float = 12.0
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            peg_mu_static: float = 0.15
            peg_mu_dynamic: float = 0.12
            wood_color: tuple = (0.45, 0.32, 0.20)
            peg_color: tuple = (0.74, 0.75, 0.78)
            contact_offset: float = 0.0015

        @configclass
        class MugSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_mug)
            body_w: float = 0.064
            body_h: float = 0.090
            wall_t: float = 0.006
            floor_t: float = 0.008
            bar_t: float = 0.008
            bar_w: float = 0.012
            win_x0: float = 0.032
            win_x1: float = 0.058
            win_z0: float = -0.0135
            win_z1: float = 0.0255
            mug_mass: float = 0.25
            mug_com: tuple = (0.004, 0.0, -0.006)
            mug_inertia: tuple = (3.2e-4, 3.4e-4, 2.4e-4)
            mu_static: float = 0.40
            mu_dynamic: float = 0.35
            color: tuple = (0.92, 0.92, 0.90)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rack"] = RackSpawnerCfg
        _SPAWNER_CACHE["mug"] = MugSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class MugRackHangSceneCfg(BaseCfg):
    """Config for `MugRackHangScene`. Honesty is asserted in __post_init__: the
    handle window admits the peg with real clearance, the upward peg tilt beats
    the peg friction (a hung handle slides to the root instead of dangling at the
    tip forever), the airborne gate separates every hang from every stand (even a
    mug stacked on the other mug), pegs are far enough apart that a hung mug
    cannot touch a neighbor peg, the marker sits above the swept volume of any
    hang, everything manipulated is jaw-sized, and the score weights sum to the
    cap."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    thread_dir_min: float = tunable(0.5)   # |peg dir . window normal| gate (peg roughly through)
    thread_margin: float = tunable(0.004)  # peg-axis span margin at root/tip (m)
    win_shrink: float = tunable(0.004)     # window shrink for the centerline test (< peg_w/2)
    airborne_z: float = tunable(0.155)     # mug center height gate (m) — above ALL stand states
    settle_lin: float = tunable(0.05)      # max |lin vel| while hanging still (m/s)
    settle_ang: float = tunable(0.40)      # max |ang vel| while hanging still (rad/s)
    hang_streak: int = tunable(60)         # substeps of uninterrupted still hang (0.5 s @ 120 Hz)
    near_tol: float = tunable(0.10)        # carried-near latch: |aperture - peg mid| (m)
    lift_near_tol: float = tunable(0.15)   # lifted-near latch: airborne + this close (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------------
    rack_yaw_deg: float = tunable(25.0)    # rack yaw +/- deg
    rack_jitter: float = tunable(0.03)     # rack xy jitter (+/- m)
    spawn_r_min: float = tunable(0.16)     # mug floor spawns: annulus around the origin (m)
    spawn_r_max: float = tunable(0.42)
    spawn_keepout: float = tunable(0.30)   # min mug distance from the rack center (m)
    spawn_sep: float = tunable(0.22)       # mug-mug floor separation (m)

    # --- info: rack layout --------------------------------------------------------------------------
    rack_pos: tuple = info((0.45, 0.0))    # rack origin on the ground (nominal)
    base_size: tuple = info((0.16, 0.30, 0.030))
    panel_size: tuple = info((0.030, 0.26, 0.35))   # panel top at 0.030 + 0.35 = 0.380
    face_x: float = info(-0.015)           # panel front face (pegs cantilever toward -x)
    peg_ys: tuple = info((-0.08, 0.0, 0.08))
    peg_zs: tuple = info((0.24, 0.33, 0.285))       # peg root heights on the face
    peg_len: float = info(0.085)
    peg_w: float = info(0.014)             # square cross-section
    peg_tilt_deg: float = info(12.0)       # upward tilt — beats peg friction (asserted)
    peg_mu_static: float = info(0.15)
    peg_mu_dynamic: float = info(0.12)
    # --- info: mug geometry (local frame: origin at body center; handle toward +x) -----------------
    body_w: float = info(0.064)
    body_h: float = info(0.090)
    wall_t: float = info(0.006)
    floor_t: float = info(0.008)
    bar_t: float = info(0.008)             # handle bar thickness (x/z)
    bar_w: float = info(0.012)             # handle loop thickness in y
    win_x0: float = info(0.032)            # window x-range: wall outer face .. post inner face
    win_x1: float = info(0.058)
    win_z0: float = info(-0.0135)          # window z-range: bar inner faces
    win_z1: float = info(0.0255)
    ap_local: tuple = info((0.045, 0.0, 0.006))     # window (aperture) center, mug frame
    mug_mass: float = info(0.25)
    mug_com: tuple = info((0.004, 0.0, -0.006))
    mug_inertia: tuple = info((3.2e-4, 3.4e-4, 2.4e-4))
    mug_names: tuple = info(("mug_white", "mug_black"))
    mug_colors: tuple = info(((0.92, 0.92, 0.90), (0.08, 0.08, 0.09)))
    # --- info: marker --------------------------------------------------------------------------------
    marker_size: float = info(0.024)
    marker_color: tuple = info((0.10, 0.75, 0.20))
    marker_z: float = info(0.392)          # marker center: sits on the panel top edge
    # --- info: shared physics ------------------------------------------------------------------------
    mu_static: float = info(0.60)
    mu_dynamic: float = info(0.50)
    mug_mu_static: float = info(0.40)
    mug_mu_dynamic: float = info(0.35)
    contact_offset: float = info(0.0015)
    # rubric weights (sum == the non-success cap)
    w_near: float = info(0.10)
    w_lift: float = info(0.10)
    w_thread: float = info(0.20)
    w_hang: float = info(0.20)
    score_cap: float = info(0.60)

    def __post_init__(self) -> None:
        t = math.radians(self.peg_tilt_deg)
        # window admits the peg with clearance, even rotated (worst diagonal in plane)
        diag = self.peg_w * math.sqrt(2.0)
        assert (self.win_x1 - self.win_x0) >= diag + 0.004, "window too narrow for the peg"
        assert (self.win_z1 - self.win_z0) >= diag + 0.012, "window too short for the peg"
        # the centerline window test is conservative: physics keeps the peg centerline
        # at least peg_w/2 - contact_offset from a touching window edge
        assert self.win_shrink < self.peg_w / 2 - self.contact_offset, "win_shrink too big"
        # upward tilt beats peg friction -> a hung handle slides to the peg root
        assert math.tan(t) > 1.3 * self.peg_mu_static, "peg too grippy: hang won't slide to root"
        # airborne gate separates every hang from every stand:
        #   hang: contact at the peg TOP surface; mug center below the contact by at
        #   most sqrt(dz^2 + dx^2) at a 60-deg worst-case tilt about the peg axis
        dz_c = self.win_z1              # window top edge (bar inner face) above mug center
        dx_c = self.win_x1              # farthest bar point from the mug center line
        drop60 = dz_c * 0.5 + dx_c * math.sin(math.radians(60.0))
        hang_min = min(self.peg_zs) + self.peg_w / 2 - drop60
        assert hang_min > self.airborne_z + 0.010, f"hang band {hang_min} vs airborne gate"
        # stands: floor, base plate, one mug stacked on the other (the tallest stand)
        stack_top = self.body_h + self.body_h / 2
        assert stack_top < self.airborne_z - 0.015, "stacked mugs could fake airborne"
        assert self.base_size[2] + self.body_h / 2 < self.airborne_z - 0.015
        # a floor mug can never trip the near latch (null-safety): vertical gap alone
        assert min(self.peg_zs) - self.body_h / 2 > self.near_tol + 0.02
        # pegs mutually clear: a mug hung on one peg (body half-width + swing slack)
        # cannot touch a neighbor peg
        ys = sorted(self.peg_ys)
        for a, b in zip(ys, ys[1:]):
            assert b - a >= self.body_w / 2 + self.peg_w / 2 + 0.02, "pegs too close"
        # hung mug (top of handle at peg top + bar) stays below the panel top; the
        # marker on the top edge is above every hang's swept volume
        hang_top = max(self.peg_zs) + self.peg_w / 2 + 2 * self.bar_t + 0.01
        panel_top = self.base_size[2] + self.panel_size[2]
        assert hang_top < panel_top, "hang sweeps past the panel top"
        assert self.marker_z - self.marker_size / 2 >= panel_top - 1e-6
        # peg span admits the handle loop with travel to spare
        assert self.peg_len > self.bar_w + 0.030, "peg too short to hang on"
        # jaw-sized (80 mm Franka jaw): mug body width, wall pinch
        assert self.body_w <= 0.078 and self.wall_t <= 0.078
        # everything within a front-mounted Franka's reach envelope
        tip_x = self.rack_pos[0] + self.face_x - self.peg_len * math.cos(t)
        assert 0.0 < tip_x < 0.72 and max(self.peg_zs) + 0.06 < 0.90
        # weights sum to the cap
        s = self.w_near + self.w_lift + self.w_thread + self.w_hang
        assert abs(s - self.score_cap) < 1e-9, "score weights must sum to the cap"


# ----- small quaternion helpers (wxyz, torch, batched) ----------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("mug_rack_hang")
class MugRackHangScene(BaseScene):
    cfg: MugRackHangSceneCfg

    def __init__(self, cfg: MugRackHangSceneCfg | None = None) -> None:
        super().__init__(cfg or MugRackHangSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        rack_spawn = sp["rack"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            base_size=c.base_size, panel_size=c.panel_size, face_x=c.face_x,
            peg_ys=c.peg_ys, peg_zs=c.peg_zs, peg_len=c.peg_len, peg_w=c.peg_w,
            peg_tilt_deg=c.peg_tilt_deg,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            peg_mu_static=c.peg_mu_static, peg_mu_dynamic=c.peg_mu_dynamic,
            contact_offset=c.contact_offset)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rack_pos[0], c.rack_pos[1], 0.0)),
            ),
            "marker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marker",
                spawn=sim_utils.CuboidCfg(
                    size=(c.marker_size, c.marker_size, c.marker_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.marker_color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], c.marker_z)),
            ),
        }
        for i, name in enumerate(c.mug_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mug_" + name,
                spawn=sp["mug"](
                    body_w=c.body_w, body_h=c.body_h, wall_t=c.wall_t, floor_t=c.floor_t,
                    bar_t=c.bar_t, bar_w=c.bar_w,
                    win_x0=c.win_x0, win_x1=c.win_x1, win_z0=c.win_z0, win_z1=c.win_z1,
                    mug_mass=c.mug_mass, mug_com=c.mug_com, mug_inertia=c.mug_inertia,
                    mu_static=c.mug_mu_static, mu_dynamic=c.mug_mu_dynamic,
                    color=c.mug_colors[i], contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.6 - 0.4 * i, -0.6, c.body_h / 2 + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # kills residual contact-velocity noise on a light hung body
                # (a chattering handle-on-peg contact flickers the settle gates otherwise)
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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.rack: RigidObject = env.iscene["rack"]
        self.marker: RigidObject = env.iscene["marker"]
        self.mugs: dict[str, RigidObject] = {n: env.iscene[n] for n in c.mug_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        t = math.radians(c.peg_tilt_deg)
        self._peg_roots = torch.tensor(
            [[c.face_x, y, z] for y, z in zip(c.peg_ys, c.peg_zs)], device=dev)
        self._peg_dir = torch.tensor([-math.cos(t), 0.0, math.sin(t)], device=dev)
        self._ap_local = torch.tensor(c.ap_local, device=dev)
        # per-episode target peg
        self.green_idx = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._threaded_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hung_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        # uninterrupted still-hang streak (substeps), advanced in post_step
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the rack (yaw + xy jitter), draw the target peg
        (1-of-3 via torch.rand comparison), park the green marker on the panel top
        edge above the target peg's column, scatter both mugs upright on the floor
        (annulus, keep-outs, free yaw), clear latches and streak. Caller settles."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rack: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        q_rack = _qz(psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rack_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        rp[:, 1] = c.rack_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rack_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rack
        self.rack.write_root_state_to_sim(st, env_ids)

        # --- target peg: uniform 1-of-3 (torch.rand comparison, not randint) ---
        u = torch.rand(m, device=dev)
        gi = torch.full((m,), 2, dtype=torch.long, device=dev)
        gi[u < 2.0 / 3.0] = 1
        gi[u < 1.0 / 3.0] = 0
        self.green_idx[env_ids] = gi

        # --- marker on the panel top edge above the target column ---
        from isaaclab.utils.math import quat_apply

        peg_y = torch.tensor(c.peg_ys, device=dev)[gi]
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 1] = peg_y
        loc[:, 2] = c.marker_z
        mk = torch.zeros(m, 13, device=dev)
        mk[:, 0:3] = rp + quat_apply(q_rack, loc) + origin
        mk[:, 3:7] = q_rack
        self.marker.write_root_state_to_sim(mk, env_ids)

        # --- mug floor spawns: 2 slots, keep-out resampled ---
        xy = torch.zeros(m, 2, 2, device=dev)
        bad = torch.ones(m, 2, dtype=torch.bool, device=dev)
        for _ in range(60):
            if not bad.any():
                break
            k = int(bad.sum())
            cand = torch.rand(k, 2, device=dev) * (2 * c.spawn_r_max) - c.spawn_r_max
            xy[bad] = cand
            rr = xy.norm(dim=-1)
            ok = (rr > c.spawn_r_min) & (rr < c.spawn_r_max)
            ok &= (xy - rp[:, None, 0:2]).norm(dim=-1) > c.spawn_keepout
            d = (xy[:, 0] - xy[:, 1]).norm(dim=-1, keepdim=True).expand(m, 2)
            ok &= d > c.spawn_sep
            bad = ~ok
        yaw = torch.rand(m, 2, device=dev) * 2 * math.pi

        for i, name in enumerate(c.mug_names):
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:2] = xy[:, i]
            s[:, 2] = c.body_h / 2 + 0.002
            s[:, 3:7] = _qz(yaw[:, i])
            s[:, 0:3] += origin
            self.mugs[name].write_root_state_to_sim(s, env_ids)

        # --- clear latches + streak ---
        self._near[env_ids] = False
        self._lifted[env_ids] = False
        self._threaded_ever[env_ids] = False
        self._hung_ever[env_ids] = False
        self._streak[env_ids] = 0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "marker": self.marker.data.root_state_w[env_ids].clone(),
            "mugs": {n: b.data.root_state_w[env_ids].clone() for n, b in self.mugs.items()},
            "green_idx": self.green_idx[env_ids].clone(),
            "near": self._near[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "threaded_ever": self._threaded_ever[env_ids].clone(),
            "hung_ever": self._hung_ever[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.marker.write_root_state_to_sim(state["marker"], env_ids)
        for n, b in self.mugs.items():
            b.write_root_state_to_sim(state["mugs"][n], env_ids)
        self.green_idx[env_ids] = state["green_idx"]
        self._near[env_ids] = state["near"]
        self._lifted[env_ids] = state["lifted"]
        self._threaded_ever[env_ids] = state["threaded_ever"]
        self._hung_ever[env_ids] = state["hung_ever"]
        self._streak[env_ids] = state["streak"]

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden wall RACK stands on the floor: a base plate and an upright "
            f"panel carrying three square steel PEGS that stick out toward you, "
            f"tilted {c.peg_tilt_deg:.0f} degrees upward, each at a different height "
            f"({', '.join(f'{z * 100:.0f}' for z in c.peg_zs)} cm) and lateral "
            f"position. A small GREEN MARKER block sits on the panel's top edge "
            f"directly above ONE peg — the target — and moves to a different peg's "
            f"column on different episodes. Two identical open mugs stand upright "
            f"on the floor nearby: one WHITE, one BLACK. Each mug has a closed loop "
            f"HANDLE whose open window ({(c.win_x1 - c.win_x0) * 1000:.0f} x "
            f"{(c.win_z1 - c.win_z0) * 1000:.0f} mm) fits over a peg "
            f"({c.peg_w * 1000:.0f} mm square). The rack pose, the target peg and "
            f"both mug spawns change every episode.\n"
            f"Goal: hang the WHITE mug on the marked peg — thread its handle window "
            f"over the peg below the green marker and let go, so the mug hangs "
            f"free in the air on that peg, supported only by the peg through its "
            f"handle, and comes to rest. The black mug is a decoy; hanging it, "
            f"using an unmarked peg, hooking the mug by its rim, or standing the "
            f"mug anywhere does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the white mug and hang it by its handle on the peg below the "
            "green marker, then let go so it hangs freely and comes to rest. "
            "Ignore the black mug and the unmarked pegs."
        )

    # ----- frames / live predicates ------------------------------------------------------------------
    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _peg_world(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(root_w (N,3), dir_w (N,3)) of peg `idx` (per-env long tensor)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        q = self.rack.data.root_quat_w
        p = self.rack.data.root_pos_w
        root_w = p + quat_apply(q, self._peg_roots[idx])
        dir_w = quat_apply(q, self._peg_dir.unsqueeze(0).expand(n, 3))
        return root_w, dir_w

    def _threaded(self, body, idx: torch.Tensor) -> torch.Tensor:
        """(N,) bool: peg `idx`'s centerline passes through `body`'s handle window —
        computed in the mug frame: the peg line crosses the window plane (y = 0)
        inside the window rectangle (shrunk by `win_shrink`; physics keeps a
        touching peg's centerline peg_w/2 from an edge), with the crossing inside
        the peg's span and the peg roughly perpendicular to the loop plane."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        root_w, dir_w = self._peg_world(idx)
        qm = body.data.root_quat_w
        pm = body.data.root_pos_w
        p = quat_apply_inverse(qm, root_w - pm)
        d = quat_apply_inverse(qm, dir_w)
        dy = d[:, 1]
        ok_dir = dy.abs() > c.thread_dir_min
        safe = torch.where(dy.abs() < 1e-4, torch.full_like(dy, 1e-4), dy)
        t = -p[:, 1] / safe
        qpt = p + t.unsqueeze(-1) * d
        in_span = (t > c.thread_margin) & (t < c.peg_len - c.thread_margin)
        s = c.win_shrink
        in_win = (qpt[:, 0] > c.win_x0 + s) & (qpt[:, 0] < c.win_x1 - s) \
            & (qpt[:, 2] > c.win_z0 + s) & (qpt[:, 2] < c.win_z1 - s)
        return ok_dir & in_span & in_win

    def _aperture_w(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        return body.data.root_pos_w + quat_apply(
            body.data.root_quat_w, self._ap_local.unsqueeze(0).expand(n, 3))

    def _airborne(self, body) -> torch.Tensor:
        z = body.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z > self.cfg.airborne_z

    def _status(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        white = self.mugs[c.mug_names[0]]
        root_w, dir_w = self._peg_world(self.green_idx)
        peg_mid = root_w + dir_w * (c.peg_len / 2)
        d_ap = (self._aperture_w(white) - peg_mid).norm(dim=-1)
        return {
            "threaded": self._threaded(white, self.green_idx),
            "airborne": self._airborne(white),
            "settled": self._settled(white),
            "d_aperture": d_ap,
            "hung_still": self._streak >= c.hang_streak,
        }

    def _update_latches(self) -> None:
        c = self.cfg
        s = self._status()
        self._near |= s["d_aperture"] < c.near_tol
        self._lifted |= s["airborne"] & (s["d_aperture"] < c.lift_near_tol)
        self._threaded_ever |= s["threaded"]
        self._hung_ever |= s["hung_still"]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        s = self._status()
        hang_now = s["threaded"] & s["airborne"] & s["settled"]
        self._streak = torch.where(hang_now, self._streak + 1,
                                   torch.zeros_like(self._streak))
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the WHITE mug hangs on the GREEN peg — peg centerline through
        the handle window within the peg span, mug airborne (the height gate
        excludes every standing support in the scene, asserted in cfg), and the
        hang has been continuously still for `hang_streak` substeps (a live
        suspended equilibrium under gravity and peg contact alone), all states
        finite. Disturbing the mug breaks the streak; success returns only after
        it truly hangs still again."""
        s = self._status()
        self._update_latches()
        c = self.cfg
        pos = torch.stack([b.data.root_pos_w for b in self.mugs.values()], dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return s["threaded"] & s["airborne"] & (self._streak >= c.hang_streak) & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.10 carried-near + 0.10 lifted-near +
        0.20 ever-threaded + 0.20 ever-hung-still, cap 0.60 (0 for doing nothing —
        the mugs start on the floor, far below and away from every peg); exactly
        1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_near * self._near.float()
                + c.w_lift * self._lifted.float()
                + c.w_thread * self._threaded_ever.float()
                + c.w_hang * self._hung_ever.float()).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="mug_rack_hang", robot="null"))
