"""PagodaRelayScene — relocate a three-tier pagoda from its start pedestal into the
walled tray, rebuilt in the same size order, using ONE spare pedestal as the only
buffer and moving ONE tier at a time — a physical Tower-of-Hanoi relay (sim_gen task
`living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_i194`).

Derived from libero_90 `living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_
place_them_in_the_tray` ("stack the left bowl on the right bowl and place them in the
tray"): the seed stacks one bowl on another (an xy/dz readout) and parks the pair
inside a tray's bounding box — two free-space pick-and-place moves in any order the
solver likes, nothing constrains WHICH move is legal WHEN. Here the stack-then-deliver
goal is kept but the move ORDER becomes the task: the three tiers start THREADED on a
pedestal post through their center holes, so only the TOP tier of any pile is ever
extractable (the post makes lower tiers physically captive — you cannot pull the large
tier out from under the mid tier sideways), a tier may only ever rest threaded on one
of the three posts, only one tier may be off-post (in flight) at a time, and a larger
tier must never arrive above a smaller one. With three tiers, one start post, one goal
post (inside the tray) and one buffer post, the shortest legal plan is the classic
7-move Hanoi relay — the small tier alone must be parked and re-picked THREE times.
The rules are enforced continuously by the scene itself (per-substep latches with a
short debounce): any multi-tier carry or size inversion VOIDS the episode — score
pinned to 0 and success impossible until reset.

Strategy vs the corpus (survey of every tasks_v7 card): no existing task constrains
the ORDER of otherwise-easy moves via captivity (threaded stacking posts), and none
has a rules-of-the-game layer enforced by continuous latches — the corpus tasks are
decided by force budgets, mechanisms, pouring, toppling, balance, or geometry, and all
their subgoals can be attempted in any order. Here every individual move is a trivial
pick-drop (the seed's own skill), but the EPISODE is decided by move sequencing under
captivity: the naive strategy (carry the whole stack to the tray in one go, or unload
tiers onto the table) is constructed in smoke and rejected by the rule latches.

success(): all three tiers threaded on the TRAY post, in pagoda order (large at level
0, mid at level 1, small at level 2, each within lvl_tol of its level height),
upright, settled, all states finite, and NO rule violation latched. score(): latched,
non-decreasing stages anchored in the demonstrated 7-move solution — 0.15 the small
tier legally relocated off the start post (seated on buffer or tray post), 0.40 the
large tier seated on the tray post at level 0, 0.70 large + mid seated in the tray in
order, 1.0 iff success(). A violated episode scores 0 regardless of geometry. The
null policy scores ~0 (the pagoda spawns assembled on the start pedestal).

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - pedestal_a (kinematic, RED pad): square pad + slender steel post — the start.
  - pedestal_b (kinematic, BLUE pad): identical — the buffer.
  - tray (kinematic): walled wooden tray with the same post rising from its floor —
    the goal.
  - tiers x3 (dynamic): stepped square pagoda tiers (crimson large / amber mid / teal
    small), each a narrow base pedestal under a wide grasp flange, with a square
    center hole that threads over the round posts; explicit CoM authored.

Per-episode randomization (readback-verified in smoke): global pattern yaw + xy
jitter, per-station xy + yaw jitter, per-tier resting yaw on the start post.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
PAD_S = 0.170  # pedestal pad square side
PAD_H = 0.018  # pedestal pad height == tray floor thickness (level-0 rest height everywhere)
POST_R = 0.011  # post radius (22 mm dia — threads the 34 mm square tier holes)
POST_H = 0.115  # post height above the pad top
POST_TOP = PAD_H + POST_H  # post tip height above the station origin (0.133)
TRAY_IN = 0.230  # tray interior floor square side
TRAY_T = 0.010  # tray wall thickness
TRAY_WH = 0.045  # tray wall height above the floor top
TIER_F = (0.150, 0.120, 0.090)  # flange (grasp plate) outer square side: large, mid, small
TIER_B = (0.110, 0.080, 0.050)  # base pedestal outer square side (narrower: grasp underhang)
HOLE = 0.034  # square center hole side in the FLANGE (the threading gauge)
HOLE_B = 0.042  # larger relief hole in the base pedestal: the base ring rests on the
#                 FLAT of the flange below, clear of its hole edge — two frames with
#                 the same hole put sharp inner edges exactly on top of each other,
#                 a degenerate edge-on-edge contact that rings at ~0.05-0.08 m/s
#                 forever (PhysX edge-contact limit cycle)
BA_H = 0.022  # base pedestal height (finger clearance under the flange)
FL_T = 0.012  # flange plate thickness (pinchable)
TIER_H = BA_H + FL_T  # tier stacking pitch (0.034)
ST_A = (-0.22, -0.04)  # nominal station offsets in the pattern frame: start pedestal,
ST_B = (0.00, 0.20)  # buffer pedestal,
ST_C = (0.24, -0.06)  # tray (goal)
N_TIERS = 3


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


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, material=None) -> None:
    """Author one colliding z-axis cylinder child prim (the posts)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset, material)


def _body_root(stage, prim_path: str, translation, orientation, mass: float,
               lin_damp: float, ang_damp: float, iters: int = 8, vel_iters: int = 1,
               com=None, kinematic: bool = False):
    """Author the rigid-body root xform shared by the compound spawners."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        # stations are the load-bearing ground truth of the rubric: kinematic =
        # infinitely rigid (no micro-vibration under stack impacts), still
        # teleportable per-env at reset
        rb.CreateKinematicEnabledAttr(True)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    if com is not None:
        # MassAPI mass alone leaves the CoM at the body origin — author it explicitly.
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(iters)
    pxrb.CreateSolverVelocityIterationCountAttr(vel_iters)
    return root


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one stacking pedestal as a kinematic compound body: a colored square pad
    with a slender steel post rising from its center. Origin = pad bottom center."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8, kinematic=True)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    steel = (0.72, 0.74, 0.78)
    _box(stage, f"{prim_path}/pad", (PAD_S, PAD_S, PAD_H),
         (0.0, 0.0, PAD_H / 2), cfg.color, 0.0015, material=mat)
    _cyl(stage, f"{prim_path}/post", POST_R, POST_H,
         (0.0, 0.0, PAD_H + POST_H / 2), steel, 0.0012, material=mat)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the goal tray as a kinematic compound body: wooden floor + four walls +
    the same slender post rising from the floor center. Origin = floor bottom center;
    the floor is PAD_H thick so level-0 rest height matches the pedestals."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.8, 0.8, kinematic=True)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    wood = (0.46, 0.31, 0.16)
    rim = (0.55, 0.38, 0.20)
    steel = (0.72, 0.74, 0.78)
    outer = TRAY_IN + 2 * TRAY_T
    _box(stage, f"{prim_path}/floor", (outer, outer, PAD_H),
         (0.0, 0.0, PAD_H / 2), wood, 0.0015, material=mat)
    for tag, sy in (("n", 1.0), ("s", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (outer, TRAY_T, TRAY_WH),
             (0.0, sy * (TRAY_IN / 2 + TRAY_T / 2), PAD_H + TRAY_WH / 2), rim,
             0.0015, material=mat)
    for tag, sx in (("e", 1.0), ("w", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (TRAY_T, TRAY_IN, TRAY_WH),
             (sx * (TRAY_IN / 2 + TRAY_T / 2), 0.0, PAD_H + TRAY_WH / 2), rim,
             0.0015, material=mat)
    _cyl(stage, f"{prim_path}/post", POST_R, POST_H,
         (0.0, 0.0, PAD_H + POST_H / 2), steel, 0.0012, material=mat)
    return root


def _frame(stage, prim_path: str, tag: str, outer: float, hole: float, z0: float,
           h: float, color, material) -> None:
    """Author a square plate `outer` x `outer` x `h` (bottom at z0) with a square
    center hole of side `hole` cut out, as 4 box children (n/s rails + e/w bars)."""
    half = (outer - hole) / 2
    zc = z0 + h / 2
    for t, sy in ((f"{tag}_n", 1.0), (f"{tag}_s", -1.0)):
        _box(stage, f"{prim_path}/{t}", (outer, half, h),
             (0.0, sy * (outer + hole) / 4, zc), color, 0.0012, material=material)
    for t, sx in ((f"{tag}_e", 1.0), (f"{tag}_w", -1.0)):
        _box(stage, f"{prim_path}/{t}", (half, hole, h),
             (sx * (outer + hole) / 4, 0.0, zc), color, 0.0012, material=material)


def _spawn_tier(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one stepped pagoda tier: a narrow base pedestal (side cfg.size_b) under
    a wide grasp flange (side cfg.size_f, thickness FL_T), both pierced by the square
    center hole that threads over the posts. Origin = tier bottom center. The flange
    overhangs the base by (size_f - size_b)/2 per side — the pinchable ring — with
    BA_H of finger clearance beneath it. CoM authored at the true two-plate centroid."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    a_b, a_f = float(cfg.size_b), float(cfg.size_f)
    v_b = (a_b * a_b - HOLE_B * HOLE_B) * BA_H
    v_f = (a_f * a_f - HOLE * HOLE) * FL_T
    z_com = (v_b * (BA_H / 2) + v_f * (BA_H + FL_T / 2)) / (v_b + v_f)
    # vel_iters=4 (kills GPU phantom contact velocity) + moderate damping. The
    # base carries the larger relief hole (HOLE_B) so stacked tiers rest flange-
    # flat on the plate below instead of edge-on-edge (see HOLE_B above).
    root = _body_root(stage, prim_path, translation, orientation,
                      cfg.mass_props.mass, 0.10, 0.80, iters=16, vel_iters=4,
                      com=(0.0, 0.0, z_com))
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    _frame(stage, prim_path, "base", a_b, HOLE_B, 0.0, BA_H, cfg.color, mat)
    _frame(stage, prim_path, "flange", a_f, HOLE, BA_H, FL_T, cfg.color, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (explicit @configclass subclasses
    of RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pedestal" not in _SPAWNER_CACHE:

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            color: tuple = (0.5, 0.5, 0.5)
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class TierSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tier)
            size_f: float = 0.150
            size_b: float = 0.110
            color: tuple = (0.5, 0.5, 0.5)
            mu_s: float = 0.55
            mu_d: float = 0.50

        _SPAWNER_CACHE.update(pedestal=PedestalSpawnerCfg, tray=TraySpawnerCfg,
                              tier=TierSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PagodaRelaySceneCfg(BaseCfg):
    """Config for `PagodaRelayScene`. The honesty knobs are asserted in
    `__post_init__`: the posts thread the holes with real clearance and stand proud of
    the full stack (lower tiers are physically captive — top-only access is physics,
    not fiat), every tier is jaw-graspable by its flange ring, the seated test
    separates a threaded tier from one perched beside the post, the stations cannot
    overlap under randomization, and the tray accepts the largest tier with margin."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_xy_tol: float = tunable(0.020)  # station-local |xy| for "threaded on this post"
    seat_z_lo: float = tunable(0.006)  # seated needs z_bottom >= PAD_H - this
    seat_z_hi: float = tunable(0.010)  # ... and z_bottom <= POST_TOP - this (no post-tip perch)
    tilt_max_deg: float = tunable(12.0)  # seated tier up-axis vs world up
    lvl_tol: float = tunable(0.010)  # |z_bottom - level height| for the pagoda levels
    # settle gates (m/s, rad/s). settle_lin sits ABOVE the ~0.05-0.09 m/s phantom
    # velocity the GPU contact solver reports for stacked tier frames at rest
    # (measured: positions frozen to the mm over 2 s while lin readback holds
    # 0.05-0.09) and far BELOW real motion (drops land at ~1.5 m/s; a tier really
    # moving at 0.12 m/s leaves the 2 cm seat window in ~0.2 s, so the seated +
    # level windows and the persistence hold do the real at-rest work).
    settle_lin: float = tunable(0.12)
    settle_ang: float = tunable(0.8)
    carry_debounce: int = tunable(3)  # substeps of >=2 tiers off-post before violation
    order_debounce: int = tunable(3)  # substeps of bigger-above-smaller before violation

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    global_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the whole pattern (m)
    pattern_yaw_max: float = tunable(30.0)  # uniform +/- yaw of the whole pattern (deg)
    st_jitter: float = tunable(0.018)  # per-station uniform +/- xy jitter (m)
    st_yaw_max: float = tunable(25.0)  # per-station uniform +/- extra yaw (deg)

    # --- info: structure -----------------------------------------------------------------------
    tier_masses: tuple = info((0.45, 0.30, 0.18))  # large, mid, small (kg)
    station_mass: float = info(20.0)
    mu_station_s: float = info(0.60)
    mu_station_d: float = info(0.55)
    mu_tier_s: float = info(0.55)
    mu_tier_d: float = info(0.50)
    post_r: float = info(POST_R)
    post_top: float = info(POST_TOP)
    tier_h: float = info(TIER_H)
    hole: float = info(HOLE)

    def __post_init__(self) -> None:
        # -- threading: the flange hole (the gauge) clears the post with real slack --
        assert HOLE / 2 - POST_R >= 0.005, "hole must clear the post by >= 5 mm per face"
        assert HOLE_B > HOLE, "base relief hole must exceed the flange hole"
        assert (HOLE_B - HOLE) / 2 >= 0.003, \
            "base ring must rest on the FLAT of the flange below (edge-on-edge rings)"
        assert (TIER_B[-1] - HOLE_B) / 2 >= 0.004, "smallest base must keep real hole walls"
        # -- captivity is physics: the post stands proud of the fully-built pagoda --
        assert POST_TOP - (PAD_H + N_TIERS * TIER_H) >= 0.008, \
            "post must stand proud of the full stack (lower tiers stay captive)"
        # -- embodiment: every tier is graspable by its flange ring --
        for f, b in zip(TIER_F, TIER_B):
            assert (f - b) / 2 >= 0.015, "flange must overhang the base >= 15 mm per side"
        assert FL_T <= 0.020, "flange plate must be pinchable by a parallel jaw"
        assert BA_H >= 0.015, "finger clearance under the flange"
        # -- pagoda order is visible: strictly nested sizes --
        for k in range(N_TIERS - 1):
            assert TIER_F[k] - TIER_F[k + 1] >= 0.02 and TIER_B[k] - TIER_B[k + 1] >= 0.02, \
                "tiers must be strictly nested in size"
            assert self.tier_masses[k] > self.tier_masses[k + 1]
        # -- the seated test separates threaded from perched-beside-the-post --
        thread_max = HOLE / 2 * math.sqrt(2.0) - POST_R  # worst threaded center offset
        beside_min = HOLE / 2 + POST_R  # closest an un-threaded tier can rest to the post
        assert thread_max + 0.004 < self.seat_xy_tol < beside_min - 0.006, \
            "seat_xy_tol must separate threaded from beside-the-post"
        # -- no post-tip perch passes the z window; every level is inside it --
        assert PAD_H + (N_TIERS - 1) * TIER_H + self.lvl_tol < POST_TOP - self.seat_z_hi, \
            "the top level must still count as seated"
        # -- the tray accepts the largest tier with real margin; pads hold the bases --
        assert (TRAY_IN - TIER_F[0]) / 2 >= 0.030, "tray interior must clear the large flange"
        assert PAD_S / 2 - TIER_B[0] / 2 >= 0.02, "pad must hold the largest base"
        # -- stations cannot collide under randomization (jitter is per-station) --
        half_tray = TRAY_IN / 2 + TRAY_T
        half_pad = PAD_S / 2
        for p, q, hp, hq in ((ST_A, ST_B, half_pad, half_pad),
                             (ST_A, ST_C, half_pad, half_tray),
                             (ST_B, ST_C, half_pad, half_tray)):
            d = math.hypot(p[0] - q[0], p[1] - q[1])
            assert d >= hp + hq + 2 * self.st_jitter + 0.02, \
                "stations must stay separated under jitter"
        assert self.carry_debounce >= 1 and self.order_debounce >= 1

    # -- derived scalars used by rubric + solve ------------------------------------------------
    @property
    def level_z(self) -> tuple:
        """Nominal z_bottom of pagoda levels 0..2 above a station origin."""
        return tuple(PAD_H + k * TIER_H for k in range(N_TIERS))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pagoda_relay")
class PagodaRelayScene(BaseScene):
    cfg: PagodaRelaySceneCfg

    TIERS = ("tier_l", "tier_m", "tier_s")  # index 0 = largest (Hanoi size order)
    STATIONS = ("pedestal_a", "pedestal_b", "tray")  # index 0 = start, 1 = buffer, 2 = GOAL

    def __init__(self, cfg: PagodaRelaySceneCfg | None = None) -> None:
        super().__init__(cfg or PagodaRelaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        tier_colors = ((0.75, 0.15, 0.12), (0.87, 0.56, 0.10), (0.10, 0.55, 0.50))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "pedestal_a": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PedestalA",
                spawn=spawners["pedestal"](
                    color=(0.72, 0.12, 0.10), mu_s=c.mu_station_s, mu_d=c.mu_station_d,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.station_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ST_A[0], ST_A[1], 0.0)),
            ),
            "pedestal_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PedestalB",
                spawn=spawners["pedestal"](
                    color=(0.15, 0.30, 0.70), mu_s=c.mu_station_s, mu_d=c.mu_station_d,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.station_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ST_B[0], ST_B[1], 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=spawners["tray"](
                    mu_s=c.mu_station_s, mu_d=c.mu_station_d,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.station_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ST_C[0], ST_C[1], 0.0)),
            ),
        }
        for k, name in enumerate(self.TIERS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tier" + "LMS"[k],
                spawn=spawners["tier"](
                    size_f=TIER_F[k], size_b=TIER_B[k], color=tier_colors[k],
                    mu_s=c.mu_tier_s, mu_d=c.mu_tier_d,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tier_masses[k]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ST_A[0], ST_A[1], PAD_H + k * TIER_H + 0.002)),
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
        self.stations: list[RigidObject] = [env.iscene[nm] for nm in self.STATIONS]
        self.tiers: list[RigidObject] = [env.iscene[nm] for nm in self.TIERS]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # rule + stage latch state (zeroed in reset, serialized in get/set_state)
        self._violated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._carry_ctr = torch.zeros(n, device=dev)
        self._order_ctr = torch.zeros(n, device=dev)
        self._s1 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._s3 = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the three stations on a jittered, yawed pattern (each
        with its own extra xy + yaw jitter), thread the assembled pagoda onto the
        START pedestal (random per-tier yaws, small settling gaps), zero the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        center = (torch.rand(m, 2, device=dev) * 2 - 1) * c.global_jitter
        pat_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pattern_yaw_max)
        q_pat = _qz(pat_yaw)
        zcol = torch.zeros(m, 1, device=dev)

        st_pos: list[torch.Tensor] = []
        st_quat: list[torch.Tensor] = []
        for body, nominal in zip(self.stations, (ST_A, ST_B, ST_C)):
            nom = torch.tensor([nominal[0], nominal[1], 0.0], device=dev).expand(m, 3)
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.st_jitter
            pos = torch.cat([center, zcol], dim=-1) + _qapply(q_pat, nom) \
                + torch.cat([jit, zcol], dim=-1)
            yaw = pat_yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.st_yaw_max)
            quat = _qz(yaw)
            write(body, pos, quat)
            st_pos.append(pos)
            st_quat.append(quat)

        # --- pagoda assembled on the START pedestal (A), random per-tier yaw ---
        for k, tier in enumerate(self.tiers):
            lift = torch.tensor([0.0, 0.0, PAD_H + k * TIER_H + 0.0015 * (k + 1)],
                                device=dev).expand(m, 3)
            write(tier, st_pos[0] + _qapply(st_quat[0], lift),
                  _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))

        self._violated[env_ids] = False
        self._carry_ctr[env_ids] = 0.0
        self._order_ctr[env_ids] = 0.0
        for buf in (self._s1, self._s2, self._s3):
            buf[env_ids] = False

    # ----- state (full, restorable — includes rule + stage latches) ---------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        st: dict[str, Any] = {}
        for nm, body in zip(self.STATIONS + self.TIERS, self.stations + self.tiers):
            st[nm] = body.data.root_state_w[env_ids].clone()
        st["_latch"] = torch.stack([
            self._violated[env_ids].float(), self._s1[env_ids].float(),
            self._s2[env_ids].float(), self._s3[env_ids].float(),
            self._carry_ctr[env_ids], self._order_ctr[env_ids]], dim=-1).clone()
        return st

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, body in zip(self.STATIONS + self.TIERS, self.stations + self.tiers):
            body.write_root_state_to_sim(state[nm], env_ids)
        if "_latch" in state:
            lt = state["_latch"]
            self._violated[env_ids] = lt[:, 0] > 0.5
            self._s1[env_ids] = lt[:, 1] > 0.5
            self._s2[env_ids] = lt[:, 2] > 0.5
            self._s3[env_ids] = lt[:, 3] > 0.5
            self._carry_ctr[env_ids] = lt[:, 4]
            self._order_ctr[env_ids] = lt[:, 5]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A stacking-post relay: three stations stand on the table, each with a "
            f"slender steel POST ({2 * POST_R * 1000:.0f} mm dia, tip "
            f"{POST_TOP * 100:.1f} cm up). The RED pad is the start, the BLUE pad is "
            f"a spare, and a walled WOODEN TRAY ({TRAY_IN * 100:.0f} cm square inside) "
            f"with the same post in its center is the destination. On the red post "
            f"sits an assembled three-tier PAGODA, each tier a stepped square plate "
            f"with a {HOLE * 1000:.0f} mm center hole threaded over the post: crimson "
            f"LARGE ({TIER_F[0] * 100:.0f} cm) at the bottom, amber MID "
            f"({TIER_F[1] * 100:.0f} cm), teal SMALL ({TIER_F[2] * 100:.0f} cm) on "
            f"top. Because the post runs through every hole and stands proud of the "
            f"stack, only the TOP tier of any pile can be lifted off — lower tiers "
            f"are captive. Station poses and tier headings change every episode — "
            f"read the scene by looking.\n"
            f"Goal: rebuild the pagoda inside the TRAY, threaded on the tray post in "
            f"the same order (large at the bottom, then mid, then small), all tiers "
            f"upright and at rest. RULES, enforced by the scene: move ONE tier at a "
            f"time — at most one tier may be off a post at any instant, so a tier "
            f"parked on the table (or anywhere off a post) freezes the relay: moving "
            f"any other tier while one is off-post breaks the rule; and NEVER place "
            f"a larger tier above a smaller one on any post. Only a tier threaded on "
            f"a post counts as placed. Breaking a rule voids the episode: the score "
            f"drops to 0 "
            f"and success becomes impossible until the scene is reset. With one spare "
            f"post, the shortest legal plan is the classic 7-move relay "
            f"(tolerances: seated within {c.seat_xy_tol * 100:.1f} cm of a post, "
            f"levels within {c.lvl_tol * 1000:.0f} mm)."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Move the three pagoda tiers one at a time from the red post to the post "
            "inside the wooden tray, using the blue post as a spare, and rebuild the "
            "pagoda in the tray in the same order — never have two tiers off a post "
            "at once, and never put a larger tier on top of a smaller one."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def seat_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (seated, lz): seated (N, tiers, stations) bool — tier threaded on
        that station's post (station-local |xy| within seat_xy_tol, bottom in the
        [pad, below-post-tip] window, upright); lz (N, tiers, stations) station-local
        z of each tier bottom."""
        c = self.cfg
        n, dev = self.env.num_envs, self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        cos_max = math.cos(math.radians(c.tilt_max_deg))
        seated = torch.zeros(n, N_TIERS, len(self.STATIONS), dtype=torch.bool, device=dev)
        lz = torch.zeros(n, N_TIERS, len(self.STATIONS), device=dev)
        for t, tier in enumerate(self.tiers):
            up_ok = _qapply(tier.data.root_quat_w, ez)[:, 2] >= cos_max
            for s, stn in enumerate(self.stations):
                loc = _qapply(_qinv(stn.data.root_quat_w),
                              tier.data.root_pos_w - stn.data.root_pos_w)
                lz[:, t, s] = loc[:, 2]
                seated[:, t, s] = loc[:, 0].abs().le(c.seat_xy_tol) \
                    & loc[:, 1].abs().le(c.seat_xy_tol) \
                    & (loc[:, 2] >= PAD_H - c.seat_z_lo) \
                    & (loc[:, 2] <= POST_TOP - c.seat_z_hi) \
                    & up_ok
        return seated, lz

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    def _finite(self) -> torch.Tensor:
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for body in (*self.stations, *self.tiers):
            ok &= body.data.root_state_w.isfinite().all(dim=-1)
        return ok

    # ----- rule enforcement + stage latches (every physics substep) ---------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Runs once per physics substep: latch rule violations (debounced) and the
        latched score stages. Rules: (1) at most ONE tier off-post at a time — a tier
        is 'off-post' when seated at no station, which also makes the table useless
        as a buffer (a parked tier pins all others); (2) never a bigger tier above a
        smaller one on the same post."""
        c = self.cfg
        seated, lz = self.seat_state()
        off = (~seated.any(dim=-1)).float().sum(dim=-1)  # tiers off any post
        bad_carry = off >= 2.0
        self._carry_ctr = torch.where(bad_carry, self._carry_ctr + 1.0,
                                      torch.zeros_like(self._carry_ctr))
        bad_order = torch.zeros_like(bad_carry)
        for s in range(len(self.STATIONS)):
            for i in range(N_TIERS - 1):  # i bigger than j
                for j in range(i + 1, N_TIERS):
                    bad_order |= seated[:, i, s] & seated[:, j, s] \
                        & (lz[:, i, s] > lz[:, j, s] + TIER_H / 2)
        self._order_ctr = torch.where(bad_order, self._order_ctr + 1.0,
                                      torch.zeros_like(self._order_ctr))
        self._violated |= (self._carry_ctr >= float(c.carry_debounce)) \
            | (self._order_ctr >= float(c.order_debounce))

        ok = ~self._violated
        lv = self.cfg.level_z
        self._s1 |= ok & (seated[:, 2, 1] | seated[:, 2, 2])  # small legally off the start post
        s2_now = seated[:, 0, 2] & (lz[:, 0, 2] - lv[0]).abs().le(c.lvl_tol)
        self._s2 |= ok & s2_now
        self._s3 |= ok & s2_now & seated[:, 1, 2] \
            & (lz[:, 1, 2] - lv[1]).abs().le(c.lvl_tol)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: no rule violation latched, and the pagoda stands rebuilt on the
        TRAY post — large/mid/small threaded at levels 0/1/2 (each within lvl_tol),
        upright (inside the seated test), all tiers settled, all states finite."""
        c = self.cfg
        seated, lz = self.seat_state()
        lv = c.level_z
        ok = ~self._violated
        for t in range(N_TIERS):
            ok &= seated[:, t, 2] & (lz[:, t, 2] - lv[t]).abs().le(c.lvl_tol) \
                & self.settled(self.tiers[t])
        return ok & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched and non-decreasing along the demonstrated
        7-move solution (latches advance in post_step): 0.15 small tier legally
        relocated off the start post; 0.40 large tier seated in the tray at level 0;
        0.70 large + mid rebuilt in the tray in order; 1.0 iff success(). A violated
        episode scores 0 outright. The null policy holds ~0 (the pagoda spawns
        assembled on the start pedestal)."""
        base = 0.15 * self._s1.float() + 0.25 * self._s2.float() + 0.30 * self._s3.float()
        base = torch.where(self._violated, torch.zeros_like(base), base)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="pagoda_relay", robot="null", env_spacing=3.0))
