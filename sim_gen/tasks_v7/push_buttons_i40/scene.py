"""CounterweightScaleScene — counterweigh the balance: count the red reference cubes
in the RED pan, then load the BLUE pan with the unique subset of the three blue
weight blocks whose total mass matches, until the beam physically settles level
(sim_gen task `push_buttons_i40`).

Derived from rlbench/push_buttons, but STRATEGICALLY different: the seed is a
touch-selection task — press each of three colored buttons with a fingertip, in a
prescribed color order; every judged quantity is a momentary BUTTON STATE and the
plan is "aim at colored targets and poke them one by one". Here nothing is pressed
and no touch sequence exists: the task is a MEASUREMENT problem solved through a
MECHANISM. A solver must (1) COUNT the k red reference cubes sitting in the red pan
(k changes per episode), (2) reason that the blue blocks weigh 1, 2 and 4 units
(size-coded, stated in the description) and select the unique subset summing to k,
(3) transport those blocks into the blue pan, and (4) let a two-pan BALANCE BEAM —
a revolute-jointed seesaw with self-leveling hanging pans and a pendulum
counterweight — settle level under gravity. Success is a physical equilibrium of a
jointed mechanism, not a set of poked latches; the plan (count -> subset-sum ->
load one pan) and the code structure (joint-angle rubric over an articulated beam,
mass-conservation clauses) share nothing with the seed.

The scene (fully procedural, no external assets):
  - a light-gray kinematic BENCH slab (1.0 x 0.7 x 0.10 m, top at 0.10 m);
  - a BALANCE SCALE standing on the bench: a heavy dynamic BASE (30 kg foot +
    column; dynamic, not kinematic — a joint anchored to a teleported kinematic
    body0 stays world-fixed at the spawn pose on this stack), an amber BEAM on a
    revolute pivot atop the column (axis = beam-local Y, limits +/- `limit_deg`)
    with a below-pivot bob (the restoring pendulum: level is the only stable
    attitude when the pans carry equal mass), and TWO HANGING PANS on their own
    revolute hangers at the beam tips (+/- `arm_l`): pans swing free about the same
    axis, so they SELF-LEVEL and every load acts through its hanger point — where a
    block sits inside a pan cannot bias the beam, only total pan mass can;
  - the RED pan (red rims) holds k red REFERENCE CUBES (k in 1..7 per episode),
    32 mm, 100 g = 1 unit each;
  - the BLUE pan (blue rims) starts empty;
  - three blue WEIGHT BLOCKS staged on the bench in front of the scale: small
    (32 mm, 1 unit), medium (40 mm, 2 units), large (51 mm, 4 units) — every k in
    1..7 has exactly one subset of {1,2,4} summing to it.

Discrimination is built into the mechanism: with the bob's restoring moment
(M_beam * |com_z| ~= 0.061 kg m) a one-unit (100 g) imbalance rests at
atan(0.1 * arm_l / 0.061) ~= 15 deg — beyond the +/-12 deg joint limits — while a
correct load settles at 0 deg; the `level_tol_deg` = 4 deg gate would need a < 27 g
error to fool, and every wrong subset is >= 100 g off.

success() (all judged when settled, velocities gated):
  A. |beam tilt| <= `level_tol_deg`;
  B. everything still (beam ang vel, pans / blocks lin vel);
  C. ALL k reference cubes still inside the RED pan (rejects "empty both pans and
     let the symmetric beam level itself" and "move references across");
  D. every weight block either inside the BLUE pan or resting farther than
     `exclusion_r` from the scale axis (rejects propping the beam or a pan with a
     block, parking mass on the beam, and compensating inside the red pan).
Given C and D, physics itself enforces that the blue-pan mass equals k units.

score() (latched, anchored in the demonstrated solve trajectory): 0.55 * best
fractional approach of the blue-pan candidate mass toward k units (gated on C — no
credit while the references are disturbed) + 0.35 * ever-success, capped at 0.90;
exactly 1.0 iff success() holds now. Null policy scores 0 (blue pan empty ->
progress 0; beam rests on the red-side limit).

Per-episode randomization (readback-verifiable): k in 1..7, scale xy jitter + yaw
jitter + optional 180 deg flip (which side the red pan faces changes), weight-block
staging-slot permutation + xy jitter + free yaw, reference grid jitter. Absent
reference cubes park in an off-bench ground depot.

No execution order is required (blocks may enter the blue pan in any order).
Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _rot_xy(ang: torch.Tensor, xy: tuple) -> torch.Tensor:
    """(m, 2) world offset of a body-local xy point under yaw `ang`."""
    ca, sa = torch.cos(ang), torch.sin(ang)
    x, y = float(xy[0]), float(xy[1])
    return torch.stack([ca * x - sa * y, sa * x + ca * y], dim=-1)


# ----- custom compound spawners (base / beam / pan) ---------------------------------------------
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 32) -> None:
    """Dynamic rigid-body armor on a compound root: MassAPI mass (PhysX derives the
    inertia and CoM from the child colliders at uniform density), damping, no
    sleeping while velocities are judged, depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_grip(stage_path: str, mat_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material so pan floors and blocks hold
    their loads at every reachable tilt (custom spawner colliders otherwise get the
    ~0.5 default with no restitution control)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(mat_path).IsValid():
        sim_utils.spawn_rigid_body_material(
            mat_path,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, mat_path)


def _spawn_scale_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the scale BASE: heavy DYNAMIC compound (a joint anchored to a
    teleported kinematic body0 stays world-fixed at spawn on this stack — the base
    must be a heavy dynamic fixture). Local origin: footprint centre at bench-top
    level, beam arm along local x."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.base_mass, lin_damp=0.5, ang_damp=0.5, iters=16)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/foot", center=(0.0, 0.0, c.foot_h / 2),
             size=(c.foot_x, c.foot_y, c.foot_h), color=c.base_color, collide=collide)
    _add_box(stage, f"{prim_path}/column",
             center=(0.0, 0.0, c.foot_h + (c.col_top - c.foot_h) / 2),
             size=(c.col_x, c.col_y, c.col_top - c.foot_h),
             color=c.base_color, collide=collide)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the BEAM: dynamic compound whose origin IS the pivot. Children: the
    bar along local x, a small axle hub, and two below-pivot BOB plates flanking
    the column (the pendulum restoring mass — CoM sits ~38 mm below the pivot).
    Plus the REVOLUTE pivot joint to the sibling base (authored at spawn;
    post-play joints are dead). The joint pair is collision-filtered by USD
    default, which is exactly right here (bar and bob overlap the column)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.beam_mass, lin_damp=0.2, ang_damp=c.beam_ang_damp)
    collide = _make_collide(c.contact_offset)
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
             size=(c.bar_len, c.bar_w, c.bar_t), color=c.beam_color, collide=collide)
    _add_box(stage, f"{prim_path}/hub", center=(0.0, 0.0, 0.0),
             size=(0.016, 0.060, 0.016), color=c.hub_color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/bob_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * c.bob_y, -c.bob_depth),
                 size=(c.bob_x, c.bob_t, c.bob_h), color=c.hub_color, collide=collide)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/ScaleBase"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_h)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-float(c.limit_deg))
    j.CreateUpperLimitAttr(float(c.limit_deg))
    return root


def _spawn_pan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one HANGING PAN: dynamic compound whose origin IS its hanger axis on
    the beam tip. A stirrup (crossbar + two side arms) carries a rimmed square tray
    `pan_drop` below the hanger; the CoM sits well below the hanger, so the pan
    self-levels at every beam attitude and its load acts through the hanger point.
    Plus the REVOLUTE hanger joint to the sibling beam (same local axis Y; the
    default joint-pair collision filter absorbs the crossbar/bar overlap)."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    _rigid_dynamic(root, c.pan_mass, lin_damp=0.30, ang_damp=c.pan_ang_damp)
    collide = _make_collide(cfg.contact_offset)
    half = c.pan_w / 2
    floor_top = -c.pan_drop
    _add_box(stage, f"{prim_path}/crossbar", center=(0.0, 0.0, -0.006),
             size=(0.012, 2 * (half + 0.012), 0.012), color=c.rim_color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/arm_{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (half + 0.006), -c.pan_drop / 2),
                 size=(0.030, 0.012, c.pan_drop + 0.008), color=c.rim_color, collide=collide)
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, floor_top - c.floor_t / 2),
             size=(c.pan_w, c.pan_w, c.floor_t), color=c.floor_color, collide=collide)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/rim_x{'p' if sgn > 0 else 'n'}",
                 center=(sgn * (half - c.rim_t / 2), 0.0, floor_top + c.rim_h / 2),
                 size=(c.rim_t, c.pan_w, c.rim_h), color=c.rim_color, collide=collide)
        _add_box(stage, f"{prim_path}/rim_y{'p' if sgn > 0 else 'n'}",
                 center=(0.0, sgn * (half - c.rim_t / 2), floor_top + c.rim_h / 2),
                 size=(c.pan_w - 2 * c.rim_t, c.rim_t, c.rim_h),
                 color=c.rim_color, collide=collide)
    _bind_grip(f"{prim_path}/floor", "/World/simgenScaleGripMat", 0.60, 0.50)

    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hanger")
    j.CreateBody0Rel().SetTargets([f"{base}/ScaleBeam"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hang_x), 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(-60.0)
    j.CreateUpperLimitAttr(60.0)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class ScaleBaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_scale_base)
            base_mass: float = 30.0
            foot_x: float = 0.24
            foot_y: float = 0.18
            foot_h: float = 0.020
            col_x: float = 0.030
            col_y: float = 0.024
            col_top: float = 0.150
            base_color: tuple = (0.30, 0.30, 0.34)
            contact_offset: float = 0.002

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            beam_mass: float = 1.6
            beam_ang_damp: float = 4.0
            bar_len: float = 0.37
            bar_w: float = 0.024
            bar_t: float = 0.016
            bob_y: float = 0.030
            bob_x: float = 0.050
            bob_t: float = 0.014
            bob_h: float = 0.110
            bob_depth: float = 0.077
            pivot_h: float = 0.16
            limit_deg: float = 12.0
            beam_color: tuple = (0.85, 0.65, 0.20)
            hub_color: tuple = (0.20, 0.20, 0.24)
            contact_offset: float = 0.002

        @configclass
        class PanSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pan)
            pan_mass: float = 0.15
            pan_ang_damp: float = 2.0
            pan_w: float = 0.130
            pan_drop: float = 0.100
            floor_t: float = 0.008
            rim_t: float = 0.008
            rim_h: float = 0.032
            hang_x: float = 0.16
            rim_color: tuple = (0.8, 0.2, 0.15)
            floor_color: tuple = (0.9, 0.55, 0.5)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["base"] = ScaleBaseSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["pan"] = PanSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CounterweightScaleSceneCfg(BaseCfg):
    """Config for `CounterweightScaleScene`. Honesty knobs asserted in
    `__post_init__`: a one-unit imbalance rests beyond the joint limits while the
    level tolerance would need a < 1/3-unit error to fool; the staging slots lie
    outside the exclusion radius, which itself covers the scale's whole reach (a
    propping block can never satisfy clause D); every block fits its pan."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    level_tol_deg: float = tunable(4.0)    # |beam tilt| below this counts as level
    beam_still_avel: float = tunable(0.12)  # max beam |ang vel| when judging (rad/s)
    settle_speed: float = tunable(0.06)    # max |lin vel| (pans and blocks) when judging (m/s)
    still_steps: int = tunable(60)         # consecutive sub-threshold steps (0.5 s) for "at rest"
    exclusion_r: float = tunable(0.28)     # unused blocks must rest farther than this
    pan_xy_tol: float = tunable(0.052)     # in-pan: |x|,|y| below this in the pan frame
    pan_z_lo: float = tunable(-0.106)      # in-pan height band, pan frame (floor top -0.100)
    pan_z_hi: float = tunable(0.030)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    k_min: int = tunable(1)                # reference-cube count k ~ U{k_min..k_max}
    k_max: int = tunable(7)
    scale_jitter: float = tunable(0.03)    # scale xy jitter (+/- m)
    scale_yaw_deg: float = tunable(15.0)   # scale yaw jitter (+/- deg)
    scale_flip: bool = tunable(True)       # random 180 deg flip (red pan side swaps)
    slot_shuffle: bool = tunable(True)     # permute which staging slot holds which block
    block_jitter: float = tunable(0.02)    # weight-block xy jitter (+/- m)
    block_yaw_deg: float = tunable(180.0)  # weight-block free yaw (+/- deg)
    ref_jitter: float = tunable(0.0015)    # reference-cube grid jitter (+/- m)

    # --- tunable: placement (bench frame) ----------------------------------------------------
    scale_pos: tuple = tunable((0.0, 0.15))    # scale axis on the bench
    slot_y: float = tunable(-0.22)             # staging row
    slot_xs: tuple = tunable((-0.16, 0.0, 0.16))

    # --- info: bench -------------------------------------------------------------------------
    bench_size: tuple = info((1.0, 0.70, 0.10))
    bench_top: float = info(0.10)
    bench_color: tuple = info((0.75, 0.75, 0.78))
    # --- info: scale structure (see spawners; duplicated here for the rubric) ----------------
    arm_l: float = info(0.16)          # hanger distance from the pivot
    pivot_h: float = info(0.16)        # pivot height above the bench top
    limit_deg: float = info(12.0)      # beam joint limits
    beam_mass: float = info(1.6)
    pan_w: float = info(0.130)
    pan_drop: float = info(0.100)      # hanger to pan-floor top
    rim_h: float = info(0.032)
    rim_t: float = info(0.008)
    base_foot: tuple = info((0.24, 0.18))
    contact_offset: float = info(0.002)
    # --- info: blocks ------------------------------------------------------------------------
    unit_mass: float = info(0.100)     # one unit = 100 g
    ref_size: float = info(0.032)
    n_ref_slots: int = info(7)
    # (name, size, units): the three candidate weights — every k in 1..7 is a
    # unique subset sum of {1, 2, 4}
    cand_specs: tuple = info((("small", 0.032, 1), ("medium", 0.040, 2), ("large", 0.051, 4)))
    ref_color: tuple = info((0.85, 0.12, 0.10))
    cand_colors: tuple = info(((0.25, 0.45, 0.90), (0.20, 0.38, 0.80), (0.15, 0.30, 0.70)))
    red_rim: tuple = info((0.80, 0.20, 0.15))
    red_floor: tuple = info((0.90, 0.55, 0.50))
    blue_rim: tuple = info((0.20, 0.35, 0.80))
    blue_floor: tuple = info((0.55, 0.65, 0.92))
    # reference grid inside the red pan (pan-local xy slots, single layer)
    ref_grid_pitch: float = info(0.0365)
    # off-bench ground depot for absent reference cubes
    depot_pos: tuple = info((1.0, 1.0))
    # rubric weights (0.55 + 0.35 = 0.90 = the non-success cap)
    w_prog: float = info(0.55)
    w_level: float = info(0.35)

    # Derived (filled in __post_init__).
    com_depth: float = field(default=None, init=False)  # beam CoM depth below the pivot

    def __post_init__(self) -> None:
        # beam CoM depth from the compound collider volumes (uniform density):
        v_bar = 0.37 * 0.024 * 0.016
        v_hub = 0.016 * 0.060 * 0.016
        v_bob = 2 * 0.050 * 0.014 * 0.110
        self.com_depth = (v_bob * 0.077) / (v_bar + v_hub + v_bob)
        restoring = self.beam_mass * self.com_depth  # kg m
        # a one-unit imbalance must rest BEYOND the joint limit ...
        tilt_1u = math.degrees(math.atan(self.unit_mass * self.arm_l / restoring))
        assert tilt_1u > self.limit_deg + 2.0, (
            f"one-unit imbalance rests at {tilt_1u:.1f} deg — must exceed the "
            f"{self.limit_deg} deg limit")
        # ... while the level gate needs a < 1/3-unit error to fool
        fool = restoring * math.tan(math.radians(self.level_tol_deg)) / self.arm_l
        assert fool < self.unit_mass / 3.0, (
            f"level_tol admits a {fool * 1000:.0f} g error — must be well under one unit")
        # staging slots lie outside the exclusion radius (null layout is clause-D clean)
        sx, sy = self.scale_pos
        for x in self.slot_xs:
            d = math.hypot(x - sx, self.slot_y - sy) - self.block_jitter - self.scale_jitter
            assert d > self.exclusion_r + 0.03, "staging slots must clear the exclusion radius"
        # the exclusion radius covers the scale's whole reach (props can't hide outside it)
        reach = self.arm_l + self.pan_w / 2 + 0.02
        assert self.exclusion_r > reach + 0.03, "exclusion radius must cover the scale"
        # every block fits through the pan opening; the reference grid fits the pan
        interior = self.pan_w - 2 * self.rim_t
        assert max(s for _n, s, _u in self.cand_specs) < interior - 0.04
        assert 2 * self.ref_grid_pitch + self.ref_size < interior
        # the pan clears the bench at full beam tilt
        low = self.pivot_h - self.arm_l * math.sin(math.radians(self.limit_deg)) \
            - self.pan_drop - 0.008
        assert low > 0.015, "pan must clear the bench at the joint limit"
        assert self.k_min >= 1 and self.k_max <= 7


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("counterweight_scale")
class CounterweightScaleScene(BaseScene):
    cfg: CounterweightScaleSceneCfg

    def __init__(self, cfg: CounterweightScaleSceneCfg | None = None) -> None:
        super().__init__(cfg or CounterweightScaleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.bench_top
        sp = _spawner_classes()
        sx, sy = c.scale_pos
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=0.60, dynamic_friction=0.50,
                                             restitution=0.0)
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
            "bench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=c.bench_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.bench_size[2] / 2)),
            ),
            # spawn order matters: each joint's body0 must already exist
            "scale_base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ScaleBase",
                spawn=sp["base"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, z0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/ScaleBeam",
                spawn=sp["beam"](pivot_h=c.pivot_h, limit_deg=c.limit_deg,
                                 beam_mass=c.beam_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, z0 + c.pivot_h)),
            ),
            "pan_red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PanRed",
                spawn=sp["pan"](hang_x=-c.arm_l, pan_w=c.pan_w, pan_drop=c.pan_drop,
                                rim_h=c.rim_h, rim_t=c.rim_t, rim_color=c.red_rim,
                                floor_color=c.red_floor, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx - c.arm_l, sy, z0 + c.pivot_h)),
            ),
            "pan_blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PanBlue",
                spawn=sp["pan"](hang_x=c.arm_l, pan_w=c.pan_w, pan_drop=c.pan_drop,
                                rim_h=c.rim_h, rim_t=c.rim_t, rim_color=c.blue_rim,
                                floor_color=c.blue_floor, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx + c.arm_l, sy, z0 + c.pivot_h)),
            ),
        }
        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.08, angular_damping=0.20,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1)
        col = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        for i in range(c.n_ref_slots):
            out[f"ref_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ref_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(c.ref_size,) * 3, rigid_props=rigid, collision_props=col,
                    physics_material=mat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.unit_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ref_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.12 * (i % 3), c.depot_pos[1] + 0.12 * (i // 3),
                         c.ref_size / 2 + 0.003)),
            )
        for (name, size, units), color, x in zip(c.cand_specs, c.cand_colors, c.slot_xs):
            out[f"cand_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cand_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(size,) * 3, rigid_props=rigid, collision_props=col,
                    physics_material=mat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=units * c.unit_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x, c.slot_y, z0 + size / 2 + 0.003)),
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
        c = self.cfg
        self.base: RigidObject = env.iscene["scale_base"]
        self.beam: RigidObject = env.iscene["beam"]
        self.pan_red: RigidObject = env.iscene["pan_red"]
        self.pan_blue: RigidObject = env.iscene["pan_blue"]
        self.refs: list[RigidObject] = [env.iscene[f"ref_{i}"] for i in range(c.n_ref_slots)]
        self.cands: list[RigidObject] = [env.iscene[f"cand_{nm}"] for nm, _s, _u in c.cand_specs]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.n_ref = torch.full((n,), c.n_ref_slots, dtype=torch.long, device=dev)
        self.ref_present = torch.ones(n, c.n_ref_slots, dtype=torch.bool, device=dev)
        self.cand_units = torch.tensor([float(u) for _n, _s, u in c.cand_specs], device=dev)
        self.prog_latch = torch.zeros(n, device=dev)
        self.level_latch = torch.zeros(n, device=dev)
        self.still_count = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample k, place the scale (base + beam level + pans
        hanging) at a jittered/yawed/possibly-flipped pose, seat k reference cubes
        in a jittered grid on the red pan floor, park the rest in the ground depot,
        shuffle the weight blocks over the staging slots; latches zeroed. The beam
        is written LEVEL and swings to the red-side limit on its own weight."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.bench_top

        # --- k and the reference presence mask ---
        k = torch.randint(c.k_min, c.k_max + 1, (m,), device=dev)
        self.n_ref[env_ids] = k
        pres = torch.arange(c.n_ref_slots, device=dev).unsqueeze(0) < k.unsqueeze(1)
        self.ref_present[env_ids] = pres

        # --- scale pose ---
        sxy = torch.tensor(c.scale_pos, device=dev).unsqueeze(0).expand(m, 2).clone()
        sxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scale_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.scale_yaw_deg)
        if c.scale_flip:
            yaw = yaw + math.pi * (torch.rand(m, device=dev) < 0.5).float()
        q = _qz(yaw)

        def write(body, xy: torch.Tensor, z: torch.Tensor | float,
                  quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.base, sxy, z0, q)
        write(self.beam, sxy, z0 + c.pivot_h, q)
        write(self.pan_red, sxy + _rot_xy(yaw, (-c.arm_l, 0.0)), z0 + c.pivot_h, q)
        write(self.pan_blue, sxy + _rot_xy(yaw, (c.arm_l, 0.0)), z0 + c.pivot_h, q)

        # --- reference cubes: grid slots on the red pan floor; absent -> depot ---
        red_xy = sxy + _rot_xy(yaw, (-c.arm_l, 0.0))
        floor_z = z0 + c.pivot_h - c.pan_drop
        p = c.ref_grid_pitch
        grid = [(-p, -p), (0.0, -p), (p, -p), (-p, 0.0), (0.0, 0.0), (p, 0.0), (0.0, p)]
        for i, body in enumerate(self.refs):
            gx, gy = grid[i]
            local = torch.empty(m, 2, device=dev)
            local[:, 0], local[:, 1] = gx, gy
            local += (torch.rand(m, 2, device=dev) * 2 - 1) * c.ref_jitter
            ca, sa = torch.cos(yaw), torch.sin(yaw)
            in_pan = red_xy + torch.stack(
                [ca * local[:, 0] - sa * local[:, 1],
                 sa * local[:, 0] + ca * local[:, 1]], dim=-1)
            depot = torch.tensor(
                [c.depot_pos[0] + 0.12 * (i % 3), c.depot_pos[1] + 0.12 * (i // 3)],
                device=dev).unsqueeze(0).expand(m, 2)
            here = pres[:, i].unsqueeze(1)
            xy = torch.where(here, in_pan, depot)
            z = torch.where(pres[:, i],
                            torch.full((m,), floor_z + c.ref_size / 2 + 0.003, device=dev),
                            torch.full((m,), c.ref_size / 2 + 0.003, device=dev))
            write(body, xy, z, torch.where(here, q, _qz(torch.zeros(m, device=dev))))

        # --- weight blocks: shuffled staging slots + jitter + free yaw ---
        perm = torch.rand(m, 3, device=dev).argsort(dim=1) if c.slot_shuffle else \
            torch.arange(3, device=dev).unsqueeze(0).expand(m, 3)
        slots = torch.tensor([[x, c.slot_y] for x in c.slot_xs], device=dev)
        for j, body in enumerate(self.cands):
            size = c.cand_specs[j][1]
            xy = slots[perm[:, j]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.block_jitter
            byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg)
            write(body, xy, z0 + size / 2 + 0.003, _qz(byaw))

        self.prog_latch[env_ids] = 0.0
        self.level_latch[env_ids] = 0.0
        self.still_count[env_ids] = 0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "pan_red": self.pan_red.data.root_state_w[env_ids].clone(),
            "pan_blue": self.pan_blue.data.root_state_w[env_ids].clone(),
            "refs": [b.data.root_state_w[env_ids].clone() for b in self.refs],
            "cands": [b.data.root_state_w[env_ids].clone() for b in self.cands],
            "n_ref": self.n_ref[env_ids].clone(),
            "ref_present": self.ref_present[env_ids].clone(),
            "prog_latch": self.prog_latch[env_ids].clone(),
            "level_latch": self.level_latch[env_ids].clone(),
            "still_count": self.still_count[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.pan_red.write_root_state_to_sim(state["pan_red"], env_ids)
        self.pan_blue.write_root_state_to_sim(state["pan_blue"], env_ids)
        for b, st in zip(self.refs, state["refs"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.cands, state["cands"]):
            b.write_root_state_to_sim(st, env_ids)
        self.n_ref[env_ids] = state["n_ref"]
        self.ref_present[env_ids] = state["ref_present"]
        self.prog_latch[env_ids] = state["prog_latch"]
        self.level_latch[env_ids] = state["level_latch"]
        self.still_count[env_ids] = state["still_count"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A two-pan BALANCE SCALE stands on a light-gray bench: a dark base column "
            f"carries an amber BEAM on a central pivot, and from each beam tip hangs a "
            f"square open pan ({c.pan_w * 1000:.0f} mm wide, low rims) that always hangs "
            f"level. One pan has RED rims, the other BLUE rims; which side each faces "
            f"changes between episodes. The beam can tip about "
            f"{c.limit_deg:.0f} degrees each way and currently rests tipped toward the red "
            f"pan, because the RED pan holds between 1 and 7 identical RED REFERENCE CUBES "
            f"({c.ref_size * 1000:.0f} mm, one weight unit each) — COUNT them by sight. The "
            f"BLUE pan is empty. On the bench in front of the scale lie three BLUE WEIGHT "
            f"BLOCKS in shuffled positions, told apart by size: the small block "
            f"({c.cand_specs[0][1] * 1000:.0f} mm) weighs 1 unit — the same as one red cube "
            f"— the medium block ({c.cand_specs[1][1] * 1000:.0f} mm) weighs 2 units, and "
            f"the large block ({c.cand_specs[2][1] * 1000:.0f} mm) weighs 4 units. Exactly "
            f"one combination of blue blocks matches the red cubes' total weight.\n"
            f"Goal: counterbalance the scale. Place blue weight blocks INTO the BLUE pan "
            f"until both pans carry equal weight and the beam settles LEVEL (within "
            f"{c.level_tol_deg:.0f} degrees), in any order. Do not touch or move the red "
            f"cubes — all of them must remain inside the red pan. Blue blocks you do not "
            f"use must be left resting on the bench well away from the scale (at least "
            f"{c.exclusion_r * 100:.0f} cm from the base column); a block leaning on the "
            f"scale, wedged under a pan or beam, or dropped into the red pan voids the "
            f"result. The scale is judged only when everything has come to rest: a beam "
            f"merely swinging through level does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Count the red cubes in the red pan of the balance scale. Blue blocks weigh 1, "
            "2 and 4 units by size; place blue blocks totaling exactly that count into the "
            "blue pan so the beam settles level. Do not disturb the red cubes, and leave "
            "unused blue blocks on the bench well away from the scale."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def tilt(self) -> torch.Tensor:
        """(N,) beam tilt in radians; positive = BLUE side down."""
        from isaaclab.utils.math import quat_apply

        q = self.beam.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        # blue hangs at beam-local +arm_l: tilt>0 when that end points down
        v = quat_apply(q, ex)
        return torch.atan2(-v[:, 2], v[:, :2].norm(dim=-1))

    def _blocks(self, bodies: list) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.stack([b.data.root_pos_w for b in bodies], dim=1)  # (N, B, 3)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in bodies], dim=1)
        return pos, vel

    def _in_pan(self, pan: RigidObject, pos: torch.Tensor) -> torch.Tensor:
        """(N, B) bool: block centres inside the pan interior, judged in the PAN'S
        BODY FRAME (the pan self-levels, so this is the honest containment test)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, b = pos.shape[0], pos.shape[1]
        pq = pan.data.root_quat_w[:, None, :].expand(n, b, 4).reshape(n * b, 4)
        pp = pan.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(pq, (pos - pp).reshape(n * b, 3)).reshape(n, b, 3)
        in_xy = (loc[:, :, 0].abs() < c.pan_xy_tol) & (loc[:, :, 1].abs() < c.pan_xy_tol)
        in_z = (loc[:, :, 2] > c.pan_z_lo) & (loc[:, :, 2] < c.pan_z_hi)
        return in_xy & in_z

    # ----- predicates -------------------------------------------------------------------------
    def level(self) -> torch.Tensor:
        """(N,) bool: |beam tilt| within `level_tol_deg`."""
        return self.tilt().abs() <= math.radians(self.cfg.level_tol_deg)

    def _still_now(self) -> torch.Tensor:
        """(N,) bool: beam ang vel, pan lin vel and every present block lin vel below
        threshold at THIS instant (also true for a moment at a swing's turning point)."""
        c = self.cfg
        ok = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_still_avel
        for pan in (self.pan_red, self.pan_blue):
            ok = ok & (pan.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        _rp, rv = self._blocks(self.refs)
        ok = ok & ((rv < c.settle_speed) | ~self.ref_present).all(dim=1)
        _cp, cv = self._blocks(self.cands)
        return ok & (cv < c.settle_speed).all(dim=1)

    def still(self) -> torch.Tensor:
        """(N,) bool: SUSTAINED rest — velocities sub-threshold for `still_steps`
        consecutive sim steps (counter kept by `post_step`). A beam swinging through
        its turning point dips below the velocity threshold for a frame or two; this
        gate is what makes 'judged only at rest' real."""
        return self.still_count >= self.cfg.still_steps

    def refs_home(self) -> torch.Tensor:
        """(N,) bool: every PRESENT reference cube inside the RED pan (clause C)."""
        pos, _v = self._blocks(self.refs)
        return (self._in_pan(self.pan_red, pos) | ~self.ref_present).all(dim=1)

    def cand_in_blue(self) -> torch.Tensor:
        """(N, 3) bool: candidate block inside the BLUE pan."""
        pos, _v = self._blocks(self.cands)
        return self._in_pan(self.pan_blue, pos)

    def cand_ok(self) -> torch.Tensor:
        """(N,) bool: every candidate either inside the BLUE pan or resting farther
        than `exclusion_r` from the scale axis (clause D — the anti-prop /
        anti-red-pan / anti-park-on-the-beam clause)."""
        pos, _v = self._blocks(self.cands)
        axis = self.base.data.root_pos_w[:, None, :2]
        far = (pos[:, :, :2] - axis).norm(dim=-1) > self.cfg.exclusion_r
        return (self.cand_in_blue() | far).all(dim=1)

    def blue_units(self) -> torch.Tensor:
        """(N,) candidate weight units currently inside the blue pan."""
        return (self.cand_in_blue().float() * self.cand_units.unsqueeze(0)).sum(dim=1)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Per-step: advance the sustained-stillness counter, then latch fractional
        load progress toward k units (gated on the references being undisturbed) and
        ever-success. Latched credit does not evaporate when a block is lifted back
        out or the beam swings."""
        now = self._still_now()
        self.still_count = torch.where(
            now, self.still_count + 1, torch.zeros_like(self.still_count))
        k = self.n_ref.float().clamp(min=1.0)
        prog = (1.0 - (k - self.blue_units()).abs() / k).clamp(0.0, 1.0)
        prog = prog * self.refs_home().float()
        self.prog_latch = torch.maximum(self.prog_latch, prog)
        self.level_latch = torch.maximum(self.level_latch, self.success().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beam level AND everything still AND all references in the red
        pan AND every candidate in the blue pan or beyond the exclusion radius.
        Physics then guarantees the blue load equals k units."""
        return self.level() & self.still() & self.refs_home() & self.cand_ok()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.55 * latched load progress + 0.35 * ever-success,
        capped at 0.90; exactly 1.0 iff success() now. Null policy: blue pan empty
        -> progress 0, beam on the red-side limit -> 0."""
        c = self.cfg
        base = (c.w_prog * self.prog_latch + c.w_level * self.level_latch).clamp(0.0, 0.90)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="counterweight_scale", robot="null"))
