"""GlazingBenchScene — clear the offcut pebbles out of the recessed seat, drop the
teal pane flush into it, then slide the keeper bar home under the roofed housing
until its tongue overlaps the pane edge (sim_gen task `track_banana_i425`).

Derived from pick_place/track_banana, but STRATEGICALLY different: the seed starts
with a banana already rigidly grasped and scores free-space trajectory tracking
along five floating waypoints — path-following of a held object, no contact events,
no scene state changes. Here NOTHING is pre-grasped and the path taken matters not
at all: the rubric reads three CONTACT-MADE scene states in a physically forced
order. (1) The 120x120 mm seat recess is fouled by 2-4 loose 18 mm offcut cubes; a
pane dropped on them rests PROUD (propped ~5+ mm above the flush band) so the seat
must be cleared first. (2) The 110 mm teal pane must free-fall the last stretch and
settle FLUSH (top 2 mm below the bench top) inside the recess. (3) A keeper bar
must be placed on the apron and SLID ~80 mm under a roofed housing whose 12 mm slot
admits only its 10 mm tongue (the 30 mm head jams against the housing face = hard
stop) until the tongue overhangs the seated pane's edge. Locking first blocks the
pane: with the tongue overhanging the recess by 15 mm, the open span (105 mm) is
smaller than the pane (110 mm), so a flat drop cannot seat — pane before keeper.

Strategy vs the corpus: this is not a waypoint task (seed), not fill-a-container
(pen_holder exemplar: count-in-cup), not push-a-tool-through-a-tunnel (i240
ram-feed), not clamp-release-and-catch (i79), and not an armed cascade poked into a
hand-proof guard (poke_cube_i83). The signature move is a SUBTRACTIVE prep (remove
foulers so a drop can seat) chained into a PRISMATIC lock whose admission is keyed
by part thickness, with the order forced purely by interference geometry — no
latch, no stored-energy chain, no counting.

success(): every PRESENT pebble is out of the seat recess, the pane is seated
flush (bench-frame |xy| <= 10 mm, center z in [33, 41] mm, up_z >= 0.995, settled),
and the keeper is locked (tongue tip bench-x <= 52 mm => >= 8 mm overlap over the
recess edge at 60 mm, |y| <= 15 mm, tongue riding the bench top, +x axis aligned
with the bench +x, settled). score(): 0.15 * pebbles-out fraction + 0.35 * seated,
capped at 0.50; 1.0 iff success(). Null policy ~0 (pebbles foul the seat, pane and
keeper lie on the ground).

Assets are fully procedural:
  - bench (KINEMATIC): 45 mm slab with a 120x120x14 mm recess (floor z 31 mm), a
    3-sided 12 mm curb rim (open toward +x), and on the +x apron a roofed housing:
    two pillars + roof plate spanning the slide channel (clear width 110 mm, clear
    height 12 mm above the bench top).
  - pane (teal, 110x110x12 mm, 24 mm grasp knob on top, 250 g) and an oversize
    DECOY pane (gray, 155x155x12 mm + knob, 450 g) lying flat on the ground at two
    shuffled slots — the decoy cannot enter the 120 mm recess at all.
  - keeper bar (tan, tongue 80x104x10 mm + head 32x104x30 mm, flat underside,
    200 g) lying on the ground.
  - offcut pebbles (red 18 mm cubes, 15 g, 2-4 present per episode) scattered on
    the recess floor.

Per-episode randomization (readback-verified in smoke): bench anchor jitters in xy
and yaws +/-20 deg (all rubric geometry is bench-frame), pebble count 2-4 with
shuffled recess slots + jitter (absent pebbles parked far off-stage), and the
pane/decoy ground slots are shuffled with free yaw + jitter. Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- geometry constants (single source of truth: spawner + cfg asserts + rubric) -------------
_TOP = 0.045  # bench top z
_BAY_HW = 0.060  # recess inner half-extent (120 x 120 opening)
_BAY_FLOOR = 0.031  # recess floor z (depth 14 mm)
_BX0, _BX1 = -0.120, 0.260  # bench slab x span (recess centered at x = 0)
_BHY = 0.145  # bench slab y half-extent
_CURB_H = 0.012  # curb rim height above the bench top (3 sides, open toward +x)

_PANE_S, _PANE_T = 0.110, 0.012  # pane plate
_KNOB = 0.024  # grasp knob cube on the pane top
_DECOY_S = 0.155  # decoy plate side (cannot enter the recess)
_PEB = 0.018  # offcut pebble cube side

_TON_L, _TON_W, _TON_T = 0.080, 0.104, 0.010  # keeper tongue (body origin = its center)
_HEAD_L, _HEAD_W, _HEAD_H = 0.032, 0.104, 0.030  # keeper head (+x end, bottom flush)
_HEAD_CX = _TON_L / 2 + _HEAD_L / 2  # 0.056 head center, body frame
_HEAD_CZ = -_TON_T / 2 + _HEAD_H / 2  # 0.010
_TIP_OFF = _TON_L / 2  # tongue tip = body origin - 0.040 along +x

_GAP = 0.012  # housing clear height above the bench top (admits tongue, not head)
_HOUSE_X0, _HOUSE_X1 = 0.095, 0.125  # housing x span (pillar faces; -x face = hard stop ref)
_CHAN_HW = 0.055  # slide channel clear half-width (110 mm; tongue 104 mm)
_PIL_W = 0.030  # pillar y width
_ROOF_T = 0.008

_LOCK_X = 0.085  # keeper body-origin bench-x when the head touches the housing face
_TIP_LOCK = _LOCK_X - _TIP_OFF  # 0.045: tongue tip overhangs the recess edge by 15 mm
_DROP_X = 0.175  # apron drop-zone keeper origin (tongue fully east of the housing)

_PEB_SLOTS = ((0.028, 0.028), (-0.028, -0.028), (0.028, -0.028), (-0.028, 0.028))
_GSLOTS = ((-0.26, 0.33), (-0.26, -0.33))  # pane / decoy ground slots (shuffled)
_BAR_SLOT = (-0.34, 0.0)  # keeper ground slot
_PARK = (1.30, 1.30)  # absent-pebble park (bench frame, on the ground, off-stage)


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


def _box(stage, path: str, size, center, color, material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(0.001)
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_bench(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC bench: recessed seat, 3-sided curb rim, roofed housing.
    Origin = recess center on the ground; +x runs toward the housing/apron."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    top_m = _friction_material(stage, f"{prim_path}/top_mat", cfg.mu_bench_s, cfg.mu_bench_d)
    wood = (0.55, 0.42, 0.28)
    dark = (0.36, 0.28, 0.20)
    curb = (0.20, 0.20, 0.24)

    # --- recess floor plate (ground .. _BAY_FLOOR under the 120 x 120 opening) ---
    _box(stage, f"{prim_path}/bay_floor", (2 * _BAY_HW, 2 * _BAY_HW, _BAY_FLOOR),
         (0.0, 0.0, _BAY_FLOOR / 2), dark, material=top_m)
    # --- slab strips around the recess, full height (ground .. _TOP) ---
    _box(stage, f"{prim_path}/slab_w", (-_BX0 - _BAY_HW, 2 * _BHY, _TOP),
         ((_BX0 - _BAY_HW) / 2, 0.0, _TOP / 2), wood, material=top_m)
    _box(stage, f"{prim_path}/slab_e", (_BX1 - _BAY_HW, 2 * _BHY, _TOP),
         ((_BX1 + _BAY_HW) / 2, 0.0, _TOP / 2), wood, material=top_m)
    _box(stage, f"{prim_path}/slab_n", (2 * _BAY_HW, _BHY - _BAY_HW, _TOP),
         (0.0, (_BHY + _BAY_HW) / 2, _TOP / 2), wood, material=top_m)
    _box(stage, f"{prim_path}/slab_s", (2 * _BAY_HW, _BHY - _BAY_HW, _TOP),
         (0.0, -(_BHY + _BAY_HW) / 2, _TOP / 2), wood, material=top_m)
    # --- 3-sided curb rim above the bench top (W/N/S; open toward +x for the tongue) ---
    cz = _TOP + _CURB_H / 2
    _box(stage, f"{prim_path}/curb_w", (0.010, 2 * _BAY_HW + 0.020, _CURB_H),
         (-_BAY_HW - 0.005, 0.0, cz), curb, material=top_m)
    _box(stage, f"{prim_path}/curb_n", (2 * _BAY_HW, 0.010, _CURB_H),
         (0.0, _BAY_HW + 0.005, cz), curb, material=top_m)
    _box(stage, f"{prim_path}/curb_s", (2 * _BAY_HW, 0.010, _CURB_H),
         (0.0, -_BAY_HW - 0.005, cz), curb, material=top_m)
    # --- roofed housing on the apron: two pillars + roof plate over the channel ---
    hx = (_HOUSE_X0 + _HOUSE_X1) / 2
    hl = _HOUSE_X1 - _HOUSE_X0
    for sgn, tag in ((1.0, "n"), (-1.0, "s")):
        _box(stage, f"{prim_path}/pillar_{tag}", (hl, _PIL_W, _GAP),
             (hx, sgn * (_CHAN_HW + _PIL_W / 2), _TOP + _GAP / 2), curb, material=top_m)
    _box(stage, f"{prim_path}/roof", (hl, 2 * (_CHAN_HW + _PIL_W), _ROOF_T),
         (hx, 0.0, _TOP + _GAP + _ROOF_T / 2), curb, material=top_m)
    return root


def _spawn_keeper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC keeper bar: thin tongue + tall head, flat underside.
    Body origin = tongue center. MassAPI + material authored here (custom spawn
    funcs ignore the RigidObjectCfg mass/material fields)."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(8)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    tan = (0.78, 0.62, 0.34)
    _box(stage, f"{prim_path}/tongue", (_TON_L, _TON_W, _TON_T), (0.0, 0.0, 0.0),
         tan, material=mat)
    _box(stage, f"{prim_path}/head", (_HEAD_L, _HEAD_W, _HEAD_H),
         (_HEAD_CX, 0.0, _HEAD_CZ), (0.62, 0.45, 0.20), material=mat)
    return root


def _spawn_pane(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC pane: square plate + grasp knob on top. Body origin =
    plate center. MassAPI + material authored here."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(8)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_s, cfg.mu_d)
    s = float(cfg.side)
    _box(stage, f"{prim_path}/plate", (s, s, _PANE_T), (0.0, 0.0, 0.0),
         tuple(cfg.color), material=mat)
    _box(stage, f"{prim_path}/knob", (_KNOB, _KNOB, _KNOB),
         (0.0, 0.0, _PANE_T / 2 + _KNOB / 2), tuple(cfg.knob_color), material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bench" not in _SPAWNER_CACHE:

        @configclass
        class BenchSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bench)
            mu_bench_s: float = 0.30
            mu_bench_d: float = 0.25

        @configclass
        class KeeperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_keeper)
            mass: float = 0.20
            mu_s: float = 0.20
            mu_d: float = 0.15

        @configclass
        class PaneSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pane)
            side: float = _PANE_S
            mass: float = 0.25
            mu_s: float = 0.50
            mu_d: float = 0.45
            color: tuple = (0.10, 0.65, 0.60)
            knob_color: tuple = (0.05, 0.40, 0.38)

        _SPAWNER_CACHE.update(bench=BenchSpawnerCfg, keeper=KeeperSpawnerCfg,
                              pane=PaneSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GlazingBenchSceneCfg(BaseCfg):
    """Config for `GlazingBenchScene`. Interference geometry is asserted in
    `__post_init__`: the housing slot admits the tongue but not the head, the
    locked tongue overhang blocks a flat pane drop, a pebble props the pane out
    of the flush band, and the decoy can never enter the recess."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_xy: float = tunable(0.010)  # pane center |bench xy| for seated
    seat_z: tuple = tunable((0.033, 0.041))  # pane center z band (flush rest = 0.037;
    #   a pane propped on one 18 mm pebble sits >= ~0.046)
    seat_upz: float = tunable(0.995)  # pane must lie flat
    lock_tip: float = tunable(0.052)  # tongue-tip bench-x <= this => >= 8 mm overlap
    lock_y: float = tunable(0.015)  # keeper center |bench y|
    lock_z: tuple = tunable((0.044, 0.056))  # tongue center z band (rides the top = 0.050)
    lock_align: float = tunable(0.99)  # keeper body +x axis . bench +x (signed)
    bay_out_xy: float = tunable(0.062)  # pebble in-recess window: |x|,|y| <= this ...
    bay_out_z: float = tunable(0.048)  # ... and center z < this (recess rest = 0.040;
    #   on the bench top = 0.054; on the ground far away fails the xy window)
    settle_lin: float = tunable(0.03)  # judged bodies must be at rest
    settle_ang: float = tunable(0.60)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    anchor_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the bench anchor (m)
    yaw_max: float = tunable(20.0)  # uniform +/- bench yaw (deg)
    peb_min: int = tunable(2)  # pebble count range (inclusive)
    peb_max: int = tunable(4)
    peb_jitter: float = tunable(0.008)  # pebble recess-slot jitter (m)
    slot_jitter: float = tunable(0.025)  # ground-slot jitter for pane/decoy/keeper (m)

    # --- info: structure ---------------------------------------------------------------------
    pane_side: float = info(_PANE_S)
    pane_thick: float = info(_PANE_T)
    pane_mass: float = info(0.25)
    decoy_side: float = info(_DECOY_S)
    decoy_mass: float = info(0.45)
    keeper_mass: float = info(0.20)
    peb_size: float = info(_PEB)
    peb_mass: float = info(0.015)
    mu_pane_s: float = info(0.50)
    mu_pane_d: float = info(0.45)
    mu_keeper_s: float = info(0.20)  # keeper + bench are slick-ish: the bar slides
    mu_keeper_d: float = info(0.15)
    mu_bench_s: float = info(0.30)
    mu_bench_d: float = info(0.25)
    mu_peb_s: float = info(0.40)
    mu_peb_d: float = info(0.35)
    mu_ground_s: float = info(0.50)
    mu_ground_d: float = info(0.45)
    jaw_width: float = info(0.078)  # Franka parallel-jaw max opening (embodiment)

    def __post_init__(self) -> None:
        # -- housing slot keys on thickness: tongue passes, head jams --
        assert _TON_T + 0.0015 < _GAP, "tongue must pass under the housing roof"
        assert _HEAD_H > _GAP + 0.015, "head must jam against the housing face"
        assert _TON_W + 0.004 < 2 * _CHAN_HW, "tongue must fit the channel width"
        assert _HEAD_W + 0.004 < 2 * _CHAN_HW, "head must fit between the pillars"
        # -- locked pose: head face on the housing face, tongue overhangs the recess --
        assert abs((_LOCK_X + _HEAD_CX - _HEAD_L / 2) - _HOUSE_X1) < 1e-9, \
            "locked pose = head front face flush with the housing +x face"
        overlap = _BAY_HW - _TIP_LOCK  # 0.015
        assert overlap >= (_BAY_HW - self.lock_tip) + 0.006, \
            "locked overhang must clear the tip threshold with margin"
        assert _BAY_HW - self.lock_tip >= 0.008, "threshold must mean >= 8 mm overlap"
        # -- order forcing: with the keeper locked, a flat pane drop cannot seat --
        assert 2 * _BAY_HW - overlap < _PANE_S - 0.004, \
            "locked tongue must shrink the opening below the pane side"
        # -- pane seats flush with clearance, under the tongue, knob clear of the tongue --
        assert _PANE_S + 0.008 < 2 * _BAY_HW, "pane must drop into the recess freely"
        seat_top = _BAY_FLOOR + _PANE_T  # 0.043
        assert seat_top + 0.0015 < _TOP, "seated pane top must sit below the bench top"
        seat_c = _BAY_FLOOR + _PANE_T / 2  # 0.037
        assert self.seat_z[0] < seat_c < self.seat_z[1]
        assert _KNOB / 2 + 0.030 < _TIP_LOCK, "knob must stay clear of the locked tongue"
        assert _BAY_FLOOR + _PANE_T / 2 + _KNOB < _TOP + _CURB_H + 0.030  # knob graspable
        # -- a single pebble props the pane out of the flush band --
        propped_c = _BAY_FLOOR + _PEB / 2 + (_PANE_T / 2) * 0.9  # tilted-rest lower bound
        assert propped_c > self.seat_z[1] + 0.003, \
            "a pebble-propped pane must rest above the seated z band"
        assert _PEB > _TOP - _BAY_FLOOR + 0.003, "pebble taller than the recess depth"
        # -- decoy refusal: it cannot enter the recess in any yaw --
        assert _DECOY_S > 2 * _BAY_HW + 0.010, "decoy must never fit the recess"
        # -- pebble in/out z separation --
        assert _BAY_FLOOR + _PEB / 2 < self.bay_out_z - 0.006  # in the recess counts
        assert _TOP + _PEB / 2 > self.bay_out_z + 0.004  # on the bench top does not
        # -- drop zone exists on the apron east of the housing --
        assert _DROP_X - _TIP_OFF > _HOUSE_X1 + 0.006, "drop zone clear of the housing"
        assert _DROP_X + _HEAD_CX + _HEAD_L / 2 < _BX1 - 0.004, "keeper fits the apron"
        # -- embodiment: every manipulated part fits the parallel jaw --
        assert _KNOB < self.jaw_width, "pane/decoy knob graspable"
        assert _HEAD_L < self.jaw_width, "keeper head graspable across its length"
        assert _PEB < self.jaw_width, "pebble graspable"
        # -- lock z band brackets the riding tongue, excludes a tongue on the recess floor --
        ride_c = _TOP + _TON_T / 2  # 0.050
        assert self.lock_z[0] < ride_c < self.lock_z[1]
        assert _BAY_FLOOR + _TON_T / 2 < self.lock_z[0] - 0.006
        assert self.peb_min >= 1 and self.peb_max <= len(_PEB_SLOTS)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("glazing_bench")
class GlazingBenchScene(BaseScene):
    cfg: GlazingBenchSceneCfg

    def __init__(self, cfg: GlazingBenchSceneCfg | None = None) -> None:
        super().__init__(cfg or GlazingBenchSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        peb_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_peb_s, dynamic_friction=c.mu_peb_d, restitution=0.0)
        peb_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0)
        peb_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sp["bench"](mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                                  rigid_props=kin,
                                  mu_bench_s=c.mu_bench_s, mu_bench_d=c.mu_bench_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "pane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pane",
                spawn=sp["pane"](side=_PANE_S, mass=c.pane_mass,
                                 mu_s=c.mu_pane_s, mu_d=c.mu_pane_d,
                                 color=(0.10, 0.65, 0.60), knob_color=(0.05, 0.40, 0.38)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_GSLOTS[0][0], _GSLOTS[0][1], _PANE_T / 2 + 0.002)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sp["pane"](side=_DECOY_S, mass=c.decoy_mass,
                                 mu_s=c.mu_pane_s, mu_d=c.mu_pane_d,
                                 color=(0.55, 0.55, 0.58), knob_color=(0.40, 0.40, 0.42)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_GSLOTS[1][0], _GSLOTS[1][1], _PANE_T / 2 + 0.002)),
            ),
            "keeper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Keeper",
                spawn=sp["keeper"](mass=c.keeper_mass,
                                   mu_s=c.mu_keeper_s, mu_d=c.mu_keeper_d),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_BAR_SLOT[0], _BAR_SLOT[1], _TON_T / 2 + 0.002)),
            ),
        }
        for i in range(len(_PEB_SLOTS)):
            out[f"peb_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Peb" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(_PEB, _PEB, _PEB),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.12, 0.10)),
                    physics_material=peb_mat, rigid_props=peb_rigid,
                    collision_props=peb_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.peb_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_PEB_SLOTS[i][0], _PEB_SLOTS[i][1], _BAY_FLOOR + _PEB / 2 + 0.002)),
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
        self.bench: RigidObject = env.iscene["bench"]
        self.pane: RigidObject = env.iscene["pane"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.keeper: RigidObject = env.iscene["keeper"]
        self.pebs: list[RigidObject] = [env.iscene[f"peb_{i}"]
                                        for i in range(len(_PEB_SLOTS))]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.anchor = torch.zeros(n, 2, device=dev)  # bench anchor (env-local xy)
        self.yaw = torch.zeros(n, device=dev)  # bench yaw
        self.present = torch.zeros(n, len(_PEB_SLOTS), dtype=torch.bool, device=dev)
        self.pane_slot = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter/yaw the bench (kinematic teleport), scatter 2-4
        pebbles on the recess floor (absent ones parked off-stage), lay pane and
        decoy flat at shuffled ground slots and the keeper at its slot, free yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        axy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.anchor_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max)
        self.anchor[env_ids] = axy
        self.yaw[env_ids] = yaw
        half = yaw / 2
        zeros = torch.zeros(m, device=dev)
        q_yaw = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(local: torch.Tensor) -> torch.Tensor:
            wx = axy[:, 0] + local[:, 0] * cy - local[:, 1] * sy
            wy = axy[:, 1] + local[:, 0] * sy + local[:, 1] * cy
            return torch.stack([wx, wy, local[:, 2]], dim=-1)

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        def col(x): return torch.full((m,), x, device=dev)

        def free_yaw_quat():
            wyaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.pi
            ch, sh = torch.cos(wyaw / 2), torch.sin(wyaw / 2)
            return torch.stack([ch, zeros, zeros, sh], dim=-1)

        write(self.bench, to_world(torch.stack([zeros, zeros, zeros], dim=-1)), q_yaw)

        # --- pebbles: count in [peb_min, peb_max], shuffled slots + jitter ---
        count = torch.randint(c.peb_min, c.peb_max + 1, (m,), device=dev)
        perm = torch.rand(m, len(_PEB_SLOTS), device=dev).argsort(dim=-1)
        slots = torch.tensor(_PEB_SLOTS, device=dev)
        present = torch.zeros(m, len(_PEB_SLOTS), dtype=torch.bool, device=dev)
        for i in range(len(_PEB_SLOTS)):
            on = count > i
            present[:, i] = on
            sxy = slots[perm[:, i]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.peb_jitter
            px = torch.where(on, sxy[:, 0], col(_PARK[0]))
            py = torch.where(on, sxy[:, 1], col(_PARK[1] + 0.06 * i))
            pz = torch.where(on, col(_BAY_FLOOR + _PEB / 2 + 0.002), col(_PEB / 2 + 0.002))
            write(self.pebs[i], to_world(torch.stack([px, py, pz], dim=-1)),
                  free_yaw_quat())
        self.present[env_ids] = present

        # --- pane / decoy at shuffled ground slots; keeper at its slot ---
        gperm = torch.rand(m, 2, device=dev).argsort(dim=-1)
        self.pane_slot[env_ids] = gperm[:, 0]
        gslots = torch.tensor(_GSLOTS, device=dev)
        for j, body in enumerate([self.pane, self.decoy]):
            sxy = gslots[gperm[:, j]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
            pos = to_world(torch.stack(
                [sxy[:, 0], sxy[:, 1], col(_PANE_T / 2 + 0.002)], dim=-1))
            write(body, pos, free_yaw_quat())
        bxy = torch.tensor(_BAR_SLOT, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        write(self.keeper, to_world(torch.stack(
            [bxy[:, 0], bxy[:, 1], col(_TON_T / 2 + 0.002)], dim=-1)), free_yaw_quat())

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bench": self.bench.data.root_state_w[env_ids].clone(),
            "pane": self.pane.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "keeper": self.keeper.data.root_state_w[env_ids].clone(),
            "pebs": [b.data.root_state_w[env_ids].clone() for b in self.pebs],
            "anchor": self.anchor[env_ids].clone(),
            "yaw": self.yaw[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "pane_slot": self.pane_slot[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.bench.write_root_state_to_sim(state["bench"], env_ids)
        self.pane.write_root_state_to_sim(state["pane"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.keeper.write_root_state_to_sim(state["keeper"], env_ids)
        for b, st in zip(self.pebs, state["pebs"]):
            b.write_root_state_to_sim(st, env_ids)
        self.anchor[env_ids] = state["anchor"]
        self.yaw[env_ids] = state["yaw"]
        self.present[env_ids] = state["present"]
        self.pane_slot[env_ids] = state["pane_slot"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A low glazing bench stands on the floor: a 4.5 cm thick wooden slab with "
            "a square recessed seat (12 x 12 cm, 1.4 cm deep) cut into it, rimmed by a "
            "low dark curb on three sides — the side toward the long apron is open. On "
            "that apron stands a small roofed housing: two dark pillars carrying a "
            "roof plate that leaves only a 1.2 cm high slot above the bench top. "
            "Inside the recess lie a few loose RED offcut cubes (about 1.8 cm — taller "
            "than the recess is deep, so anything laid on top of them rests proud). On "
            "the ground around the bench lie three loose parts: a TEAL square pane "
            "(11 x 11 cm, 1.2 cm thick) with a small grasp knob on its face, a larger "
            "GRAY pane (15.5 cm — too big for the seat) with the same knob, and a tan "
            "keeper bar shaped like a spatula: a thin tongue (1 cm thick) ending in a "
            "tall head (3 cm). Bench position and heading, the number and spots of the "
            "red offcuts, and which slot holds which pane change every episode — read "
            "the scene by looking.\n"
            "Goal, in the only order the geometry allows: first clear every red offcut "
            "out of the recessed seat (set them down anywhere off the seat). Then drop "
            "the TEAL pane flat into the recess so it settles flush — its face must "
            "end below the bench top; on any leftover offcut it rests proud and does "
            "not count. Finally place the keeper bar on the apron beyond the housing, "
            "tongue toward the seat, and slide it under the housing roof until the "
            "head jams against the housing — the thin tongue passes through the slot "
            "and its tip must end overhanging the seated pane's edge by about a "
            "centimeter. The tall head cannot pass the slot, so the bar only locks "
            "tongue-first; and once the tongue overhangs the recess, a pane can no "
            "longer be dropped in — seat the pane before locking the keeper. The gray "
            "pane fits nowhere and is a distractor. Everything must end at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Clear the red offcut cubes out of the recessed seat, drop the teal pane "
            "flat into the recess so it sits flush, then slide the tan keeper bar "
            "tongue-first under the roofed housing until its head jams and the tongue "
            "tip overhangs the pane's edge. Leave the gray pane alone."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def to_bench(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world position -> bench frame (anchor + yaw removed)."""
        p = pos_w - self.env_origins
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        dx = p[:, 0] - self.anchor[:, 0]
        dy = p[:, 1] - self.anchor[:, 1]
        return torch.stack([dx * cy + dy * sy, -dx * sy + dy * cy, p[:, 2]], dim=-1)

    def bench_dir(self, local_vec) -> torch.Tensor:
        """(N, 3) world direction of a bench-frame vector (rotation only)."""
        vx, vy, vz = (float(v) for v in local_vec)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        return torch.stack([vx * cy - vy * sy, vx * sy + vy * cy,
                            torch.full_like(cy, vz)], dim=-1)

    def body_axis(self, body, local_axis) -> torch.Tensor:
        """(N, 3) world direction of a body-frame axis."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ax = torch.tensor([float(v) for v in local_axis],
                          device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ax)

    def up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of the body's +z axis."""
        return self.body_axis(body, (0.0, 0.0, 1.0))[:, 2]

    def settled(self, body, ang: bool = True) -> torch.Tensor:
        c = self.cfg
        ok = body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        if ang:
            ok = ok & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return ok

    def peb_in_bay(self, i: int) -> torch.Tensor:
        """(N,) bool: pebble i is inside the recess volume (bench frame)."""
        c = self.cfg
        loc = self.to_bench(self.pebs[i].data.root_pos_w)
        return (loc[:, 0].abs() <= c.bay_out_xy) & (loc[:, 1].abs() <= c.bay_out_xy) \
            & (loc[:, 2] < c.bay_out_z)

    def pebs_out_frac(self) -> torch.Tensor:
        """(N,) float: fraction of PRESENT pebbles cleared out of the recess."""
        out = torch.zeros_like(self.yaw)
        for i in range(len(self.pebs)):
            out = out + (self.present[:, i] & ~self.peb_in_bay(i)).float()
        return out / self.present.float().sum(dim=-1).clamp(min=1.0)

    def pane_seated(self) -> torch.Tensor:
        """(N,) bool: teal pane flush in the recess, flat, at rest."""
        c = self.cfg
        loc = self.to_bench(self.pane.data.root_pos_w)
        return (loc[:, 0].abs() <= c.seat_xy) & (loc[:, 1].abs() <= c.seat_xy) \
            & (loc[:, 2] >= c.seat_z[0]) & (loc[:, 2] <= c.seat_z[1]) \
            & (self.up_z(self.pane) >= c.seat_upz) & self.settled(self.pane)

    def keeper_locked(self) -> torch.Tensor:
        """(N,) bool: keeper slid home — tongue riding the bench top through the
        housing, tip past the overlap threshold, aligned tongue-first, at rest."""
        c = self.cfg
        loc = self.to_bench(self.keeper.data.root_pos_w)
        tip_x = loc[:, 0] - _TIP_OFF * (
            self.body_axis(self.keeper, (1.0, 0.0, 0.0))
            * self.bench_dir([1.0, 0.0, 0.0])).sum(dim=-1)
        align = (self.body_axis(self.keeper, (1.0, 0.0, 0.0))
                 * self.bench_dir([1.0, 0.0, 0.0])).sum(dim=-1)
        return (tip_x <= c.lock_tip) & (loc[:, 1].abs() <= c.lock_y) \
            & (loc[:, 2] >= c.lock_z[0]) & (loc[:, 2] <= c.lock_z[1]) \
            & (align >= c.lock_align) & self.settled(self.keeper)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: recess cleared of every present pebble, pane seated flush,
        keeper locked (all judged bodies at rest via the seated/locked gates)."""
        return (self.pebs_out_frac() >= 1.0 - 1e-6) & self.pane_seated() \
            & self.keeper_locked()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * pebbles-out fraction + 0.35 * seated, capped
        at 0.50; 1.0 iff success(). Null policy ~0 (pebbles foul the seat, the pane
        lies on the ground far outside the seated window)."""
        base = (0.15 * self.pebs_out_frac()
                + 0.35 * self.pane_seated().float()).clamp(max=0.50)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="glazing_bench", robot="null",
                                      env_spacing=3.0))
