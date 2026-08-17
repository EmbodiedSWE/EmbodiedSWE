"""MokaBalanceScene — weigh the moka pot on a two-pan balance scale (sim_gen task
`libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i141`).

Derived from libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove, but
STRATEGICALLY different. The seed is a single support-surface pick-and-place: grasp
the moka pot and set it down ON the flat stove — success is a pose predicate on ONE
object (xy near the burner site, z in a band), checked the instant the pot rests
there. Here the seed's goal state is this task's START state (the pot begins ON the
unlit stove), and the goal is not a pose at all — it is a MEASUREMENT carried out
with a passive mechanism:

  (1) move the pot from the stove into the RED hanging pan of a balance scale
      (the loaded beam slams to its -12 degree stop: placing the pot visibly
      DISTURBS the scene instead of finishing the task);
  (2) select LABELED WEIGHTS from the six on the counter (2x50 g, 2x100 g,
      2x200 g) and add them to the BLUE pan until their sum equals the pot's
      HIDDEN mass (randomized per episode over {150,200,250,300,350} g — the
      required combination differs every episode and cannot be read from the
      pot's appearance);
  (3) let the mechanism answer: only an exact counterweight brings the free beam
      back level (a 50 g error tips it to the stop; even a 5 g error would hold
      it visibly off level) and success is judged with everything at rest and
      hands off.

A solver therefore needs a different PLAN from the seed (an iterative
weigh-observe-adjust loop over MULTIPLE objects, driven by the beam's response,
with the goal predicate on a MECHANISM's state) and a different code structure
(nothing here is "move X to pose Y": the terminal condition is an equilibrium
certified by the balance itself).

Assets are fully procedural (native PhysX box colliders):
  - deck (KINEMATIC): wooden counter slab 900 x 750 x 24 mm on the ground.
  - stand (KINEMATIC): balance base plate + column + bearing prongs; the pivot
    sits 300 mm above the deck.
  - beam (DYNAMIC, joint to the stand): a 440 mm bar with a needle, on a
    revolute X-axis joint limited to +/-12 degrees. Its CoM 20 mm below the
    pivot is the only restoring torque, so a level beam PROVES <~1.8 g of
    imbalance (tan(2.5 deg) * 0.40 kg * 0.02 m / 0.20 m).
  - basket_r / basket_b (DYNAMIC, joints to the beam ends at y = +/-200 mm):
    hanging pans (crossbar hanger + masts + fenced platform) on free revolute
    X-axis joints — they hang plumb at any beam angle, so the payload's lever
    arm is FIXED at 200 mm regardless of where things sit on the pan.
  - pot (DYNAMIC): octagonal moka pot, 63 mm across, 85 mm tall, side handle.
    Its MASS is randomized per episode via the physics view (readback-verified)
    and is the hidden quantity to be measured.
  - six weights (DYNAMIC): octagonal cylinders on a 2x3 counter grid whose
    assignment is permuted per episode — silver 50 g (small), brass 100 g,
    copper 200 g (large); their sizes/colors are the labels.
  - frypan (DYNAMIC distractor, 230 g — matches no legal combination) and the
    unlit STOVE plate (KINEMATIC) the pot starts on.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * pot_in_red  — pot ever at rest in the red pan (latched)
  0.20 * wt_in_blue  — a labeled weight ever at rest in the blue pan (latched)
  0.20 * balanced    — pot in red + weight(s) in blue + beam level and
                       quasi-static + cached-mass imbalance <= 10 g (latched)
  1.0 iff success()  — live: pot alone in the red pan, ONLY weights in the blue
                       pan, nothing parked on the beam, beam level (2.5 deg) and
                       calm, every payload at rest, |sum(red) - sum(blue)| <=
                       10 g from masses read back from the physics engine at
                       reset, and finite. Non-success capped at 0.60.

The mass clause is an anti-pinning backstop: a gripper (or external wrench)
holding the beam level with the WRONG weights passes every geometric clause but
not this one; conversely any honest level beam passes it automatically because a
free beam holding level physically certifies <= ~1.8 g of imbalance (asserted in
__post_init__).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- procedural compound spawners -------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         orient=None) -> None:
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0, com=None, inertia=None) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    ma = UsdPhysics.MassAPI.Apply(root)
    ma.CreateMassAttr(float(mass))
    if com is not None:  # MassAPI mass alone leaves the CoM at the body origin
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _oct45():
    """wxyz quat: 45 degrees about z (the octagon's second square)."""
    return (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC balance stand. Origin = base-plate BOTTOM centre (on the deck):
    base plate, column, two bearing prongs flanking the beam bar, visual pin."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 8.0, kinematic=True)

    co = cfg.contact_offset
    col = cfg.stand_color
    bs = cfg.stand_base_size
    _box(stage, f"{prim_path}/base", bs, (0.0, 0.0, bs[2] / 2), col, co)
    ch = cfg.pivot_h - 0.035 - bs[2]  # column top 35 mm below the pivot (bar swing room)
    _box(stage, f"{prim_path}/column", (cfg.stand_col_s, cfg.stand_col_s, ch),
         (0.0, 0.0, bs[2] + ch / 2), col, co)
    for sx in (1.0, -1.0):  # bearing prongs flank the beam bar (6 mm clearance)
        _box(stage, f"{prim_path}/prong_{'p' if sx > 0 else 'n'}",
             (0.012, 0.030, 0.060), (sx * 0.024, 0.0, cfg.pivot_h - 0.005),
             col, co)
    # pivot pin: VISUAL ONLY (spans the prongs, would graze the bar otherwise)
    _box(stage, f"{prim_path}/pin", (0.060, 0.008, 0.008),
         (0.0, 0.0, cfg.pivot_h), (0.30, 0.30, 0.32), None)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC balance beam. Origin = the PIVOT. Bar hangs 20 mm below it (the
    CoM offset is the balance's only restoring torque); dark needle points up."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.beam_mass, kinematic=False, lin_damp=0.0,
                ang_damp=cfg.beam_ang_damp, com=(0.0, 0.0, -cfg.beam_com_drop),
                inertia=cfg.beam_inertia)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/bar", cfg.beam_bar_size,
         (0.0, 0.0, -cfg.beam_com_drop), cfg.beam_color, co)
    _box(stage, f"{prim_path}/needle", (0.008, 0.008, 0.060),
         (0.0, 0.0, 0.040), (0.10, 0.10, 0.11), co)
    return root


def _spawn_basket(prim_path: str, cfg: Any, color, translation=None, orientation=None):
    """DYNAMIC hanging pan. Origin = its PIVOT on the beam bar end: crossbar
    hanger (rests visually over the bar), two masts, fenced platform below."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.basket_mass, kinematic=False, lin_damp=cfg.basket_lin_damp,
                ang_damp=cfg.basket_ang_damp, com=(0.0, 0.0, -cfg.basket_com_drop),
                inertia=cfg.basket_inertia)

    co = cfg.contact_offset
    _box(stage, f"{prim_path}/crossbar", (0.174, 0.012, 0.012), (0.0, 0.0, 0.0), color, co)
    for sx in (1.0, -1.0):
        _box(stage, f"{prim_path}/mast_{'p' if sx > 0 else 'n'}",
             (0.012, 0.012, 0.124), (sx * 0.081, 0.0, -0.068), color, co)
    pw, pd = cfg.platform_w, cfg.platform_d
    _box(stage, f"{prim_path}/platform", (pw, pd, 0.008),
         (0.0, 0.0, -cfg.platform_drop - 0.004), color, co)
    rz = -cfg.platform_drop + cfg.rim_h / 2
    for sx in (1.0, -1.0):
        _box(stage, f"{prim_path}/rim_x{'p' if sx > 0 else 'n'}",
             (cfg.rim_t, pd, cfg.rim_h), (sx * (pw / 2 - cfg.rim_t / 2), 0.0, rz), color, co)
    for sy in (1.0, -1.0):
        _box(stage, f"{prim_path}/rim_y{'p' if sy > 0 else 'n'}",
             (pw - 2 * cfg.rim_t, cfg.rim_t, cfg.rim_h),
             (0.0, sy * (pd / 2 - cfg.rim_t / 2), rz), color, co)
    return root


def _spawn_basket_r(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    return _spawn_basket(prim_path, c, c.red_color, translation, orientation)


def _spawn_basket_b(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    return _spawn_basket(prim_path, c, c.blue_color, translation, orientation)


def _spawn_pot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC moka pot. Origin = base BOTTOM centre: octagonal base + waisted
    octagonal top + lid knob + side handle along local +x. Its authored mass is
    the DEFAULT — reset() overwrites it through the physics view."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    m = cfg.pot_default_mass
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, m, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.038),
                inertia=tuple(m * u for u in cfg.pot_unit_inertia))

    co = cfg.contact_offset
    body, dark = cfg.pot_color, cfg.pot_handle_color
    b, bh = cfg.pot_base_s, 0.044
    t, th = 0.050, 0.032
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/base_{tag}", (b, b, bh), (0.0, 0.0, bh / 2), body, co,
             orient=orient)
        _box(stage, f"{prim_path}/top_{tag}", (t, t, th), (0.0, 0.0, bh + th / 2), body, co,
             orient=orient)
    _box(stage, f"{prim_path}/knob", (0.014, 0.014, 0.009),
         (0.0, 0.0, bh + th + 0.0045), dark, co)
    _box(stage, f"{prim_path}/handle", (0.036, 0.012, 0.020),
         (0.043, 0.0, 0.055), dark, co)
    return root


def _spawn_weight(prim_path: str, cfg: Any, s: float, h: float, mass: float, color):
    """DYNAMIC labeled weight: an octagonal puck (two stacked squares), origin
    at its BOTTOM centre."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    ixy = mass * (s * s + h * h) / 12.0
    iz = mass * (s * s) / 6.0
    _rigid_root(root, mass, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, h / 2), inertia=(ixy, ixy, iz))
    co = cfg.contact_offset
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/puck_{tag}", (s, s, h), (0.0, 0.0, h / 2), color, co,
             orient=orient)
    # small top nub so the grades read as weights
    _box(stage, f"{prim_path}/nub", (0.008, 0.008, 0.006), (0.0, 0.0, h + 0.003), color, co)
    return root


def _spawn_w50(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xform.Define(stage, prim_path)
    _apply_xform(xf, translation, orientation)
    return _spawn_weight(prim_path, c, c.w50_s, c.w50_h, 0.050, c.silver_color)


def _spawn_w100(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xform.Define(stage, prim_path)
    _apply_xform(xf, translation, orientation)
    return _spawn_weight(prim_path, c, c.w100_s, c.w100_h, 0.100, c.brass_color)


def _spawn_w200(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xform.Define(stage, prim_path)
    _apply_xform(xf, translation, orientation)
    return _spawn_weight(prim_path, c, c.w200_s, c.w200_h, 0.200, c.copper_color)


def _spawn_frypan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC distractor frypan (230 g — matches no legal weight combination)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    m = cfg.frypan_mass
    _rigid_root(root, m, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.008), inertia=(m * 0.0012, m * 0.0012, m * 0.002))
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/base", (0.110, 0.110, 0.012), (0.0, 0.0, 0.006),
         cfg.frypan_color, co)
    _box(stage, f"{prim_path}/handle", (0.050, 0.016, 0.010), (0.080, 0.0, 0.014),
         (0.05, 0.05, 0.06), co)
    return root


def _spawn_stove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC unlit stove plate (the pot starts here — the seed's goal spot)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 4.0, kinematic=True)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/base", (0.160, 0.160, 0.008), (0.0, 0.0, 0.004),
         (0.30, 0.30, 0.33), co)
    _box(stage, f"{prim_path}/coil", (0.100, 0.100, 0.004), (0.0, 0.0, 0.010),
         (0.12, 0.12, 0.13), co)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0) -> Any:
    """Build (once) and instantiate a RigidObjectSpawnerCfg subclass wrapping
    `spawn_fn`, carrying the scene cfg through a single `cfg` field. The
    spawner-cfg rigid_props are applied by the isaaclab clone wrapper AFTER the
    spawn fn runs, so they are the authoritative word (sleep stays OFF: sleeping
    GPU bodies freeze mid-settle and ignore velocity writes — corpus lesson)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None

        _Cfg.__name__ = f"{kind.title()}SpawnerCfg"
        _SPAWNER_CACHE[kind] = _Cfg

    if kinematic:
        rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rp = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, sleep_threshold=0.0, stabilization_threshold=0.0,
            max_depenetration_velocity=0.5,
            linear_damping=lin_damp, angular_damping=ang_damp,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class MokaBalanceSceneCfg(BaseCfg):
    """Config for `MokaBalanceScene`. Honesty physics asserted in __post_init__."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    level_max_deg: float = tunable(2.5)     # beam tilt within this of level (deg)
    beam_calm_omega: float = tunable(0.15)  # quasi-static gate on the beam (rad/s)
    settle_speed: float = tunable(0.10)     # payload lin speed at rest (m/s)
    basket_calm_omega: float = tunable(0.50)  # residual pan swing gate (rad/s)
    latch_speed: float = tunable(0.10)      # calm gate for the pot/weight latches (m/s)
    imb_tol: float = tunable(0.010)         # anti-pinning mass clause (kg)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pot_mass_choices: tuple = tunable((0.150, 0.200, 0.250, 0.300, 0.350))
    stove_jitter: tuple = tunable((0.015, 0.015))  # stove xy jitter (+/- m)
    pot_jitter: float = tunable(0.008)     # pot xy jitter on the stove (+/- m)
    pot_yaw_deg: float = tunable(180.0)    # pot free yaw (+/- deg)
    weight_jitter: float = tunable(0.006)  # weight xy jitter at its grid spot (+/- m)
    weight_yaw_deg: float = tunable(180.0)
    frypan_jitter: tuple = tunable((0.020, 0.020))
    frypan_yaw_deg: float = tunable(30.0)

    # --- info: layout (world nominal, deck top = z 0.024) ----------------------------------------
    deck_size: tuple = info((0.90, 0.75, 0.024))
    deck_pos: tuple = info((0.45, 0.0))
    stand_pos: tuple = info((0.24, 0.0))
    stove_pos: tuple = info((0.42, 0.27))
    stove_top_dz: float = info(0.012)       # stove collider top above the deck top
    weight_grid_x: tuple = info((0.48, 0.56))
    weight_grid_y: tuple = info((-0.30, -0.21, -0.12))
    frypan_pos: tuple = info((0.62, 0.25))
    # --- info: stand / beam / baskets ------------------------------------------------------------
    stand_base_size: tuple = info((0.16, 0.16, 0.012))
    stand_col_s: float = info(0.030)
    pivot_h: float = info(0.300)            # pivot height above the DECK TOP
    beam_bar_size: tuple = info((0.024, 0.44, 0.020))
    beam_mass: float = info(0.40)
    beam_com_drop: float = info(0.020)      # CoM below the pivot (restoring arm)
    beam_inertia: tuple = info((0.007, 0.0004, 0.007))
    beam_ang_damp: float = info(2.0)        # zeta ~0.76 at the balance's slow mode
    beam_limit_deg: float = info(12.0)      # revolute stop
    hang_y: float = info(0.200)             # basket pivots at beam local y = +/- this
    basket_mass: float = info(0.10)
    basket_com_drop: float = info(0.115)
    basket_inertia: tuple = info((0.00035, 0.00035, 0.00025))
    basket_lin_damp: float = info(0.8)
    basket_ang_damp: float = info(1.5)
    basket_limit_deg: float = info(30.0)
    platform_w: float = info(0.150)         # pan platform (x)
    platform_d: float = info(0.140)         # pan platform (y)
    platform_drop: float = info(0.130)      # platform TOP below the basket pivot
    rim_t: float = info(0.008)
    rim_h: float = info(0.014)
    # --- info: pot / weights / frypan ------------------------------------------------------------
    pot_base_s: float = info(0.063)         # octagon across-flats (base)
    pot_h: float = info(0.085)
    pot_default_mass: float = info(0.250)
    pot_unit_inertia: tuple = info((0.00093, 0.00093, 0.00066))  # per kg
    w50_s: float = info(0.026)
    w50_h: float = info(0.014)
    w100_s: float = info(0.034)
    w100_h: float = info(0.018)
    w200_s: float = info(0.044)
    w200_h: float = info(0.022)
    frypan_mass: float = info(0.230)        # matches NO legal weight combination
    # --- info: judge volumes ---------------------------------------------------------------------
    bask_vol_x: float = info(0.070)         # basket-local |x| bound
    bask_vol_y: float = info(0.068)         # basket-local |y| bound
    bask_vol_z: tuple = info((-0.142, -0.095))  # basket-local z band (origin near platform)
    bar_band_x: float = info(0.080)         # beam-local no-parking band
    bar_band_y: float = info(0.260)
    bar_band_z: tuple = info((-0.025, 0.120))
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.55, 0.44, 0.30))
    stand_color: tuple = info((0.35, 0.36, 0.40))
    beam_color: tuple = info((0.62, 0.63, 0.68))
    red_color: tuple = info((0.80, 0.10, 0.10))
    blue_color: tuple = info((0.10, 0.20, 0.80))
    pot_color: tuple = info((0.55, 0.56, 0.60))
    pot_handle_color: tuple = info((0.05, 0.05, 0.06))
    silver_color: tuple = info((0.75, 0.76, 0.78))
    brass_color: tuple = info((0.72, 0.58, 0.20))
    copper_color: tuple = info((0.70, 0.40, 0.25))
    frypan_color: tuple = info((0.15, 0.15, 0.17))
    # rubric weights (0.20 * 3 = 0.60 = the non-success cap)
    w_pot: float = info(0.20)
    w_wt: float = info(0.20)
    w_lvl: float = info(0.20)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def pivot_z(self) -> float:
        return self.deck_top + self.pivot_h

    @property
    def weight_masses(self) -> tuple:
        return (0.050, 0.050, 0.100, 0.100, 0.200, 0.200)

    @property
    def weight_spots(self) -> tuple:
        return tuple((gx, gy) for gx in self.weight_grid_x for gy in self.weight_grid_y)

    @property
    def bask_inner_x(self) -> float:
        return self.platform_w / 2 - self.rim_t

    @property
    def bask_inner_y(self) -> float:
        return self.platform_d / 2 - self.rim_t

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the physical claims the task rests on).
        stat = self.beam_mass * self.beam_com_drop / self.hang_y  # kg per unit tan(tilt)
        assert math.tan(math.radians(self.level_max_deg)) * stat < self.imb_tol, \
            "a FREE beam holding level must certify less imbalance than the mass clause"
        assert math.tan(math.radians(self.beam_limit_deg)) * stat < self.imb_tol, \
            "any imbalance the mass clause rejects must also slam a free beam to its stop"
        achievable = set()
        for r in range(1, 7):
            for combo in combinations(self.weight_masses, r):
                achievable.add(round(sum(combo), 4))
        for mch in self.pot_mass_choices:
            assert round(mch, 4) in achievable, f"pot mass {mch} not weighable exactly"
        assert all(abs(self.frypan_mass - a) > 1.5 * self.imb_tol for a in
                   achievable | {0.0}), "the frypan must not equal any weight combination"
        # pot fits the pan with margin (body corner radius + the side handle reach)
        pot_r_corner = self.pot_base_s * math.sqrt(2.0) / 2
        assert pot_r_corner < self.bask_inner_y - 0.004, "pot body must fit the pan (y)"
        assert 0.043 + 0.018 < self.bask_inner_x - 0.004, "pot handle must fit the pan (x)"
        # pot + hover clearance passes under the hanger crossbar
        assert self.pot_h + 0.012 < self.platform_drop - 0.006, \
            "the pot must fit below the hanger crossbar with drop clearance"
        # weight grid spacing beats worst-case footprints + jitter
        r200 = self.w200_s * math.sqrt(2.0) / 2
        min_gap = min(self.weight_grid_x[1] - self.weight_grid_x[0],
                      self.weight_grid_y[1] - self.weight_grid_y[0])
        assert min_gap > 2 * r200 + 2 * self.weight_jitter + 0.004, \
            "adjacent grid weights must never spawn in contact"
        # a fully tipped beam keeps the pans well above the deck
        low = self.pivot_h - self.hang_y * math.sin(math.radians(self.beam_limit_deg)) \
            - self.platform_drop - 0.008
        assert low > 0.05, "pans must stay clear of the deck at full beam tilt"
        # the stove (with jitter) stays clear of the red pan's footprint
        assert self.stove_pos[0] - 0.080 - self.stove_jitter[0] \
            > self.stand_pos[0] + self.platform_w / 2 + 0.004, \
            "stove must not spawn under the red pan"
        # the weight grid stays clear of the blue pan's footprint
        assert self.weight_grid_x[0] - self.weight_jitter - r200 \
            > self.stand_pos[0] + self.platform_w / 2 + 0.02, \
            "weights must not spawn under the blue pan"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("moka_balance")
class MokaBalanceScene(BaseScene):
    cfg: MokaBalanceSceneCfg

    # payload index map (order of self.payloads): 0 pot, 1..6 weights, 7 frypan
    WEIGHT_KEYS = ("w50a", "w50b", "w100a", "w100b", "w200a", "w200b")

    def __init__(self, cfg: MokaBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or MokaBalanceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def kin_slab(size, color):
            return sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        out = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=kin_slab(c.deck_size, c.deck_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], c.deck_size[2] / 2)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=_compound_spawner_cfg("stand", _spawn_stand, c, 8.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], c.deck_top)),
            ),
            "stove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stove",
                spawn=_compound_spawner_cfg("stove", _spawn_stove, c, 4.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stove_pos[0], c.stove_pos[1], c.deck_top)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=_compound_spawner_cfg("beam", _spawn_beam, c, c.beam_mass,
                                            kinematic=False, lin_damp=0.0,
                                            ang_damp=c.beam_ang_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], c.pivot_z)),
            ),
            "basket_r": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BasketR",
                spawn=_compound_spawner_cfg("basket_r", _spawn_basket_r, c, c.basket_mass,
                                            kinematic=False, lin_damp=c.basket_lin_damp,
                                            ang_damp=c.basket_ang_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1] + c.hang_y, c.pivot_z)),
            ),
            "basket_b": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BasketB",
                spawn=_compound_spawner_cfg("basket_b", _spawn_basket_b, c, c.basket_mass,
                                            kinematic=False, lin_damp=c.basket_lin_damp,
                                            ang_damp=c.basket_ang_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1] - c.hang_y, c.pivot_z)),
            ),
            "pot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot",
                spawn=_compound_spawner_cfg("pot", _spawn_pot, c, c.pot_default_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stove_pos[0], c.stove_pos[1],
                         c.deck_top + c.stove_top_dz + 0.002)),
            ),
            "frypan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frypan",
                spawn=_compound_spawner_cfg("frypan", _spawn_frypan, c, c.frypan_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.frypan_pos[0], c.frypan_pos[1], c.deck_top + 0.001)),
            ),
        }
        w_spawn = {"w50": _spawn_w50, "w100": _spawn_w100, "w200": _spawn_w200}
        w_mass = {"w50": 0.050, "w100": 0.100, "w200": 0.200}
        for i, key in enumerate(self.WEIGHT_KEYS):
            grade = key[:-1]
            out[key] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + key.upper(),
                spawn=_compound_spawner_cfg(grade, w_spawn[grade], c, w_mass[grade],
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.weight_spots[i][0], c.weight_spots[i][1], c.deck_top + 0.001)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external wrenches (the smoke pinning probe) act cleanly only
                # with per-iteration application — the sim's own recommendation
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
        n, dev = env.num_envs, env.device
        self.deck: RigidObject = env.iscene["deck"]
        self.stand: RigidObject = env.iscene["stand"]
        self.stove: RigidObject = env.iscene["stove"]
        self.beam: RigidObject = env.iscene["beam"]
        self.basket_r: RigidObject = env.iscene["basket_r"]
        self.basket_b: RigidObject = env.iscene["basket_b"]
        self.pot: RigidObject = env.iscene["pot"]
        self.frypan: RigidObject = env.iscene["frypan"]
        self.weights: list[RigidObject] = [env.iscene[k] for k in self.WEIGHT_KEYS]
        self.payloads: list[RigidObject] = [self.pot, *self.weights, self.frypan]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # cached payload masses (N, 8) — refreshed from the physics view at reset
        self.masses_cache = torch.zeros(n, 8, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._l_pot = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_wt = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_lvl = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: revolute X joints stand->beam (the knife edge, +/-12 deg) and
        beam->each basket (free hangers, +/-30 deg). Joint pairs never collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.pivot_h))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.beam_limit_deg)
            j.CreateUpperLimitAttr(c.beam_limit_deg)
            for tag, sy in (("r", 1.0), ("b", -1.0)):
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/hanger_{tag}")
                j.CreateBody0Rel().SetTargets([f"{base}/Beam"])
                j.CreateBody1Rel().SetTargets(
                    [f"{base}/Basket{'R' if sy > 0 else 'B'}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("X")
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, sy * c.hang_y, 0.0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-c.basket_limit_deg)
                j.CreateUpperLimitAttr(c.basket_limit_deg)

    def _set_body_mass(self, body, env_ids: torch.Tensor, new_mass: torch.Tensor) -> None:
        """Write per-env masses through the physics view, scaling the inertia by
        the mass ratio (geometry unchanged). CPU indices per the view API."""
        view = body.root_physx_view
        ids = env_ids.detach().to("cpu", dtype=torch.long)
        masses = view.get_masses().clone()
        flat = masses.view(masses.shape[0], -1)
        ratio = new_mass.detach().cpu().view(-1, 1) / flat[ids]
        flat[ids] = new_mass.detach().cpu().view(-1, 1)
        inertias = view.get_inertias().clone()
        inertias.view(inertias.shape[0], -1)[ids] *= ratio
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)

    def refresh_cached_masses(self) -> None:
        """Re-read every payload's mass from the physics engine into the cache
        the mass clause uses (public: smoke instrumentation re-syncs with it)."""
        cols = []
        for b in self.payloads:
            m = b.root_physx_view.get_masses()
            cols.append(m.view(m.shape[0], -1)[:, 0].to(self.env.device))
        self.masses_cache = torch.stack(cols, dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stove jitter, pot on the stove (free yaw, hidden random
        mass), beam+baskets written LEVEL/PLUMB as one linkage, weights permuted
        over the counter grid, frypan jitter; latches cleared, masses cached."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            body.write_root_state_to_sim(st, env_ids)

        # --- stove: kinematic, xy jitter ---
        spos = torch.zeros(m, 3, device=dev)
        spos[:, 0] = c.stove_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stove_jitter[0]
        spos[:, 1] = c.stove_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stove_jitter[1]
        spos[:, 2] = c.deck_top
        write(self.stove, spos)

        # --- pot: on the stove coil, free yaw, hidden random mass ---
        ppos = spos.clone()
        ppos[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pot_jitter
        ppos[:, 2] = c.deck_top + c.stove_top_dz + 0.002
        pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pot_yaw_deg)
        write(self.pot, ppos, _qz(pyaw))
        choices = torch.tensor(c.pot_mass_choices, device=dev)
        # torch.rand-based draw (first-randint-after-seed degeneracy — corpus lesson)
        idx = (torch.rand(m, device=dev) * len(c.pot_mass_choices)).long().clamp(
            max=len(c.pot_mass_choices) - 1)
        self._set_body_mass(self.pot, env_ids, choices[idx])

        # --- balance: whole linkage written together, level and plumb ---
        bpos = torch.zeros(m, 3, device=dev)
        bpos[:, 0], bpos[:, 1], bpos[:, 2] = c.stand_pos[0], c.stand_pos[1], c.pivot_z
        write(self.beam, bpos)
        for basket, sy in ((self.basket_r, 1.0), (self.basket_b, -1.0)):
            kpos = bpos.clone()
            kpos[:, 1] += sy * c.hang_y
            write(basket, kpos)

        # --- weights: per-env permutation over the grid (torch.rand + argsort) ---
        spots = torch.tensor(c.weight_spots, device=dev)  # (6, 2)
        perm = torch.rand(m, 6, device=dev).argsort(dim=1)
        for k, body in enumerate(self.weights):
            sxy = spots[perm[:, k]]
            wpos = torch.zeros(m, 3, device=dev)
            wpos[:, 0:2] = sxy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.weight_jitter
            wpos[:, 2] = c.deck_top + 0.001
            wyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.weight_yaw_deg)
            write(body, wpos, _qz(wyaw))

        # --- frypan distractor ---
        fpos = torch.zeros(m, 3, device=dev)
        fpos[:, 0] = c.frypan_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.frypan_jitter[0]
        fpos[:, 1] = c.frypan_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.frypan_jitter[1]
        fpos[:, 2] = c.deck_top + 0.001
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.frypan_yaw_deg)
        write(self.frypan, fpos, _qz(fyaw))

        # --- clear latches, cache the as-reset masses (the mass clause's truth) ---
        for latch in (self._l_pot, self._l_wt, self._l_lvl):
            latch[env_ids] = False
        self.refresh_cached_masses()

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {
            "stove": self.stove.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "basket_r": self.basket_r.data.root_state_w[env_ids].clone(),
            "basket_b": self.basket_b.data.root_state_w[env_ids].clone(),
            "pot": self.pot.data.root_state_w[env_ids].clone(),
            "frypan": self.frypan.data.root_state_w[env_ids].clone(),
            "masses": self.masses_cache[env_ids].clone(),
            "l_pot": self._l_pot[env_ids].clone(),
            "l_wt": self._l_wt[env_ids].clone(),
            "l_lvl": self._l_lvl[env_ids].clone(),
        }
        for k, body in zip(self.WEIGHT_KEYS, self.weights):
            out[k] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("stove", "beam", "basket_r", "basket_b", "pot", "frypan",
                     *self.WEIGHT_KEYS):
            body = getattr(self, name) if hasattr(self, name) else None
            if body is None:
                body = self.weights[self.WEIGHT_KEYS.index(name)]
            body.write_root_state_to_sim(state[name], env_ids)
        self.masses_cache[env_ids] = state["masses"]
        self._l_pot[env_ids] = state["l_pot"]
        self._l_wt[env_ids] = state["l_wt"]
        self._l_lvl[env_ids] = state["l_lvl"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden COUNTER ({c.deck_size[0] * 100:.0f} x {c.deck_size[1] * 100:.0f} cm "
            f"slab) lies on the ground. On it stands a two-pan BALANCE SCALE: a gray "
            f"column holds a knife-edge pivot {c.pivot_h * 100:.0f} cm up, carrying a "
            f"{c.beam_bar_size[1] * 100:.0f} cm beam that can tip +/-"
            f"{c.beam_limit_deg:.0f} degrees, with a needle at its centre. From the "
            f"beam's ends hang two pans on free pivots: a RED pan on one side and a "
            f"BLUE pan on the other. The beam's only restoring torque is its own "
            f"centre of mass just below the pivot, so it rests level ONLY when the "
            f"two pans carry equal weight — roughly 2 grams of difference already "
            f"tilts it visibly, and 50 grams slams it to its stop.\n"
            f"An UNLIT STOVE plate sits on the counter (its position varies) with an "
            f"octagonal aluminium MOKA POT standing on it. The pot's mass differs "
            f"per episode (somewhere between 150 g and 350 g in 50 g steps) and "
            f"cannot be judged from its appearance. Near the counter's front edge lie "
            f"SIX LABELED WEIGHTS in a shuffled grid: two small SILVER 50 g, two "
            f"BRASS 100 g and two large COPPER 200 g octagonal pucks. A dark FRYING "
            f"PAN also lies on the counter; it is not a labeled weight.\n"
            f"Goal: WEIGH the moka pot with the balance. Move the pot from the stove "
            f"into the RED pan (alone), then place labeled weights on the BLUE pan "
            f"until their sum equals the pot's mass, and let everything come to rest. "
            f"Success: the pot sits in the red pan, ONLY labeled weights sit in the "
            f"blue pan (their total matching the pot within {c.imb_tol * 1000:.0f} g), "
            f"nothing rests on the beam itself, and the free beam holds LEVEL (within "
            f"{c.level_max_deg:.1f} degrees) with the beam and every object at rest, "
            f"hands off. A beam held level by force with the wrong weights, weights "
            f"sharing the red pan, the frying pan used as a counterweight, or a "
            f"still-swinging beam passing through level does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Weigh the moka pot: move it from the stove into the balance's red pan, "
            "then add labeled weights to the blue pan until the beam settles level. "
            "Only the pot in the red pan, only weights in the blue pan, hands off at "
            "the end."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def beam_tilt(self) -> torch.Tensor:
        """(N,) signed beam tilt in RADIANS (positive = red/+y end up): the
        elevation of the beam's local +y axis."""
        from isaaclab.utils.math import quat_apply

        q = self.beam.data.root_quat_w
        ey = torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(q.shape[0], 3)
        return torch.asin(quat_apply(q, ey)[:, 2].clamp(-1.0, 1.0))

    def level(self) -> torch.Tensor:
        return self.beam_tilt().abs() <= math.radians(self.cfg.level_max_deg)

    def beam_calm(self) -> torch.Tensor:
        return self.beam.data.root_ang_vel_w.norm(dim=-1) <= self.cfg.beam_calm_omega

    def _payload_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.payloads], dim=1)  # (N,8,3)

    def _payload_speed(self) -> torch.Tensor:
        return torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in self.payloads], dim=1)  # (N,8)

    def in_basket(self, basket) -> torch.Tensor:
        """(N,8) bool: payload origin inside the pan's cargo volume (basket-local
        box just above the platform — a pan's cargo rides with it at any tilt)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q, p = basket.data.root_quat_w, basket.data.root_pos_w
        pos = self._payload_pos()
        n, k, _ = pos.shape
        qq = q[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        lp = quat_apply_inverse(qq, (pos - p[:, None, :]).reshape(n * k, 3)).reshape(n, k, 3)
        return (lp[:, :, 0].abs() <= c.bask_vol_x) & (lp[:, :, 1].abs() <= c.bask_vol_y) \
            & (lp[:, :, 2] > c.bask_vol_z[0]) & (lp[:, :, 2] < c.bask_vol_z[1])

    def bar_parked(self) -> torch.Tensor:
        """(N,) bool: some payload rests ON the beam bar (beam-local band above
        the bar, between the hangers) — parking mass on the beam is not weighing."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        q, p = self.beam.data.root_quat_w, self.beam.data.root_pos_w
        pos = self._payload_pos()
        n, k, _ = pos.shape
        qq = q[:, None, :].expand(n, k, 4).reshape(n * k, 4)
        lp = quat_apply_inverse(qq, (pos - p[:, None, :]).reshape(n * k, 3)).reshape(n, k, 3)
        band = (lp[:, :, 0].abs() <= c.bar_band_x) & (lp[:, :, 1].abs() <= c.bar_band_y) \
            & (lp[:, :, 2] > c.bar_band_z[0]) & (lp[:, :, 2] < c.bar_band_z[1])
        return band.any(dim=1)

    def imbalance(self) -> torch.Tensor:
        """(N,) kg: sum of cached masses geometrically in the RED pan minus the
        BLUE pan. Masses come from the physics view AT RESET (readback, not
        nominal) — a beam pinned level with the wrong weights cannot pass."""
        in_r = self.in_basket(self.basket_r).float()
        in_b = self.in_basket(self.basket_b).float()
        return (self.masses_cache * in_r).sum(dim=1) - (self.masses_cache * in_b).sum(dim=1)

    def pot_on_stove(self) -> torch.Tensor:
        """(N,) bool: pot standing on the stove plate (the initial state)."""
        d = (self.pot.data.root_pos_w[:, :2]
             - self.stove.data.root_pos_w[:, :2]).norm(dim=-1)
        dz = (self.pot.data.root_pos_w[:, 2] - self.stove.data.root_pos_w[:, 2]) \
            - self.cfg.stove_top_dz
        return (d < 0.06) & (dz.abs() < 0.02)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every payload at rest and the pans not visibly swinging."""
        c = self.cfg
        pay = (self._payload_speed() < c.settle_speed).all(dim=1)
        pans = torch.ones_like(pay)
        for basket in (self.basket_r, self.basket_b):
            pans &= (basket.data.root_ang_vel_w.norm(dim=-1) < c.basket_calm_omega) \
                & (basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return pay & pans

    def misplaced(self) -> torch.Tensor:
        """(N,) bool: any non-pot payload in the red pan, the pot or frypan in
        the blue pan, or anything parked on the beam."""
        in_r = self.in_basket(self.basket_r)
        in_b = self.in_basket(self.basket_b)
        return in_r[:, 1:].any(dim=1) | in_b[:, 0] | in_b[:, 7] | self.bar_parked()

    def _update_latches(self) -> None:
        c = self.cfg
        in_r = self.in_basket(self.basket_r)
        in_b = self.in_basket(self.basket_b)
        speed = self._payload_speed()
        calm = speed < c.latch_speed
        self._l_pot |= in_r[:, 0] & calm[:, 0]
        self._l_wt |= (in_b[:, 1:7] & calm[:, 1:7]).any(dim=1)
        # balanced latch: quasi-static (a beam SWINGING through level crosses at
        # ~1.5 rad/s — the omega gate rejects it) and honest by cached mass
        self._l_lvl |= in_r[:, 0] & in_b[:, 1:7].any(dim=1) & self.level() \
            & self.beam_calm() & (self.imbalance().abs() <= c.imb_tol)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, all live: pot alone in the red pan, only labeled weights in
        the blue pan, nothing on the beam, the FREE beam level and quasi-static,
        every payload and pan at rest, cached-mass imbalance within tolerance,
        finite. The mechanism itself certifies the measurement: a free beam that
        holds level physically proves < ~1.8 g of imbalance."""
        c = self.cfg
        self._update_latches()
        in_r = self.in_basket(self.basket_r)
        in_b = self.in_basket(self.basket_b)
        finite = torch.isfinite(self.pot.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.beam.data.root_pos_w).all(dim=-1)
        return in_r[:, 0] & in_b[:, 1:7].any(dim=1) & ~self.misplaced() \
            & self.level() & self.beam_calm() & self.settled() \
            & (self.imbalance().abs() <= c.imb_tol) & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*pot_in_red + 0.20*weight_in_blue +
        0.20*balanced (all latched; ~0 for doing nothing), capped at 0.60 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_pot * self._l_pot.float() + c.w_wt * self._l_wt.float()
                + c.w_lvl * self._l_lvl.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="moka_balance", robot="null"))
