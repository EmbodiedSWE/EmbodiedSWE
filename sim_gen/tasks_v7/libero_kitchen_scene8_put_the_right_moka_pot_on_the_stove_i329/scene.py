"""CubbyRakeStoveScene — rake the copper pot out of the roofed cubby, then stove it
(sim_gen task `libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i329`).

Derived from libero_90/libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove,
but STRATEGICALLY different. The seed is a direct pick-and-place with a
positional identity twist: two identical moka pots stand in the open, grab the
RIGHT one and set it on an always-accessible stove — the hand reaches the goal
object from step one. Here the two pots start DEEP INSIDE a low roofed CUBBY
(a two-lane shelf unit): each pot sits ~170 mm behind the mouth of an 85 mm
wide, 150 mm tall lane. No gripper fits down the lane, and the roof kills any
top-down grasp — the goal object is UNREACHABLE by hand. The counter carries
the remedy: a long-handled RAKE (a flat bar with a downward hook flange at its
far end). The solver must:

  (1) slide the rake nose-first into the COPPER pot's lane, riding ABOVE the
      pot, until the flange passes beyond the pot's knob;
  (2) lower the rake so the flange drops into the gap behind the knob, hooking
      the knob head;
  (3) DRAG the pot down the lane and out of the mouth onto the slick apron —
      a sub-tipping friction pull (slick cubby floor, catch below the tipping
      threshold) that is the load-bearing mechanic of the task;
  (4) only now pick the freed pot and set it upright on the BURNER PLATE on
      the open counter, leaving the steel pot untouched in its lane, and take
      hands off everything.

A solver therefore needs a different PLAN from the seed (tool acquisition ->
guided insertion -> hook -> extraction drag -> place, where the placement is
physically impossible before the extraction) and different CODE structure (the
core interaction is a held-tool contact drag through a confined corridor, not
a grasp of the goal object). Distinct from sibling i167 (falsework-prop): here
no mechanism is opened and nothing must stand hands-free under load — the gate
is REACH: a recessed object that only a wielded tool can retrieve.

Assets are fully procedural (native PhysX box colliders):
  - deck (KINEMATIC): counter slab 900 x 750 x 24 mm on the ground.
  - cubby (KINEMATIC): floor slab + back wall + two side walls + centre rib +
    roof. Two lanes 85 mm wide x 260 mm deep x 150 mm tall, open at the front.
    The floor slab extends 100 mm past the mouth as a pale APRON. Floor and
    apron are SLICK (authored physics material) — the drag slides the pot
    instead of tipping it.
  - stove (KINEMATIC): a small base plate + raised octagonal BURNER PLATE on
    the open counter, GRIPPY so the delivered pot stays put.
  - rake (DYNAMIC): 400 mm flat bar, hook flange (22 mm drop) at the far end,
    grip block at the near end.
  - pot_t / pot_w (DYNAMIC): two same-shape octagonal moka pots with a
    mushroom-head lid knob — COPPER (target) and STEEL (distractor). Which
    LANE the copper one spawns in is a per-episode coin flip.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * reached   — the copper pot ever dragged >= 40 mm mouth-ward while
                     still inside the cubby (latched)
  0.30 * extracted — the copper pot ever fully outside the mouth, upright and
                     at rest on the apron (latched)
  0.25 * on_stove  — the copper pot ever upright at rest on the burner plate
                     (latched)
  1.0 iff success() — live: copper pot upright at rest on the burner plate,
                     steel pot still inside the cubby, the rake clear of the
                     stove, everything at rest, finite. Non-success capped at
                     0.75.

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


def _shared_mat(stage, name: str, mu_s: float, mu_d: float):
    """Define (once) a physics material at the world level and return its path.
    World-level materials sit OUTSIDE the per-env cloned subtree, so every
    clone's binding relationship keeps pointing at the same absolute prim."""
    from pxr import UsdPhysics, UsdShade

    path = f"/World/PhysMats/{name}"
    if not stage.GetPrimAtPath(path).IsValid():
        mat = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(mu_s))
        api.CreateDynamicFrictionAttr(float(mu_d))
        api.CreateRestitutionAttr(0.0)
    return path


def _bind_mat(stage, prim, mat_path: str) -> None:
    from pxr import UsdShade

    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float | None,
         orient=None, mat_path: str | None = None) -> None:
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
        if mat_path is not None:
            _bind_mat(stage, seg.GetPrim(), mat_path)


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


def _spawn_cubby(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC cubby. Origin = centre of the interior floor rectangle at the
    BOTTOM of the floor slab (sits on the deck). Mouth faces local +x.
    Floor slab (slick, incl. the apron past the mouth), back wall, two side
    walls, centre rib splitting the interior into two lanes, roof."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 15.0, kinematic=True)

    c = cfg
    co = c.contact_offset
    slick = _shared_mat(stage, "slick", c.mu_slick[0], c.mu_slick[1])
    hd = c.interior_d / 2                       # mouth at +hd, back inner at -hd
    hw = c.interior_w / 2
    ow = c.interior_w + 2 * c.wall_t            # outer width
    z0, z1 = c.floor_t, c.floor_t + c.interior_h  # floor top / roof underside
    # floor: interior part + apron part (both slick; apron paler as a cue)
    x_lo = -hd - c.back_t
    _box(stage, f"{prim_path}/floor", (hd - x_lo, ow, c.floor_t),
         ((hd + x_lo) / 2, 0.0, c.floor_t / 2), c.cubby_color, co, mat_path=slick)
    _box(stage, f"{prim_path}/apron", (c.apron_len, ow, c.floor_t),
         (hd + c.apron_len / 2, 0.0, c.floor_t / 2), c.apron_color, co, mat_path=slick)
    # back wall
    _box(stage, f"{prim_path}/back", (c.back_t, ow, c.interior_h),
         (-hd - c.back_t / 2, 0.0, (z0 + z1) / 2), c.cubby_color, co)
    # side walls (from the back to the mouth)
    for tag, sy in (("left", 1.0), ("right", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (hd - x_lo, c.wall_t, c.interior_h),
             ((hd + x_lo) / 2, sy * (hw + c.wall_t / 2), (z0 + z1) / 2),
             c.cubby_color, co)
    # centre rib (full height, splits the two lanes)
    _box(stage, f"{prim_path}/rib", (c.interior_d, c.rib_t, c.interior_h),
         (0.0, 0.0, (z0 + z1) / 2), c.rib_color, co)
    # roof
    _box(stage, f"{prim_path}/roof", (hd - x_lo, ow, c.roof_t),
         ((hd + x_lo) / 2, 0.0, z1 + c.roof_t / 2), c.roof_color, co)
    return root


def _spawn_stove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC stove: base plate + raised octagonal burner (grippy).
    Origin = plate BOTTOM centre (sits on the deck)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 4.0, kinematic=True)
    c = cfg
    co = c.contact_offset
    grip = _shared_mat(stage, "grip", c.mu_grip[0], c.mu_grip[1])
    _box(stage, f"{prim_path}/plate", (c.stove_plate, c.stove_plate, c.plate_t),
         (0.0, 0.0, c.plate_t / 2), c.stove_color, co)
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/pad_{tag}", (c.pad_across, c.pad_across, c.pad_h),
             (0.0, 0.0, c.plate_t + c.pad_h / 2), c.pad_color, co,
             orient=orient, mat_path=grip)
    return root


def _spawn_rake(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC rake. Origin = bar geometric centre. Bar along local x
    (z in [-bar_t/2, +bar_t/2]); hook FLANGE hangs DOWN at the +x end;
    grip block sits ON TOP at the -x end."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    c = cfg
    _rigid_root(root, c.rake_mass, kinematic=False, lin_damp=0.2, ang_damp=0.5,
                com=(-0.030, 0.0, 0.004), inertia=c.rake_inertia)
    co = c.contact_offset
    hx = c.bar_len / 2
    _box(stage, f"{prim_path}/bar", (c.bar_len, c.bar_w, c.bar_t),
         (0.0, 0.0, 0.0), c.rake_color, co)
    _box(stage, f"{prim_path}/flange", (c.flange_t, c.flange_w, c.flange_drop),
         (hx - c.flange_t / 2, 0.0, -c.bar_t / 2 - c.flange_drop / 2),
         c.flange_color, co)
    _box(stage, f"{prim_path}/grip", (0.036, c.bar_w, 0.026),
         (-hx + 0.018, 0.0, c.bar_t / 2 + 0.013), c.grip_color, co)
    return root


def _spawn_pot(prim_path: str, cfg: Any, color, translation=None, orientation=None):
    """DYNAMIC moka pot. Origin = base BOTTOM centre: octagonal base + waisted
    octagonal top + mushroom-head lid KNOB (post + wider head — the rake's
    catch feature). No side handle: the pot must fit its lane at any yaw.
    Base prims carry the smooth pot-base material (slide, don't tip)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    c = cfg
    m = c.pot_mass
    _rigid_root(root, m, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.036),
                inertia=tuple(m * u for u in c.pot_unit_inertia))
    co = c.contact_offset
    base_mat = _shared_mat(stage, "potbase", c.mu_pot[0], c.mu_pot[1])
    dark = c.pot_trim_color
    b, bh = c.pot_across, c.pot_base_h
    t, th = c.pot_top_across, c.pot_top_h
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/base_{tag}", (b, b, bh), (0.0, 0.0, bh / 2),
             color, co, orient=orient, mat_path=base_mat)
        _box(stage, f"{prim_path}/top_{tag}", (t, t, th), (0.0, 0.0, bh + th / 2),
             color, co, orient=orient)
    zb = bh + th
    _box(stage, f"{prim_path}/knob_post", (c.knob_post_s, c.knob_post_s, c.knob_post_h),
         (0.0, 0.0, zb + c.knob_post_h / 2), dark, co)
    _box(stage, f"{prim_path}/knob_head", (c.knob_head_s, c.knob_head_s, c.knob_head_h),
         (0.0, 0.0, zb + c.knob_post_h + c.knob_head_h / 2), dark, co)
    return root


def _spawn_pot_t(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    return _spawn_pot(prim_path, c, c.target_color, translation, orientation)


def _spawn_pot_w(prim_path: str, cfg: Any, translation=None, orientation=None):
    c = cfg.cfg
    return _spawn_pot(prim_path, c, c.wrong_color, translation, orientation)


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
class CubbyRakeSceneCfg(BaseCfg):
    """Config for `CubbyRakeStoveScene`. Honesty physics asserted in __post_init__."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    reach_dx: float = tunable(0.040)        # in-cubby mouth-ward drag that counts as REACHED (m)
    out_margin: float = tunable(0.035)      # pot centre past the mouth plane = EXTRACTED (m)
    pad_xy_tol: float = tunable(0.040)      # pot centre within this of the burner centre (m)
    upright_max_deg: float = tunable(15.0)  # pot up-axis within this of vertical
    settle_speed: float = tunable(0.10)     # payload lin speed at rest (m/s)
    latch_speed: float = tunable(0.10)      # calm gate for the latches (m/s)
    rake_clear_r: float = tunable(0.10)     # rake bar must stay this far (xy) from the burner (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pot_x_jitter: float = tunable(0.012)    # pot depth jitter along the lane (+/- m)
    pot_y_jitter: float = tunable(0.002)    # pot cross-lane jitter (+/- m)
    pot_yaw_deg: float = tunable(180.0)     # pot free yaw (+/- deg)
    rake_jitter: float = tunable(0.015)     # rake centre xy jitter (+/- m)
    rake_yaw_deg: float = tunable(6.0)      # rake lying-yaw jitter (+/- deg)
    # (which LANE the copper pot spawns in, and the rake's 180-deg end flip, are coin flips)

    # --- info: layout (world nominal; deck top = z 0.024) ----------------------------------------
    deck_size: tuple = info((0.90, 0.75, 0.024))
    deck_pos: tuple = info((0.45, 0.0))
    cubby_pos: tuple = info((0.24, 0.0))    # interior-floor centre; mouth faces +x
    stove_pos: tuple = info((0.66, -0.22))  # burner plate centre
    rake_pos: tuple = info((0.60, 0.24))    # rake bar centre (bar ~ along world x)
    # --- info: cubby -----------------------------------------------------------------------------
    lane_w: float = info(0.085)             # each lane's clear width
    rib_t: float = info(0.012)              # centre rib thickness
    wall_t: float = info(0.012)
    back_t: float = info(0.012)
    interior_d: float = info(0.260)         # lane depth, back inner face -> mouth plane
    interior_h: float = info(0.150)         # floor top -> roof underside
    floor_t: float = info(0.008)
    roof_t: float = info(0.012)
    apron_len: float = info(0.100)          # slick floor extending past the mouth
    pot_lane_x: float = info(-0.050)        # pot centre, cubby-local x (mouth at +0.130)
    # --- info: stove -----------------------------------------------------------------------------
    stove_plate: float = info(0.140)
    plate_t: float = info(0.005)
    pad_across: float = info(0.096)         # burner octagon across-flats
    pad_h: float = info(0.006)
    # --- info: rake ------------------------------------------------------------------------------
    bar_len: float = info(0.400)
    bar_w: float = info(0.040)
    bar_t: float = info(0.008)
    flange_w: float = info(0.040)
    flange_t: float = info(0.010)
    flange_drop: float = info(0.022)        # hook depth below the bar underside
    rake_mass: float = info(0.30)
    rake_inertia: tuple = info((0.00025, 0.0045, 0.0046))
    # --- info: pots ------------------------------------------------------------------------------
    pot_across: float = info(0.050)         # octagon across-flats (base)
    pot_base_h: float = info(0.044)
    pot_top_across: float = info(0.040)
    pot_top_h: float = info(0.030)
    knob_post_s: float = info(0.012)
    knob_post_h: float = info(0.006)
    knob_head_s: float = info(0.024)        # mushroom head (the rake's catch)
    knob_head_h: float = info(0.008)
    pot_mass: float = info(0.280)
    pot_unit_inertia: tuple = info((0.00080, 0.00080, 0.00042))  # per kg
    # --- info: friction (PhysX pair-average combine) ---------------------------------------------
    mu_slick: tuple = info((0.04, 0.035))   # cubby floor + apron
    mu_pot: tuple = info((0.14, 0.12))      # pot base facets
    mu_grip: tuple = info((0.95, 0.90))     # burner plate top
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.55, 0.44, 0.30))
    cubby_color: tuple = info((0.32, 0.26, 0.20))
    rib_color: tuple = info((0.42, 0.35, 0.27))
    roof_color: tuple = info((0.28, 0.23, 0.18))
    apron_color: tuple = info((0.72, 0.66, 0.55))
    stove_color: tuple = info((0.25, 0.26, 0.29))
    pad_color: tuple = info((0.10, 0.10, 0.11))
    rake_color: tuple = info((0.50, 0.36, 0.20))
    flange_color: tuple = info((0.85, 0.20, 0.15))
    grip_color: tuple = info((0.08, 0.08, 0.09))
    target_color: tuple = info((0.70, 0.40, 0.25))  # COPPER = the right pot
    wrong_color: tuple = info((0.58, 0.59, 0.63))   # STEEL = the distractor
    pot_trim_color: tuple = info((0.05, 0.05, 0.06))
    # rubric weights (0.20 + 0.30 + 0.25 = 0.75 = the non-success cap)
    w_reach: float = info(0.20)
    w_out: float = info(0.30)
    w_pad: float = info(0.25)
    score_cap: float = info(0.75)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def interior_w(self) -> float:
        return 2 * self.lane_w + self.rib_t

    @property
    def mouth_x(self) -> float:
        """Mouth plane, cubby-local x."""
        return self.interior_d / 2

    @property
    def lane_y(self) -> float:
        """Lane centreline offset, cubby-local |y|."""
        return (self.lane_w + self.rib_t) / 2

    @property
    def knob_top(self) -> float:
        """Pot knob top above the pot base bottom."""
        return self.pot_base_h + self.pot_top_h + self.knob_post_h + self.knob_head_h

    @property
    def head_bot(self) -> float:
        """Knob head UNDERSIDE above the pot base bottom (the hook shelf)."""
        return self.pot_base_h + self.pot_top_h + self.knob_post_h

    @property
    def pot_circ_r(self) -> float:
        return self.pot_across * math.sqrt(2.0) / 2

    @property
    def floor_top(self) -> float:
        """Cubby floor top, cubby-local z (pots and apron rest here)."""
        return self.floor_t

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the physical claims the task rests on).
        co2 = 2 * self.contact_offset
        # (1) the pot fits its lane at ANY yaw, with clearance at max jitter.
        # pot_circ_r is the union-of-two-squares STAR corner radius (the proud
        # corners of the 45-deg crossed boxes), the true worst-case footprint.
        assert self.lane_w / 2 - (self.pot_circ_r + self.pot_y_jitter) >= 0.0045, \
            "pot star corners must clear the lane walls at any yaw + jitter"
        # (2) insertion corridor: bar (with the flange hanging) rides OVER the
        # knob and UNDER the roof with real margins
        ins_gap = 0.008  # flange-bottom clearance over the knob during insertion
        need = self.knob_top + ins_gap + self.flange_drop + self.bar_t
        assert self.interior_h - need >= 0.012 + co2, \
            f"insertion corridor too tight: roof margin {self.interior_h - need:.3f}"
        # (3) engaged: the flange hooks the mushroom head while the bar stays
        # clear above the knob
        eng_under = 0.004  # flange bottom below the head underside when engaged
        bar_gap = self.flange_drop - (self.knob_head_h + eng_under)
        assert bar_gap >= 0.006 + co2, \
            f"engaged bar must clear the knob top by >=6mm net (got {bar_gap:.3f})"
        # (4) the flange fits the gap between the deepest knob and the back wall
        knob_back = self.pot_lane_x - self.pot_x_jitter - self.knob_head_s / 2
        assert knob_back - (-self.interior_d / 2) >= self.flange_t + 0.008 + co2, \
            "flange must fit behind the deepest-spawned pot's knob"
        # (5) drag slides, never tips: pair-averaged floor/pot friction is at
        # least 2x below the tipping threshold at the highest catch point
        mu_eff = (self.mu_slick[0] + self.mu_pot[0]) / 2
        mu_tip = (self.pot_across / 2 - 0.003) / self.knob_top
        assert mu_eff * 2.0 <= mu_tip, \
            f"drag must be sub-tipping with 2x margin: eff {mu_eff:.3f} vs tip {mu_tip:.3f}"
        # ... and the burner holds the delivered pot firmly
        assert (self.mu_grip[0] + self.mu_pot[0]) / 2 >= 0.30, "burner must grip the pot"
        # (6) the rake reaches the deepest knob with the grip still well
        # OUTSIDE the mouth (a hand never needs to enter)
        reach = self.mouth_x - (self.pot_lane_x - self.pot_x_jitter)
        assert self.bar_len >= reach + 0.10, \
            "rake must reach the deepest pot with >=10cm of bar outside the mouth"
        # (7) the extracted pot rests fully ON the apron
        assert self.apron_len >= self.out_margin + self.pot_circ_r + 0.020, \
            "apron must be long enough to receive the extracted pot"
        # (8) bar and flange fit the lane laterally; flange catches the head
        # across the full jitter band
        assert max(self.bar_w, self.flange_w) <= self.lane_w - 0.008 - co2, \
            "bar/flange must fit the lane width"
        assert self.flange_w >= self.knob_head_s + 2 * self.pot_y_jitter + 0.008, \
            "flange must span the knob head across the cross-lane jitter band"
        # (9) the pot spawn band sits fully inside the lane depth, with the
        # flange descent gap behind it (see 4) and clearance to the mouth
        assert self.pot_lane_x - self.pot_x_jitter - self.pot_circ_r \
            > -self.interior_d / 2 + 0.004, "pot must not touch the back wall"
        assert self.pot_lane_x + self.pot_x_jitter + self.pot_circ_r \
            < self.mouth_x - 0.060, "pot must spawn well behind the mouth"
        # (10) the dragged pot passes under the roof at the mouth
        assert self.knob_top + 0.006 + co2 < self.interior_h, \
            "pot must pass under the roof edge on the way out"
        # (11) zone separations: stove clear of cubby+apron and of the rake lane
        apron_x_hi = self.cubby_pos[0] + self.mouth_x + self.apron_len
        assert self.stove_pos[0] - self.stove_plate / 2 > apron_x_hi + 0.020 \
            or abs(self.stove_pos[1]) - self.stove_plate / 2 \
            > self.interior_w / 2 + self.wall_t + 0.020, \
            "stove must sit clear of the cubby + apron footprint"
        assert abs(self.rake_pos[1] - self.stove_pos[1]) \
            > self.bar_w / 2 + self.rake_jitter + self.stove_plate / 2 + 0.030, \
            "rake spawn lane must stay clear of the stove"
        assert self.rake_pos[1] - self.bar_w / 2 - self.rake_jitter \
            > self.interior_w / 2 + self.wall_t + 0.020, \
            "rake spawn lane must stay clear of the cubby"
        # ... everything on the deck
        hxd, hyd = self.deck_size[0] / 2, self.deck_size[1] / 2
        assert self.rake_pos[0] + self.bar_len / 2 + self.rake_jitter \
            < self.deck_pos[0] + hxd - 0.010, "rake must stay on the deck"
        assert abs(self.stove_pos[1]) + self.stove_plate / 2 < hyd - 0.010
        assert self.cubby_pos[0] + self.mouth_x + self.apron_len \
            < self.deck_pos[0] + hxd - 0.010
        assert self.cubby_pos[0] - self.mouth_x - self.back_t \
            > self.deck_pos[0] - hxd + 0.010
        # (12) burner tolerance: a pot AT tolerance still stands on the pad
        assert self.pad_xy_tol + self.pot_across / 2 <= self.pad_across / 2 + 0.020, \
            "pad_xy_tol must keep the pot substantially on the burner"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("cubby_rake_stove")
class CubbyRakeStoveScene(BaseScene):
    cfg: CubbyRakeSceneCfg

    def __init__(self, cfg: CubbyRakeSceneCfg | None = None) -> None:
        super().__init__(cfg or CubbyRakeSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        deck = sim_utils.CuboidCfg(
            size=c.deck_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.7, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.deck_color),
        )

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
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=deck,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], c.deck_size[2] / 2)),
            ),
            "cubby": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cubby",
                spawn=_compound_spawner_cfg("cubby", _spawn_cubby, c, 15.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cubby_pos[0], c.cubby_pos[1], c.deck_top)),
            ),
            "stove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stove",
                spawn=_compound_spawner_cfg("stove", _spawn_stove, c, 4.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stove_pos[0], c.stove_pos[1], c.deck_top)),
            ),
            "rake": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rake",
                spawn=_compound_spawner_cfg("rake", _spawn_rake, c, c.rake_mass,
                                            kinematic=False, lin_damp=0.2, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rake_pos[0], c.rake_pos[1],
                         c.deck_top + c.bar_t / 2 + c.flange_drop + 0.002)),
            ),
            "pot_t": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PotT",
                spawn=_compound_spawner_cfg("pot_t", _spawn_pot_t, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cubby_pos[0] + c.pot_lane_x, c.cubby_pos[1] + c.lane_y,
                         c.deck_top + c.floor_t + 0.001)),
            ),
            "pot_w": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PotW",
                spawn=_compound_spawner_cfg("pot_w", _spawn_pot_w, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cubby_pos[0] + c.pot_lane_x, c.cubby_pos[1] - c.lane_y,
                         c.deck_top + c.floor_t + 0.001)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external wrenches (the solve's rake servo, the smoke probes)
                # act cleanly only with per-iteration application
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
        self.cubby: RigidObject = env.iscene["cubby"]
        self.stove: RigidObject = env.iscene["stove"]
        self.rake: RigidObject = env.iscene["rake"]
        self.pot_t: RigidObject = env.iscene["pot_t"]
        self.pot_w: RigidObject = env.iscene["pot_w"]
        self.env_origins = env.iscene.env_origins
        # latches (partial credit survives transients; success is judged live)
        self._l_reach = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_out = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_pad = torch.zeros(n, dtype=torch.bool, device=dev)
        self._x0 = torch.zeros(n, device=dev)  # copper pot spawn x, cubby-local

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: cubby + stove written at their fixed poses, the COPPER
        pot's LANE flipped per episode (steel in the other lane), both pots
        jittered (depth, cross-lane, free yaw), rake jittered on the counter
        with a 50% end flip; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, device=dev)  # burn: first post-seed draw is degenerate

        def write(body, pos, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            body.write_root_state_to_sim(st, env_ids)

        # --- fixed furniture ---
        write(self.cubby, torch.tensor([c.cubby_pos[0], c.cubby_pos[1], c.deck_top],
                                       device=dev).expand(m, 3))
        write(self.stove, torch.tensor([c.stove_pos[0], c.stove_pos[1], c.deck_top],
                                       device=dev).expand(m, 3))

        # --- pots: copper LANE is a coin flip; each jittered with free yaw ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        x0 = torch.zeros(m, device=dev)
        for body, s, is_t in ((self.pot_t, side, True), (self.pot_w, -side, False)):
            px = c.pot_lane_x + (torch.rand(m, device=dev) * 2 - 1) * c.pot_x_jitter
            ppos = torch.zeros(m, 3, device=dev)
            ppos[:, 0] = c.cubby_pos[0] + px
            ppos[:, 1] = c.cubby_pos[1] + s * c.lane_y \
                + (torch.rand(m, device=dev) * 2 - 1) * c.pot_y_jitter
            ppos[:, 2] = c.deck_top + c.floor_t + 0.001
            pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pot_yaw_deg)
            write(body, ppos, _qz(pyaw))
            if is_t:
                x0 = px
        self._x0[env_ids] = x0

        # --- rake: lying on the deck ~along x, jittered, 50% end flip ---
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rake_yaw_deg)
        flip = (torch.rand(m, device=dev) < 0.5).float() * math.pi
        rpos = torch.zeros(m, 3, device=dev)
        rpos[:, 0] = c.rake_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rake_jitter
        rpos[:, 1] = c.rake_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rake_jitter
        rpos[:, 2] = c.deck_top + c.bar_t / 2 + c.flange_drop + 0.002
        write(self.rake, rpos, _qz(ryaw + flip))

        for latch in (self._l_reach, self._l_out, self._l_pad):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cubby": self.cubby.data.root_state_w[env_ids].clone(),
            "stove": self.stove.data.root_state_w[env_ids].clone(),
            "rake": self.rake.data.root_state_w[env_ids].clone(),
            "pot_t": self.pot_t.data.root_state_w[env_ids].clone(),
            "pot_w": self.pot_w.data.root_state_w[env_ids].clone(),
            "l_reach": self._l_reach[env_ids].clone(),
            "l_out": self._l_out[env_ids].clone(),
            "l_pad": self._l_pad[env_ids].clone(),
            "x0": self._x0[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("cubby", "stove", "rake", "pot_t", "pot_w"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self._l_reach[env_ids] = state["l_reach"]
        self._l_out[env_ids] = state["l_out"]
        self._l_pad[env_ids] = state["l_pad"]
        self._x0[env_ids] = state["x0"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden COUNTER ({c.deck_size[0] * 100:.0f} x {c.deck_size[1] * 100:.0f} cm "
            f"slab) lies on the ground. At its rear-centre stands a low roofed "
            f"CUBBY: two side-by-side lanes, each {c.lane_w * 1000:.0f} mm wide, "
            f"{c.interior_d * 1000:.0f} mm deep and {c.interior_h * 1000:.0f} mm tall, "
            f"separated by a centre rib, open only at the front; the cubby floor is "
            f"SLICK and extends {c.apron_len * 1000:.0f} mm past the mouth as a pale "
            f"APRON. Deep inside the lanes — one pot per lane, ~{-c.pot_lane_x * 1000 + c.mouth_x * 1000:.0f} mm "
            f"behind the mouth — stand two moka pots of identical shape "
            f"({c.pot_across * 1000:.0f} mm across, {c.knob_top * 1000:.0f} mm tall with a "
            f"mushroom-head lid KNOB): one COPPER, one STEEL; which lane the copper "
            f"one starts in varies per episode. The lanes are too narrow and the "
            f"roof too low for any hand or gripper to reach a pot. On the open "
            f"counter lie a burner STOVE (raised dark octagonal plate, grippy top, "
            f"front-right) and a long-handled RAKE (front-left): a "
            f"{c.bar_len * 100:.0f} cm flat bar with a red HOOK FLANGE "
            f"({c.flange_drop * 1000:.0f} mm drop) at its far end and a grip block at "
            f"its near end.\n"
            f"Goal: put the COPPER moka pot on the burner plate. The only way to "
            f"free it is with the rake: slide the bar nose-first into the copper "
            f"pot's lane riding ABOVE the pot, lower it so the flange drops into "
            f"the gap behind the knob and hooks the knob head, then DRAG the pot "
            f"down the lane and out of the mouth onto the apron (the slick floor "
            f"lets it slide without tipping). Then set the freed pot upright on "
            f"the burner plate and withdraw the rake. Success: the copper pot "
            f"upright and at rest on the burner plate, the steel pot still inside "
            f"its lane, the rake at least {c.rake_clear_r * 100:.0f} cm clear of the "
            f"burner, and everything at rest. The steel pot on the burner, a pot "
            f"tipped over, a pot left on the apron, disturbing the steel pot out "
            f"of the cubby, or the rake left against the burner does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Use the rake to pull the copper moka pot out of its cubby lane: slide "
            "the bar in above the pot, hook the red flange behind its knob, drag "
            "it out onto the apron, then place it upright on the black burner "
            "plate and set the rake aside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def pot_local(self, body) -> torch.Tensor:
        """(N,3) pot position in the cubby frame (cubby is axis-aligned)."""
        return body.data.root_pos_w - self.cubby.data.root_pos_w

    def pot_up(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)[:, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def in_cubby(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the roofed interior airspace."""
        c = self.cfg
        d = self.pot_local(body)
        return (d[:, 0] > -c.interior_d / 2) & (d[:, 0] < c.mouth_x) \
            & (d[:, 1].abs() < c.interior_w / 2) \
            & (d[:, 2] > 0.0) & (d[:, 2] < c.floor_t + c.interior_h)

    def pot_out(self) -> torch.Tensor:
        """(N,) bool: the COPPER pot fully outside the mouth, upright, resting
        at floor level on the apron."""
        c = self.cfg
        d = self.pot_local(self.pot_t)
        dz = d[:, 2] - c.floor_top
        return (d[:, 0] > c.mouth_x + c.out_margin) & (d[:, 1].abs() < c.interior_w / 2) \
            & (dz > -0.004) & (dz < 0.012) & self.pot_up(self.pot_t)

    def pot_on_pad(self) -> torch.Tensor:
        """(N,) bool: the COPPER pot upright on the burner plate."""
        c = self.cfg
        sp = self.stove.data.root_pos_w
        d = (self.pot_t.data.root_pos_w[:, :2] - sp[:, :2]).norm(dim=-1)
        dz = self.pot_t.data.root_pos_w[:, 2] - (sp[:, 2] + c.plate_t + c.pad_h)
        return (d <= c.pad_xy_tol) & (dz > -0.006) & (dz < 0.014) & self.pot_up(self.pot_t)

    def rake_clear(self) -> torch.Tensor:
        """(N,) bool: the rake BAR (as an xy segment) at least rake_clear_r
        from the burner centre."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        q = self.rake.data.root_quat_w
        n = q.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(n, 3)
        u = quat_apply(q, ex)[:, :2]
        p0 = self.rake.data.root_pos_w[:, :2] - u * (c.bar_len / 2)
        p1 = self.rake.data.root_pos_w[:, :2] + u * (c.bar_len / 2)
        sp = self.stove.data.root_pos_w[:, :2]
        seg = p1 - p0
        t = ((sp - p0) * seg).sum(-1) / seg.pow(2).sum(-1).clamp(min=1e-9)
        near = p0 + seg * t.clamp(0.0, 1.0).unsqueeze(-1)
        return (sp - near).norm(dim=-1) > c.rake_clear_r

    def settled(self) -> torch.Tensor:
        """(N,) bool: rake and both pots at rest."""
        c = self.cfg
        ok = self.rake.data.root_ang_vel_w.norm(dim=-1) < 0.5
        for b in (self.rake, self.pot_t, self.pot_w):
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return ok

    def _update_latches(self) -> None:
        c = self.cfg
        d = self.pot_local(self.pot_t)
        self._l_reach |= self.in_cubby(self.pot_t) & self.pot_up(self.pot_t) \
            & (d[:, 0] - self._x0 >= c.reach_dx)
        calm = self.pot_t.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._l_out |= self.pot_out() & calm
        self._l_pad |= self.pot_on_pad() & calm

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, all live: copper pot upright at rest on the burner plate,
        steel pot still inside the cubby, rake clear of the burner, everything
        at rest, finite."""
        self._update_latches()
        finite = torch.isfinite(self.rake.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.pot_t.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.pot_w.data.root_pos_w).all(dim=-1)
        return self.pot_on_pad() & self.in_cubby(self.pot_w) & self.rake_clear() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*reached + 0.30*extracted + 0.25*on_stove
        (all latched; ~0 for doing nothing), capped at 0.75 — and exactly 1.0
        iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_reach * self._l_reach.float() + c.w_out * self._l_out.float()
                + c.w_pad * self._l_pad.float()).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="cubby_rake_stove", robot="null"))
