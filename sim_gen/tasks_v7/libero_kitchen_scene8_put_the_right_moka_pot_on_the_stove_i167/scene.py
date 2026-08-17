"""StoveProdScene — prop the stove lid open with the rod, then stove the RIGHT pot
(sim_gen task `libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i167`).

Derived from libero_90/libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove,
but STRATEGICALLY different. The seed is a pick-and-place with a positional
identity twist: two IDENTICAL moka pots, grab the RIGHT one and set it on a flat,
always-accessible stove — success is a pose predicate satisfied the instant the
pot rests there. Here the stove is a deep TUB sealed by a heavy gravity-closing
LID, and the goal surface does not exist as a destination until the solver
BUILDS temporary falsework to hold the mechanism open:

  (1) lift the lid (it swings on a rear hinge, limited to 80 degrees — always
      short of over-center, so gravity slams it shut the moment it is let go);
  (2) while holding it open, stand the loose PROP ROD in the small square WELL
      on the tub floor and ease the lid down onto the rod's tip so it seats in
      the CAPTURE POCKET on the lid's underside — only this two-point brace
      (base in well + tip in pocket) lets the lid stay open hands-free;
  (3) only now does the burner exist as a reachable surface: through the
      propped-open mouth, place the COPPER pot (not the identical-shaped steel
      one) upright on the burner pad, and step away with everything at rest.

A solver therefore needs a different PLAN from the seed (an ordered
construction: open a mechanism, erect falsework under it, THEN pick-and-place
through the opening it protects — with the pot-placement impossible before the
prop stands) and different code structure (the load-bearing predicate is a
braced mechanism, not an object pose alone).

Assets are fully procedural (native PhysX box colliders):
  - deck (KINEMATIC): wooden counter slab 900 x 750 x 24 mm on the ground.
  - tub (KINEMATIC): the stove body — floor + four walls (inner 300 x 300 mm,
    54 mm deep), a raised octagonal BURNER PAD on its floor, and a square
    SOCKET WELL (24 mm inner, 12 mm walls) that locates the rod's base.
  - lid (DYNAMIC, revolute Y joint at the tub's rear top edge, limits
    [-80 deg, +0.5 deg]): a 316 mm panel with a top handle bar and an
    underside CAPTURE POCKET (fenced cavity at 200 mm from the hinge) that
    locates the rod's tip. Its CoM 158 mm from the hinge makes gravity close
    it from ANY legal angle (cos 80 > 0: no over-center parking).
  - rod (DYNAMIC): a 14 x 14 x 232 mm prop rod, spawned lying loose on the
    counter. Standing in the well it braces the lid at ~63 degrees.
  - pot_t / pot_w (DYNAMIC): two same-shape octagonal moka pots — COPPER
    (target) and STEEL (distractor). Which SIDE each spawns on is randomized
    per episode, so identity is by appearance, not by position.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.20 * lid_open   — lid ever open >= 50 deg (latched)
  0.30 * propped    — lid open + calm with the rod braced (base in well, tip
                      in pocket, rod at rest) — hands-free falsework (latched)
  0.25 * pot_set    — the COPPER pot ever at rest upright on the burner pad
                      (latched)
  1.0 iff success() — live: lid open and calm, rod braced (base in well, tip
                      in pocket), copper pot upright on the burner, steel pot
                      NOT inside the tub, everything at rest, finite.
                      Non-success capped at 0.75.

The rod clauses are the anti-pinning backstop: a gripper (or external wrench)
holding the lid open passes the angle clause but not base-in-well + tip-in-
pocket; and a pot shoved against the rear wall to prop the lid steeply parks
the pocket beyond the rod's reach (asserted in __post_init__), besides failing
the placement clauses.

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


def _spawn_tub(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC stove tub. Origin = floor BOTTOM centre (sits on the deck):
    floor slab, four walls, raised octagonal burner pad, square socket well."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 12.0, kinematic=True)

    c = cfg
    co = c.contact_offset
    col = c.tub_color
    hx = c.tub_inner / 2  # 0.150
    wt = c.wall_t
    outer = c.tub_inner + 2 * wt
    wall_h = c.wall_top - c.floor_t
    wz = c.floor_t + wall_h / 2
    _box(stage, f"{prim_path}/floor", (outer, outer, c.floor_t),
         (0.0, 0.0, c.floor_t / 2), col, co)
    for tag, sx in (("back", -1.0), ("front", 1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (wt, outer, wall_h),
             (sx * (hx + wt / 2), 0.0, wz), col, co)
    for tag, sy in (("left", 1.0), ("right", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (c.tub_inner, wt, wall_h),
             (0.0, sy * (hx + wt / 2), wz), col, co)
    # burner pad: raised octagon on the floor
    px, py = c.pad_pos
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/pad_{tag}", (c.pad_across, c.pad_across, c.pad_h),
             (px, py, c.floor_t + c.pad_h / 2), c.pad_color, co, orient=orient)
    # socket well: four short walls around a square pit that locates the rod base
    wxc, wyc = c.well_x, c.well_y
    wi, wwt, wwh = c.well_inner, c.well_wall_t, c.well_wall_h
    wcz = c.floor_t + wwh / 2
    for tag, sx in (("xp", 1.0), ("xn", -1.0)):
        _box(stage, f"{prim_path}/well_{tag}", (wwt, wi + 2 * wwt, wwh),
             (wxc + sx * (wi / 2 + wwt / 2), wyc, wcz), c.well_color, co)
    for tag, sy in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/well_{tag}", (wi, wwt, wwh),
             (wxc, wyc + sy * (wi / 2 + wwt / 2), wcz), c.well_color, co)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC stove lid. Origin = the HINGE line (rear top edge of the tub):
    panel extends along local +x, handle bar on top near the far edge, fenced
    CAPTURE POCKET on the underside at pocket_r from the hinge."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    c = cfg
    _rigid_root(root, c.lid_mass, kinematic=False, lin_damp=0.0,
                ang_damp=c.lid_ang_damp, com=(c.lid_len / 2, 0.0, c.lid_t / 2),
                inertia=c.lid_inertia)

    co = c.contact_offset
    _box(stage, f"{prim_path}/panel", (c.lid_len, c.lid_w, c.lid_t),
         (c.lid_len / 2, 0.0, c.lid_t / 2), c.lid_color, co)
    _box(stage, f"{prim_path}/handle", (0.020, 0.080, 0.014),
         (c.lid_len - 0.026, 0.0, c.lid_t + 0.007), c.handle_color, co)
    # capture pocket: fenced cavity on the underside, ASYMMETRIC about the tip
    # seat (pocket_r). The loaded tip slides FORE along the inclined panel (that
    # lowers the lid — an energy runaway), so the FORE wall is close and DEEP:
    # it is the catch that arrests the slide and makes the brace statically
    # stable. The AFT wall sits far back and SHALLOW because the rod shaft
    # slants aft in the lid frame while the tip is inside the cavity.
    piy, pt = c.pocket_inner_y, c.pocket_wall_t
    x_fore = c.pocket_r + c.pocket_fore + pt / 2
    x_aft = c.pocket_r - c.pocket_aft - pt / 2
    _box(stage, f"{prim_path}/pocket_fore", (pt, piy + 2 * pt, c.pocket_depth),
         (x_fore, c.pocket_y, -c.pocket_depth / 2), c.pocket_color, co)
    _box(stage, f"{prim_path}/pocket_aft", (pt, piy + 2 * pt, c.pocket_aft_depth),
         (x_aft, c.pocket_y, -c.pocket_aft_depth / 2), c.pocket_color, co)
    span = (x_fore - x_aft) + pt
    xc = (x_fore + x_aft) / 2
    for tag, sy in (("lft", 1.0), ("rgt", -1.0)):
        _box(stage, f"{prim_path}/pocket_{tag}", (span, pt, c.pocket_depth),
             (xc, c.pocket_y + sy * (piy / 2 + pt / 2), -c.pocket_depth / 2),
             c.pocket_color, co)
    return root


def _spawn_rod(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC prop rod: square-section bar along local +z, origin at its BASE
    bottom centre; a small painted tip band marks the pocket end."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    c = cfg
    _rigid_root(root, c.rod_mass, kinematic=False, lin_damp=0.2, ang_damp=0.5,
                com=(0.0, 0.0, c.rod_len / 2), inertia=c.rod_inertia)
    co = c.contact_offset
    _box(stage, f"{prim_path}/bar", (c.rod_s, c.rod_s, c.rod_len),
         (0.0, 0.0, c.rod_len / 2), c.rod_color, co)
    # tip band: VISUAL ONLY (keeps the collider a single clean box)
    _box(stage, f"{prim_path}/tipband", (c.rod_s + 0.002, c.rod_s + 0.002, 0.010),
         (0.0, 0.0, c.rod_len - 0.010), (0.85, 0.20, 0.15), None)
    return root


def _spawn_pot(prim_path: str, cfg: Any, color, translation=None, orientation=None):
    """DYNAMIC moka pot. Origin = base BOTTOM centre: octagonal base + waisted
    octagonal top + lid knob + side handle along local +x."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    c = cfg
    m = c.pot_mass
    _rigid_root(root, m, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.040),
                inertia=tuple(m * u for u in c.pot_unit_inertia))

    co = c.contact_offset
    dark = c.pot_trim_color
    b, bh = c.pot_across, 0.048
    t, th = 0.050, 0.034
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/base_{tag}", (b, b, bh), (0.0, 0.0, bh / 2), color, co,
             orient=orient)
        _box(stage, f"{prim_path}/top_{tag}", (t, t, th), (0.0, 0.0, bh + th / 2), color, co,
             orient=orient)
    _box(stage, f"{prim_path}/knob", (0.014, 0.014, 0.008),
         (0.0, 0.0, bh + th + 0.004), dark, co)
    _box(stage, f"{prim_path}/handle", (0.036, 0.012, 0.020),
         (0.044, 0.0, 0.058), dark, co)
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
class StovePropSceneCfg(BaseCfg):
    """Config for `StovePropScene`. Honesty physics asserted in __post_init__."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    open_min_deg: float = tunable(50.0)     # lid counts as OPEN above this (deg)
    lid_calm_omega: float = tunable(0.20)   # quasi-static gate on the lid (rad/s)
    settle_speed: float = tunable(0.10)     # payload lin speed at rest (m/s)
    latch_speed: float = tunable(0.10)      # calm gate for the latches (m/s)
    pot_xy_tol: float = tunable(0.045)      # pot centre within this of the pad centre (m)
    upright_max_deg: float = tunable(15.0)  # pot up-axis within this of vertical
    tip_tol_x: float = tunable(0.026)       # rod tip lid-local |x - pocket_r| bound
    tip_tol_y: float = tunable(0.018)       # rod tip lid-local |y - pocket_y| bound
    tip_z_band: tuple = tunable((-0.022, 0.004))  # rod tip lid-local z band (in the pocket)
    well_tol: float = tunable(0.011)        # rod base |xy - well centre| bound (each axis)
    well_dz_band: tuple = tunable((-0.004, 0.016))  # rod base z above the tub floor

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pot_jitter: float = tunable(0.020)      # each pot xy jitter in its zone (+/- m)
    pot_yaw_deg: float = tunable(180.0)     # pot free yaw (+/- deg)
    rod_jitter: float = tunable(0.020)      # rod centre xy jitter (+/- m)
    rod_yaw_deg: float = tunable(8.0)       # rod lying-yaw jitter about its axis line (+/- deg)
    # (which SIDE the copper pot spawns on and the rod's 180-deg flip are coin flips)

    # --- info: layout (world nominal; deck top = z 0.024) ----------------------------------------
    deck_size: tuple = info((0.90, 0.75, 0.024))
    deck_pos: tuple = info((0.45, 0.0))
    tub_pos: tuple = info((0.38, 0.0))      # FIXED: the lid joint anchors to this kinematic body
    rod_pos: tuple = info((0.61, 0.0))      # rod lying centre (bar along world y)
    pot_x: float = info(0.73)               # both pot zones at this x
    pot_y: float = info(0.20)               # zones at y = +/- this (sides randomized)
    # --- info: tub -------------------------------------------------------------------------------
    tub_inner: float = info(0.300)          # inner cavity (x and y)
    wall_t: float = info(0.012)
    floor_t: float = info(0.008)
    wall_top: float = info(0.062)           # wall top above the tub origin (= hinge height)
    pad_pos: tuple = info((0.05, 0.06))     # burner pad centre, tub-local
    pad_across: float = info(0.096)         # octagon across-flats
    pad_h: float = info(0.006)
    well_y: float = info(-0.09)             # socket well centre y, tub-local
    well_inner: float = info(0.020)
    well_wall_t: float = info(0.008)
    well_wall_h: float = info(0.022)
    # --- info: lid -------------------------------------------------------------------------------
    lid_len: float = info(0.316)            # hinge -> far edge
    lid_w: float = info(0.316)
    lid_t: float = info(0.008)
    lid_mass: float = info(0.45)
    lid_inertia: tuple = info((0.0034, 0.0038, 0.0071))
    lid_ang_damp: float = info(0.5)
    lid_limit_deg: float = info(80.0)       # opening stop — SHORT of over-center
    lid_slack_deg: float = info(0.5)        # closing-side joint slack
    pocket_r: float = info(0.200)           # tip SEAT along the lid from the hinge
    pocket_y: float = info(-0.09)
    pocket_aft: float = info(0.040)         # aft (hinge-side) inner half-aperture
    pocket_fore: float = info(0.012)        # fore inner half-aperture (the catch gap)
    pocket_inner_y: float = info(0.032)
    pocket_wall_t: float = info(0.008)
    pocket_depth: float = info(0.030)       # fore + side walls: DEEP (the load catch)
    pocket_aft_depth: float = info(0.012)   # aft wall: SHALLOW (rod-shaft clearance)
    # --- info: rod / pots ------------------------------------------------------------------------
    rod_s: float = info(0.014)
    rod_len: float = info(0.232)
    rod_mass: float = info(0.060)
    rod_inertia: tuple = info((0.00027, 0.00027, 0.000004))
    pot_across: float = info(0.064)         # octagon across-flats (base)
    pot_h: float = info(0.090)
    pot_mass: float = info(0.280)
    pot_unit_inertia: tuple = info((0.00105, 0.00105, 0.00068))  # per kg
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.55, 0.44, 0.30))
    tub_color: tuple = info((0.28, 0.29, 0.32))
    pad_color: tuple = info((0.12, 0.12, 0.13))
    well_color: tuple = info((0.75, 0.65, 0.20))
    lid_color: tuple = info((0.45, 0.48, 0.52))
    handle_color: tuple = info((0.08, 0.08, 0.09))
    pocket_color: tuple = info((0.75, 0.65, 0.20))
    rod_color: tuple = info((0.50, 0.36, 0.20))
    target_color: tuple = info((0.70, 0.40, 0.25))  # COPPER = the right pot
    wrong_color: tuple = info((0.58, 0.59, 0.63))   # STEEL = the distractor
    pot_trim_color: tuple = info((0.05, 0.05, 0.06))
    # rubric weights (0.20 + 0.30 + 0.25 = 0.75 = the non-success cap)
    w_open: float = info(0.20)
    w_prop: float = info(0.30)
    w_pot: float = info(0.25)
    score_cap: float = info(0.75)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def hinge_x(self) -> float:
        """Hinge line x, tub-local (rear inner top edge)."""
        return -self.tub_inner / 2

    @property
    def hinge_z(self) -> float:
        return self.wall_top

    @property
    def prop_angle(self) -> float:
        """Lid opening angle (rad) at which the rod, standing plumb in the well,
        braces the lid: tip height = pocket underside height."""
        return math.asin((self.floor_t + self.rod_len - self.hinge_z) / self.pocket_r)

    @property
    def well_x(self) -> float:
        """Socket well centre x, tub-local: plumb below the pocket at prop_angle."""
        return self.hinge_x + self.pocket_r * math.cos(self.prop_angle)

    @property
    def catch_angle(self) -> float:
        """Lid angle (rad) at which the loaded rod tip, sliding fore along the
        panel underside, is arrested by the deep fore fence wall — the brace's
        actual resting angle (a couple of degrees below the plumb prop_angle)."""
        r_t = self.pocket_r + self.pocket_fore - self.rod_s / 2  # tip centre at catch
        bx, bz = self.well_x, self.floor_t
        for i in range(1, 400):  # rod tilts fore about its base, tip on the panel
            u = math.radians(i * 0.05)
            tx = bx + self.rod_len * math.sin(u) - self.hinge_x
            tz = bz + self.rod_len * math.cos(u) - self.hinge_z
            if math.hypot(tx, tz) >= r_t:
                return math.atan2(tz, tx)
        return self.prop_angle

    @property
    def pot_circ_r(self) -> float:
        return self.pot_across * math.sqrt(2.0) / 2

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the physical claims the task rests on).
        th = math.degrees(self.prop_angle)
        assert self.open_min_deg + 8.0 < th < self.lid_limit_deg - 10.0, \
            f"the braced angle ({th:.1f} deg) must sit inside the OPEN band with margin"
        assert self.lid_limit_deg <= 85.0, \
            "no over-center: gravity must close the lid from any legal angle"
        # a pot standing ON THE BURNER cannot hold the lid anywhere near open
        pot_top = self.floor_t + self.pot_h
        x_edge = (self.pad_pos[0] - self.hinge_x) - self.pot_circ_r
        rest = math.degrees(math.atan2(pot_top - self.hinge_z, x_edge))
        assert rest < self.open_min_deg - 25.0, \
            f"lid resting on the burner pot ({rest:.1f} deg) must read CLOSED"
        # a pot shoved to the rear wall CAN prop the lid steeply — but then the
        # pocket is beyond the plumb rod's reach (rod clauses stay honest)
        for extra in (8.0, 12.0, 16.0):
            a = self.prop_angle + math.radians(extra)
            if a >= math.radians(self.lid_limit_deg):
                continue
            px = self.hinge_x + self.pocket_r * math.cos(a)
            pz = self.hinge_z + self.pocket_r * math.sin(a)
            d = math.hypot(px - self.well_x, pz - self.floor_t)
            assert d > self.rod_len + 0.004, \
                f"rod must NOT reach the pocket at {math.degrees(a):.0f} deg (d={d:.3f})"
        # the pot drops onto the pad through open sky: at the CATCH angle (the
        # brace's true rest) the lid plane, at pot-top height, stays well
        # behind the pot's near edge
        x_lid = self.hinge_x + (pot_top + 0.015 - self.hinge_z) / math.tan(self.catch_angle)
        assert x_lid < self.pad_pos[0] - self.pot_circ_r - 0.030, \
            "braced lid must leave a vertical drop corridor over the burner pad"
        # closed lid seals the tub (pot cannot enter before the lid is opened)
        assert self.lid_len > self.tub_inner + self.wall_t, "lid must cover the front wall"
        assert self.lid_w / 2 > self.tub_inner / 2 + self.wall_t / 2, \
            "lid must rest on the side wall tops"
        assert self.wall_top - self.floor_t < self.pot_h, \
            "the cavity must be shallower than the pot (no upright pot under a closed lid)"
        # rod/pocket/well fits
        assert self.well_inner > self.rod_s + 0.004, "well must admit the rod base"
        assert math.degrees(math.atan2((self.well_inner - self.rod_s) / 2,
                                       self.well_wall_h)) < 10.0, \
            "well walls must cap the braced rod's base tilt below ~10 deg"
        assert self.pocket_inner_y > self.rod_s + 0.006, "pocket y-aperture must admit the tip"
        assert abs(self.pocket_y - self.well_y) < 1e-9, "pocket and well share a y-plane"
        # engagement clearance: while the tip is inside the cavity the rod
        # SHAFT slants AFT in the lid frame; the aft wall must be shallow
        # enough that the tip enters BELOW its rim at the hold angle, and far
        # enough back that the slanted shaft clears it once the tip is inside
        assert self.pocket_aft_depth < self.pocket_r * math.sin(math.radians(4.5)), \
            "aft wall must be shallow: tip enters below its rim at hold angles"
        th_e = self.prop_angle + math.asin(self.pocket_aft_depth / self.pocket_r)
        assert self.pocket_aft > self.pocket_aft_depth * math.tan(th_e) \
            + self.rod_s / 2 + 0.002, \
            "aft pocket wall must clear the slanted rod shaft during engagement"
        # the deep fore wall arrests the loaded fore-slide at a still-open angle
        th_c = math.degrees(self.catch_angle)
        assert self.open_min_deg + 6.0 < th_c < math.degrees(self.prop_angle), \
            f"fore-wall catch angle ({th_c:.1f} deg) must stay well inside the OPEN band"
        assert self.pocket_depth >= 0.024, \
            "fore/side fence must be deep enough to keep the caught tip enclosed"
        # closed lid: the hanging fence must strike neither the well nor the pad
        fence_x_lo = self.hinge_x + self.pocket_r - self.pocket_aft - self.pocket_wall_t
        well_x_hi = self.well_x + self.well_inner / 2 + self.well_wall_t
        assert fence_x_lo > well_x_hi + 0.015, "closed lid's fence must clear the well in x"
        assert self.hinge_z - self.pocket_depth > self.floor_t + self.pad_h + 0.010, \
            "closed lid's fence must hang clear of the tub floor and pad tops"
        # layout separations (rod lane clear of the tub; pots clear of the rod lane)
        tub_max_x = self.tub_pos[0] + self.tub_inner / 2 + self.wall_t
        rod_half = self.rod_len / 2
        rod_min_x = self.rod_pos[0] - self.rod_jitter \
            - rod_half * math.sin(math.radians(self.rod_yaw_deg)) - self.rod_s / 2 - 0.002
        assert tub_max_x < rod_min_x, "rod lane must stay clear of the tub"
        rod_max_x = self.rod_pos[0] + self.rod_jitter \
            + rod_half * math.sin(math.radians(self.rod_yaw_deg)) + self.rod_s / 2 + 0.002
        assert rod_max_x < self.pot_x - self.pot_jitter - self.pot_circ_r, \
            "pot zones must stay clear of the rod lane"
        # pot zones on the deck and clear of the tub in x
        assert self.pot_x + self.pot_jitter + self.pot_circ_r \
            < self.deck_pos[0] + self.deck_size[0] / 2 - 0.010, "pots must stay on the deck"
        assert self.pot_x - self.pot_jitter - self.pot_circ_r > tub_max_x, \
            "pot zones must stay clear of the tub"
        rod_end_y = rod_half + self.rod_jitter + 0.004
        assert rod_end_y < self.deck_size[1] / 2 - 0.010, "rod must stay on the deck"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
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
@SCENES.register("stove_prop")
class StovePropScene(BaseScene):
    cfg: StovePropSceneCfg

    def __init__(self, cfg: StovePropSceneCfg | None = None) -> None:
        super().__init__(cfg or StovePropSceneCfg())

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
            "tub": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tub",
                spawn=_compound_spawner_cfg("tub", _spawn_tub, c, 12.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tub_pos[0], c.tub_pos[1], c.deck_top)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=_compound_spawner_cfg("lid", _spawn_lid, c, c.lid_mass,
                                            kinematic=False, lin_damp=0.0,
                                            ang_damp=c.lid_ang_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tub_pos[0] + c.hinge_x, c.tub_pos[1],
                         c.deck_top + c.hinge_z)),
            ),
            "rod": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rod",
                spawn=_compound_spawner_cfg("rod", _spawn_rod, c, c.rod_mass,
                                            kinematic=False, lin_damp=0.2, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rod_pos[0] - c.rod_len / 2, c.rod_pos[1],
                         c.deck_top + c.rod_s / 2),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)),
            ),
            "pot_t": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PotT",
                spawn=_compound_spawner_cfg("pot_t", _spawn_pot_t, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_x, -c.pot_y, c.deck_top + 0.001)),
            ),
            "pot_w": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PotW",
                spawn=_compound_spawner_cfg("pot_w", _spawn_pot_w, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_x, c.pot_y, c.deck_top + 0.001)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external wrenches (the solve's lid servo, the smoke pinning
                # probe) act cleanly only with per-iteration application
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
        self.tub: RigidObject = env.iscene["tub"]
        self.lid: RigidObject = env.iscene["lid"]
        self.rod: RigidObject = env.iscene["rod"]
        self.pot_t: RigidObject = env.iscene["pot_t"]
        self.pot_w: RigidObject = env.iscene["pot_w"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # latches (partial credit survives transients; success is judged live)
        self._l_open = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_prop = torch.zeros(n, dtype=torch.bool, device=dev)
        self._l_pot = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: ONE revolute Y joint tub->lid at the rear top edge. Opening
        rotates the panel up (negative joint angle); limits [-80, +0.5] deg keep
        it short of over-center so gravity always closes an unbraced lid."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/lid_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Tub"])
            j.CreateBody1Rel().SetTargets([f"{base}/Lid"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.hinge_x, 0.0, c.hinge_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-c.lid_limit_deg)
            j.CreateUpperLimitAttr(c.lid_slack_deg)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: tub + lid written closed at their fixed anchor (the
        hinge anchors to a kinematic body — it must not move), rod lying on the
        deck (jitter + yaw + 50% end flip), pots jittered with free yaw and the
        COPPER pot's SIDE flipped per episode; latches cleared."""
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

        # --- tub + lid: fixed pose, lid CLOSED on the wall tops ---
        tpos = torch.tensor([c.tub_pos[0], c.tub_pos[1], c.deck_top],
                            device=dev).expand(m, 3)
        write(self.tub, tpos)
        lpos = torch.tensor([c.tub_pos[0] + c.hinge_x, c.tub_pos[1],
                             c.deck_top + c.hinge_z], device=dev).expand(m, 3)
        write(self.lid, lpos)

        # --- rod: lying on the deck along ~world y, centre jittered, 50% flip ---
        ryaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rod_yaw_deg)
        flip = (torch.rand(m, device=dev) < 0.5).float() * math.pi  # which end points +y
        yaw = ryaw + flip + math.pi / 2  # bar axis ~ world y
        cen = torch.zeros(m, 3, device=dev)
        cen[:, 0] = c.rod_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rod_jitter
        cen[:, 1] = c.rod_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rod_jitter
        cen[:, 2] = c.deck_top + c.rod_s / 2
        q_rod = _qmul(_qz(yaw), _qy(torch.full((m,), math.pi / 2, device=dev)))
        u = torch.stack([torch.cos(yaw), torch.sin(yaw),
                         torch.zeros_like(yaw)], dim=-1)  # world dir of rod local +z
        write(self.rod, cen - u * (c.rod_len / 2), q_rod)

        # --- pots: COPPER side is a coin flip; each jittered with free yaw ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        for body, s in ((self.pot_t, side), (self.pot_w, -side)):
            ppos = torch.zeros(m, 3, device=dev)
            ppos[:, 0] = c.pot_x + (torch.rand(m, device=dev) * 2 - 1) * c.pot_jitter
            ppos[:, 1] = s * c.pot_y + (torch.rand(m, device=dev) * 2 - 1) * c.pot_jitter
            ppos[:, 2] = c.deck_top + 0.001
            pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pot_yaw_deg)
            write(body, ppos, _qz(pyaw))

        for latch in (self._l_open, self._l_prop, self._l_pot):
            latch[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "tub": self.tub.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "rod": self.rod.data.root_state_w[env_ids].clone(),
            "pot_t": self.pot_t.data.root_state_w[env_ids].clone(),
            "pot_w": self.pot_w.data.root_state_w[env_ids].clone(),
            "l_open": self._l_open[env_ids].clone(),
            "l_prop": self._l_prop[env_ids].clone(),
            "l_pot": self._l_pot[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("tub", "lid", "rod", "pot_t", "pot_w"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        self._l_open[env_ids] = state["l_open"]
        self._l_prop[env_ids] = state["l_prop"]
        self._l_pot[env_ids] = state["l_pot"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        th = math.degrees(c.catch_angle)
        return (
            f"A wooden COUNTER ({c.deck_size[0] * 100:.0f} x {c.deck_size[1] * 100:.0f} cm "
            f"slab) lies on the ground. On it sits a deep STOVE TUB (inner "
            f"{c.tub_inner * 100:.0f} x {c.tub_inner * 100:.0f} cm, walls "
            f"{(c.wall_top - c.floor_t) * 1000:.0f} mm high) whose floor carries a raised "
            f"octagonal BURNER PAD and, nearer the front-left, a small square SOCKET "
            f"WELL with yellow walls. A heavy steel LID covers the tub completely, "
            f"hinged along the rear edge and limited to {c.lid_limit_deg:.0f} degrees — "
            f"always short of vertical, so GRAVITY SLAMS IT SHUT the moment nothing "
            f"holds it. On the lid's underside a yellow-fenced CAPTURE POCKET sits "
            f"{c.pocket_r * 100:.0f} cm from the hinge. A loose square-section PROP ROD "
            f"({c.rod_len * 100:.0f} cm, red-tipped) lies on the counter. Two moka pots of "
            f"identical shape stand near the counter's front edge, one on each side: "
            f"one COPPER, one STEEL; which side the copper one starts on varies per "
            f"episode.\n"
            f"Goal: put the COPPER moka pot on the burner — which first requires "
            f"propping the stove open. Lift the lid, stand the rod's base in the "
            f"socket well and seat its red tip in the lid's capture pocket (the rod "
            f"braces the lid at ~{th:.0f} degrees), then, hands off the lid, lower the "
            f"copper pot upright onto the burner pad through the propped-open mouth "
            f"and let everything come to rest. Success: the lid held open (>= "
            f"{c.open_min_deg:.0f} degrees) by the braced rod alone, the copper pot "
            f"upright on the burner pad, the steel pot NOT inside the tub, and every "
            f"object at rest, hands off. A lid held open by hand or wedged on a pot, "
            f"a rod braced outside the well or pocket, the steel pot on the burner, "
            f"or a pot dropped anywhere else does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Prop the stove lid open: stand the rod in the floor well and seat its "
            "red tip in the lid's pocket, then place the copper moka pot upright on "
            "the burner pad through the opening and let go of everything."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def lid_angle(self) -> torch.Tensor:
        """(N,) lid opening angle in RADIANS: elevation of the lid's local +x."""
        from isaaclab.utils.math import quat_apply

        q = self.lid.data.root_quat_w
        ex = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        return torch.asin(quat_apply(q, ex)[:, 2].clamp(-1.0, 1.0))

    def lid_open(self) -> torch.Tensor:
        return self.lid_angle() >= math.radians(self.cfg.open_min_deg)

    def lid_calm(self) -> torch.Tensor:
        return self.lid.data.root_ang_vel_w.norm(dim=-1) <= self.cfg.lid_calm_omega

    def rod_base_w(self) -> torch.Tensor:
        return self.rod.data.root_pos_w

    def rod_tip_w(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        q = self.rod.data.root_quat_w
        off = torch.tensor([0.0, 0.0, c.rod_len], device=q.device).expand(q.shape[0], 3)
        return self.rod.data.root_pos_w + quat_apply(q, off)

    def tip_in_pocket(self) -> torch.Tensor:
        """(N,) bool: rod tip inside the lid-frame pocket zone."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        lp = quat_apply_inverse(self.lid.data.root_quat_w,
                                self.rod_tip_w() - self.lid.data.root_pos_w)
        return ((lp[:, 0] - c.pocket_r).abs() <= c.tip_tol_x) \
            & ((lp[:, 1] - c.pocket_y).abs() <= c.tip_tol_y) \
            & (lp[:, 2] > c.tip_z_band[0]) & (lp[:, 2] < c.tip_z_band[1])

    def base_in_well(self) -> torch.Tensor:
        """(N,) bool: rod base inside the socket well (tub is axis-aligned)."""
        c = self.cfg
        tp = self.tub.data.root_pos_w
        d = self.rod_base_w() - tp
        return ((d[:, 0] - c.well_x).abs() <= c.well_tol) \
            & ((d[:, 1] - c.well_y).abs() <= c.well_tol) \
            & (d[:, 2] - c.floor_t > c.well_dz_band[0]) \
            & (d[:, 2] - c.floor_t < c.well_dz_band[1])

    def rod_calm(self) -> torch.Tensor:
        c = self.cfg
        return (self.rod.data.root_lin_vel_w.norm(dim=-1) <= c.settle_speed) \
            & (self.rod.data.root_ang_vel_w.norm(dim=-1) <= 0.5)

    def propped(self) -> torch.Tensor:
        """(N,) bool, live: the lid stands open and calm on the braced rod."""
        return self.lid_open() & self.lid_calm() & self.tip_in_pocket() \
            & self.base_in_well() & self.rod_calm()

    def pot_up(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)[:, 2] >= math.cos(math.radians(self.cfg.upright_max_deg))

    def pot_on_burner(self) -> torch.Tensor:
        """(N,) bool: the COPPER pot upright on the burner pad."""
        c = self.cfg
        tp = self.tub.data.root_pos_w
        pad = tp[:, :2] + torch.tensor(c.pad_pos, device=tp.device)
        d = (self.pot_t.data.root_pos_w[:, :2] - pad).norm(dim=-1)
        dz = self.pot_t.data.root_pos_w[:, 2] - (tp[:, 2] + c.floor_t + c.pad_h)
        return (d <= c.pot_xy_tol) & (dz > -0.006) & (dz < 0.014) & self.pot_up(self.pot_t)

    def wrong_in_tub(self) -> torch.Tensor:
        """(N,) bool: the STEEL pot anywhere inside the tub's airspace."""
        c = self.cfg
        d = self.pot_w.data.root_pos_w - self.tub.data.root_pos_w
        half = c.tub_inner / 2 + c.wall_t
        return (d[:, 0].abs() < half) & (d[:, 1].abs() < half) & (d[:, 2] < 0.30)

    def settled(self) -> torch.Tensor:
        """(N,) bool: rod and both pots at rest, lid quasi-static."""
        c = self.cfg
        ok = self.lid_calm()
        for b in (self.rod, self.pot_t, self.pot_w):
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return ok

    def _update_latches(self) -> None:
        c = self.cfg
        self._l_open |= self.lid_open() & self.lid_calm()
        self._l_prop |= self.propped()
        calm = self.pot_t.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._l_pot |= self.pot_on_burner() & calm

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, all live: lid open + calm on the braced rod (base in well,
        tip in pocket, rod at rest), copper pot upright on the burner pad, steel
        pot NOT inside the tub, everything at rest, finite. The rod clauses are
        the anti-pinning backstop: a hand (or wrench) holding the lid open, or a
        pot wedged under it, passes the angle but not the brace."""
        self._update_latches()
        finite = torch.isfinite(self.lid.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.rod.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.pot_t.data.root_pos_w).all(dim=-1)
        return self.propped() & self.pot_on_burner() & ~self.wrong_in_tub() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*lid_open + 0.30*propped + 0.25*pot_set (all
        latched; ~0 for doing nothing), capped at 0.75 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._l_open.float() + c.w_prop * self._l_prop.float()
                + c.w_pot * self._l_pot.float()).clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="stove_prop", robot="null"))
