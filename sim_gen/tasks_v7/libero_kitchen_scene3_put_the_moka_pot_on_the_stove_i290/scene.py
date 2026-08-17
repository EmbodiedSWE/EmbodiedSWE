"""MokaCarouselScene — deliver the moka pot to the hooded warming station by riding
the serving carousel (sim_gen task `libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i290`).

Derived from libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove, but
STRATEGICALLY different. The seed is a direct support-surface pick-and-place:
grasp the moka pot, carry it over the open stove, set it down — success is a pose
predicate satisfied by one grasp-carry-release motion. Here the target surface is
GEOMETRICALLY UNREACHABLE by that plan:

  - the warming station is one stop of a ROTATING CAROUSEL (a turntable with three
    walled bays), and the station sits under a LOW CANOPY: the canopy underside is
    100 mm above the turntable while fence(35) + pot(76) = 111 mm, so a pot can
    neither be lowered into the station bay from above nor lifted over a bay fence
    anywhere under the canopy;
  - the only way a pot gets to the station is INDIRECT: put it into an OPEN bay
    while that bay is out in front, then ROTATE the carousel (push its brass
    posts) so the bay carries the pot under the canopy, and STOP with the bay
    centered on the station;
  - one bay is already occupied by a frying pan (the seed's distractor) and the
    turntable starts at a random angle, so the solver must pick an empty bay —
    and, on many episodes, first rotate the carousel to bring an empty bay out to
    the open front before it can load the pot at all;
  - success is judged on the SETTLED outcome: pot upright in a bay, that bay's
    azimuth within the station window, turntable stopped, everything at rest.

A solver therefore needs a different PLAN from the seed (stage an empty bay ->
load -> closed-loop rotate to an azimuth window -> controlled stop; transport is
mediated by a mechanism, and the terminal predicate lives on the mechanism's
angle) and a different code structure (nothing is "move pot to pose X"; the pot's
final pose is produced by the carousel, not by the hand).

Assets are fully procedural (native PhysX box colliders):
  - deck (KINEMATIC): counter slab 1.00 x 0.90 x 0.024 m on the ground.
  - hub (KINEMATIC): pedestal column at the carousel centre.
  - disc (DYNAMIC, revolute-Z joint to the hub, no limits): a 16 mm turntable
    slab (two crossed squares) carrying three walled square BAYS (inner 120 mm,
    fences 35 mm tall) at radius 130 mm / 120 deg apart, and three brass PUSH
    POSTS at radius 190 mm between the bays. Angular damping stands in for
    bearing friction; rotation about z has no gravity load.
  - hood (KINEMATIC): the warming station canopy over the back (+x) sector —
    roof slab on three pillars, underside 100 mm above the turntable, with an
    orange lamp on top marking the station. The station azimuth is 0 (+x from
    the carousel centre), fixed in the world.
  - pot (DYNAMIC): octagonal moka pot, 54 mm across flats, 76 mm tall, side
    handle + lid knob; starts on the counter in front of the carousel.
  - frypan (DYNAMIC): dark octagonal frying pan occupying one random bay.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.25 * seated     — pot ever at rest, upright, inside any bay (latched)
  0.35 * progress   — latched running-MIN of the pot bay's azimuth error to the
                      station, scaled (1 - err/pi); only counts once seated
  1.0 iff success() — live: pot upright in a bay, bay azimuth within `az_tol_deg`
                      of the station, pot at rest, turntable stopped, finite.
  Non-success capped at 0.60.

Honesty by construction (asserted in __post_init__): the canopy is too low to
lift a pot over a bay fence anywhere under it, the pot rides through with 24 mm
of headroom, the bay geometry stays on turntable material through the crossed-
square star profile, and the whole success window lies under the canopy.

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


def _qz_tuple(ang: float):
    return (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))


def _oct45():
    """wxyz quat: 45 degrees about z (the octagon's second square)."""
    return _qz_tuple(math.pi / 4)


def _spawn_hub(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC pedestal column. Origin = its BOTTOM centre (on the deck)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 6.0, kinematic=True)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/column", (cfg.hub_s, cfg.hub_s, cfg.hub_h),
         (0.0, 0.0, cfg.hub_h / 2), cfg.hub_color, co)
    return root


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC carousel turntable. Origin = slab centre (the joint anchor).
    Slab = two crossed squares (octagram profile); three walled bays at
    `r_bay` / 120 deg apart; three brass push posts at `r_post` between them."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.disc_mass, kinematic=False, lin_damp=0.0,
                ang_damp=cfg.disc_ang_damp, com=(0.0, 0.0, 0.0),
                inertia=cfg.disc_inertia)

    co = cfg.contact_offset
    sh = cfg.slab_half
    st = cfg.slab_t
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/slab_{tag}", (2 * sh, 2 * sh, st), (0.0, 0.0, 0.0),
             cfg.disc_color, co, orient=orient)
    hi = cfg.bay_in / 2
    ft, fh = cfg.fence_t, cfg.fence_h
    fz = st / 2 + fh / 2
    for k in range(3):
        phi = k * 2 * math.pi / 3
        cx, cy = cfg.r_bay * math.cos(phi), cfg.r_bay * math.sin(phi)
        q = _qz_tuple(phi)
        cp, sp = math.cos(phi), math.sin(phi)
        # radial walls (outer/inner) then tangential walls, all in the bay frame
        for sx in (1.0, -1.0):
            off = (sx * (hi + ft / 2), 0.0)
            _box(stage, f"{prim_path}/bay{k}_r{'p' if sx > 0 else 'n'}",
                 (ft, cfg.bay_in + 2 * ft, fh),
                 (cx + cp * off[0] - sp * off[1], cy + sp * off[0] + cp * off[1], fz),
                 cfg.fence_color, co, orient=q)
        for sy in (1.0, -1.0):
            off = (0.0, sy * (hi + ft / 2))
            _box(stage, f"{prim_path}/bay{k}_t{'p' if sy > 0 else 'n'}",
                 (cfg.bay_in, ft, fh),
                 (cx + cp * off[0] - sp * off[1], cy + sp * off[0] + cp * off[1], fz),
                 cfg.fence_color, co, orient=q)
        # brass push post between this bay and the next
        psi = phi + math.pi / 3
        _box(stage, f"{prim_path}/post{k}",
             (cfg.post_s, cfg.post_s, cfg.post_h),
             (cfg.r_post * math.cos(psi), cfg.r_post * math.sin(psi),
              st / 2 + cfg.post_h / 2),
             cfg.post_color, co, orient=_qz_tuple(psi))
    return root


def _spawn_hood(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC warming-station canopy. Origin = deck top at the CAROUSEL
    CENTRE; the canopy covers the +x (station) sector: roof slab on three
    pillars outside the turntable's swept star profile, orange lamp on top."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 10.0, kinematic=True)
    co = cfg.contact_offset
    roof_und = cfg.roof_under_local  # roof underside above THIS origin (deck top)
    pil_h = roof_und
    for tag, (px, py) in (("l", (cfg.roof_x0 + cfg.roof_lx / 2, cfg.pillar_y)),
                          ("r", (cfg.roof_x0 + cfg.roof_lx / 2, -cfg.pillar_y)),
                          ("b", (cfg.pillar_back_x, 0.0))):
        _box(stage, f"{prim_path}/pillar_{tag}", (cfg.pillar_s, cfg.pillar_s, pil_h),
             (px, py, pil_h / 2), cfg.hood_color, co)
    _box(stage, f"{prim_path}/roof",
         (cfg.roof_lx, cfg.roof_ly, cfg.roof_t),
         (cfg.roof_x0 + cfg.roof_lx / 2, 0.0, roof_und + cfg.roof_t / 2),
         cfg.hood_color, co)
    # orange station lamp: VISUAL ONLY marker on the roof
    _box(stage, f"{prim_path}/lamp", (0.05, 0.05, 0.02),
         (cfg.roof_x0 + cfg.roof_lx / 2, 0.0, roof_und + cfg.roof_t + 0.01),
         cfg.lamp_color, None)
    return root


def _spawn_pot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC moka pot. Origin = base BOTTOM centre: octagonal base + waisted
    octagonal top + lid knob + short side handle along local +x."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pot_mass, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.034), inertia=cfg.pot_inertia)
    co = cfg.contact_offset
    body, dark = cfg.pot_color, cfg.pot_dark_color
    b, bh = cfg.pot_flats, 0.040
    t, th = 0.044, 0.028
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/base_{tag}", (b, b, bh), (0.0, 0.0, bh / 2), body, co,
             orient=orient)
        _box(stage, f"{prim_path}/top_{tag}", (t, t, th), (0.0, 0.0, bh + th / 2), body, co,
             orient=orient)
    _box(stage, f"{prim_path}/knob", (0.012, 0.012, 0.008),
         (0.0, 0.0, bh + th + 0.004), dark, co)
    _box(stage, f"{prim_path}/handle", (0.022, 0.011, 0.018),
         (0.036, 0.0, 0.050), dark, co)
    return root


def _spawn_frypan(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC frying pan (the seed's distractor): dark octagonal body + stub
    handle. Origin = its BOTTOM centre. Occupies one carousel bay."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pan_mass, kinematic=False, lin_damp=0.3, ang_damp=0.5,
                com=(0.0, 0.0, 0.012), inertia=cfg.pan_inertia)
    co = cfg.contact_offset
    b, bh = cfg.pan_flats, 0.028
    for tag, orient in (("a", None), ("b", _oct45())):
        _box(stage, f"{prim_path}/body_{tag}", (b, b, bh), (0.0, 0.0, bh / 2),
             cfg.pan_color, co, orient=orient)
    _box(stage, f"{prim_path}/handle", (0.016, 0.014, 0.010),
         (0.043, 0.0, 0.021), (0.05, 0.05, 0.06), co)
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
class MokaCarouselSceneCfg(BaseCfg):
    """Config for `MokaCarouselScene`. Honesty geometry asserted in __post_init__."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    az_tol_deg: float = tunable(15.0)   # bay azimuth window around the station (deg)
    settle_speed: float = tunable(0.10)  # pot/turntable lin speed at rest (m/s)
    disc_calm_omega: float = tunable(0.10)  # turntable |wz| when judging (rad/s)
    upright_cos: float = tunable(0.90)  # pot local +z . world up (~25 deg cone)
    latch_speed: float = tunable(0.10)  # calm gate for the seated latch (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pot_zone_x: tuple = tunable((0.02, 0.035))   # pot start x: nominal +/- jitter
    pot_zone_y: float = tunable(0.20)            # pot start |y| bound (uniform)
    pot_yaw_deg: float = tunable(180.0)          # pot free yaw (+/- deg)
    pan_yaw_deg: float = tunable(180.0)          # frypan free yaw in its bay

    # --- info: layout (world nominal; deck top = z 0.024) ----------------------------------------
    deck_size: tuple = info((1.00, 0.90, 0.024))
    deck_pos: tuple = info((0.42, 0.0))
    center: tuple = info((0.42, 0.0))    # carousel axis (fixed: the joint anchor)
    # --- info: hub / disc ------------------------------------------------------------------------
    hub_s: float = info(0.09)
    hub_h: float = info(0.046)
    disc_lift: float = info(0.058)       # slab CENTRE above the deck top
    slab_half: float = info(0.20)        # crossed-square half extent (star reach *sqrt2)
    slab_t: float = info(0.016)
    disc_mass: float = info(1.2)
    disc_inertia: tuple = info((0.020, 0.020, 0.035))
    disc_ang_damp: float = info(0.5)     # bearing friction stand-in
    r_bay: float = info(0.13)
    bay_in: float = info(0.12)           # bay inner side
    fence_t: float = info(0.008)
    fence_h: float = info(0.035)
    r_post: float = info(0.19)
    post_s: float = info(0.016)
    post_h: float = info(0.080)
    # --- info: hood (local to deck top at the carousel centre) -----------------------------------
    roof_dz: float = info(0.100)         # roof underside above the TURNTABLE TOP
    roof_x0: float = info(0.03)          # roof inner edge (x from the axis)
    roof_lx: float = info(0.28)
    roof_ly: float = info(0.64)
    roof_t: float = info(0.014)
    pillar_s: float = info(0.03)
    pillar_y: float = info(0.305)
    pillar_back_x: float = info(0.32)
    # --- info: pot / frypan ----------------------------------------------------------------------
    pot_flats: float = info(0.054)       # octagon across-flats (base)
    pot_h: float = info(0.076)
    pot_handle_reach: float = info(0.047)
    pot_mass: float = info(0.24)
    pot_inertia: tuple = info((0.00016, 0.00016, 0.00009))
    pan_flats: float = info(0.072)
    pan_reach: float = info(0.051)
    pan_mass: float = info(0.30)
    pan_inertia: tuple = info((0.00012, 0.00012, 0.0002))
    # --- info: judge volumes ---------------------------------------------------------------------
    bay_xy_bound: float = info(0.055)    # bay-frame |x|,|y| bound for "in this bay"
    bay_z_band: tuple = info((-0.006, 0.035))  # pot origin above the turntable top
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.55, 0.44, 0.30))
    hub_color: tuple = info((0.30, 0.31, 0.34))
    disc_color: tuple = info((0.62, 0.50, 0.32))
    fence_color: tuple = info((0.42, 0.33, 0.20))
    post_color: tuple = info((0.72, 0.58, 0.20))
    hood_color: tuple = info((0.16, 0.16, 0.18))
    lamp_color: tuple = info((0.95, 0.45, 0.10))
    pot_color: tuple = info((0.60, 0.61, 0.65))
    pot_dark_color: tuple = info((0.05, 0.05, 0.06))
    pan_color: tuple = info((0.13, 0.13, 0.15))
    # rubric weights (0.25 + 0.35 = 0.60 = the non-success cap)
    w_seat: float = info(0.25)
    w_prog: float = info(0.35)

    # ----- derived geometry ----------------------------------------------------------------------
    @property
    def deck_top(self) -> float:
        return self.deck_size[2]

    @property
    def disc_z(self) -> float:
        """World z of the slab centre."""
        return self.deck_top + self.disc_lift

    @property
    def disc_top(self) -> float:
        """World z of the turntable top surface."""
        return self.disc_z + self.slab_t / 2

    @property
    def roof_under_local(self) -> float:
        """Roof underside above the HOOD origin (the deck top)."""
        return self.disc_top - self.deck_top + self.roof_dz

    @property
    def star_reach(self) -> float:
        """Max radius swept by the crossed-square slab (the corner points)."""
        return self.slab_half * math.sqrt(2.0)

    @property
    def star_inner(self) -> float:
        """Min boundary radius of the crossed-square slab profile."""
        return self.slab_half / math.cos(math.pi / 8)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the geometric claims the task rests on).
        # 1. Under the canopy a pot can NOT be lifted over a bay fence...
        assert self.fence_h + self.pot_h > self.roof_dz + 0.008, \
            "canopy must block lifting the pot over a bay fence beneath it"
        # ...yet a pot riding IN a bay passes under with real headroom.
        assert self.pot_h + 0.020 <= self.roof_dz, \
            "a bay-riding pot must clear the canopy underside by >= 20 mm"
        assert self.post_h + 0.015 <= self.roof_dz, "posts must ride under the canopy"
        # 2. Bays and posts stay on turntable material at every azimuth.
        out_face = self.r_bay + self.bay_in / 2 + self.fence_t
        out_corner = math.hypot(out_face, self.bay_in / 2 + self.fence_t)
        assert out_corner < self.star_inner - 0.004, "bay fences must stay on the slab"
        assert self.r_post + self.post_s < self.star_inner, "posts must stay on the slab"
        # 3. The swept star clears the canopy pillars.
        assert self.pillar_y - self.pillar_s / 2 > self.star_reach + 0.005, \
            "side pillars must clear the swept turntable"
        assert self.pillar_back_x - self.pillar_s / 2 > self.star_reach + 0.005, \
            "back pillar must clear the swept turntable"
        # 4. Pot (with handle) and frypan fit a bay with margin at any yaw.
        assert self.pot_handle_reach + 0.008 <= self.bay_in / 2, "pot must fit a bay"
        assert self.pot_flats * math.sqrt(2.0) / 2 + 0.008 <= self.bay_in / 2, \
            "pot body corners must fit a bay"
        assert self.pan_reach + 0.006 <= self.bay_in / 2, "frypan must fit a bay"
        # 5. The whole success window lies under the canopy (roof covers the bay
        # centre out to +/- az_tol and far beyond).
        cover = math.degrees(math.acos(self.roof_x0 / self.r_bay))
        assert cover > self.az_tol_deg + 30.0, \
            "success window must sit deep under the canopy"
        # 6. The pot spawn zone stays clear of the swept star (5+ mm rule).
        reach = self.pot_zone_x[0] + self.pot_zone_x[1] + self.pot_handle_reach
        assert reach < self.center[0] - self.star_reach - 0.02, \
            "pot spawn zone must clear the swept turntable"
        # 7. Loading drop stays canopy-free: at the load azimuth (pi) the roof is
        # on the other side, and the hover (pot top at drop) is below nothing.
        assert self.w_seat + self.w_prog <= 0.601, "non-success cap must stay at 0.60"


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("moka_carousel")
class MokaCarouselScene(BaseScene):
    cfg: MokaCarouselSceneCfg

    def __init__(self, cfg: MokaCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or MokaCarouselSceneCfg())

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
                spawn=kin_slab(c.deck_size, c.deck_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.deck_pos[0], c.deck_pos[1], c.deck_size[2] / 2)),
            ),
            "hub": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hub",
                spawn=_compound_spawner_cfg("hub", _spawn_hub, c, 6.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.center[0], c.center[1], c.deck_top)),
            ),
            "hood": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hood",
                spawn=_compound_spawner_cfg("hood", _spawn_hood, c, 10.0, kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.center[0], c.center[1], c.deck_top)),
            ),
            "disc": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc",
                spawn=_compound_spawner_cfg("disc", _spawn_disc, c, c.disc_mass,
                                            kinematic=False, lin_damp=0.0,
                                            ang_damp=c.disc_ang_damp),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.center[0], c.center[1], c.disc_z)),
            ),
            "pot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot",
                spawn=_compound_spawner_cfg("pot", _spawn_pot, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pot_zone_x[0], 0.0, c.deck_top + 0.002)),
            ),
            "frypan": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Frypan",
                spawn=_compound_spawner_cfg("frypan", _spawn_frypan, c, c.pan_mass,
                                            kinematic=False, lin_damp=0.3, ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.center[0] + c.r_bay, c.center[1], c.disc_top + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # external wrenches (the solve's turntable servo) act cleanly only
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
        self.hub: RigidObject = env.iscene["hub"]
        self.hood: RigidObject = env.iscene["hood"]
        self.disc: RigidObject = env.iscene["disc"]
        self.pot: RigidObject = env.iscene["pot"]
        self.frypan: RigidObject = env.iscene["frypan"]
        self.env_origins = env.iscene.env_origins
        self._phi = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], device=dev)
        self._author_joint()
        # frypan's bay index per env (sampled at reset)
        self.pan_bay = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._l_seat = torch.zeros(n, dtype=torch.bool, device=dev)
        self._best_err = torch.full((n,), math.pi, device=dev)

    def _author_joint(self) -> None:
        """Per env: one revolute Z joint hub->disc (free, NO limits — the
        turntable spins continuously; joint pairs never collide)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/carousel_axle")
            j.CreateBody0Rel().SetTargets([f"{base}/Hub"])
            j.CreateBody1Rel().SetTargets([f"{base}/Disc"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.disc_lift))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: random turntable angle, frypan in a random bay
        (written together with the disc — one consistent linkage write), pot on
        the counter in front with jitter + free yaw; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        torch.rand(3, device=dev)  # burn draws: first post-seed draw degeneracy

        def write(body, pos, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            body.write_root_state_to_sim(st, env_ids)

        # --- turntable: random angle about the fixed axis ---
        theta0 = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        dpos = torch.zeros(m, 3, device=dev)
        dpos[:, 0], dpos[:, 1], dpos[:, 2] = c.center[0], c.center[1], c.disc_z
        write(self.disc, dpos, _qz(theta0))

        # --- frypan: random bay, pose computed FROM theta0 (consistent linkage) ---
        bay = (torch.rand(m, device=dev) * 3).floor().long().clamp(max=2)
        self.pan_bay[env_ids] = bay
        az = theta0 + self._phi[bay]
        fpos = torch.zeros(m, 3, device=dev)
        fpos[:, 0] = c.center[0] + c.r_bay * torch.cos(az)
        fpos[:, 1] = c.center[1] + c.r_bay * torch.sin(az)
        fpos[:, 2] = c.disc_top + 0.002
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pan_yaw_deg)
        write(self.frypan, fpos, _qz(fyaw))

        # --- pot: on the counter in front, jitter + free yaw ---
        ppos = torch.zeros(m, 3, device=dev)
        ppos[:, 0] = c.pot_zone_x[0] + (torch.rand(m, device=dev) * 2 - 1) * c.pot_zone_x[1]
        ppos[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.pot_zone_y
        ppos[:, 2] = c.deck_top + 0.002
        pyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pot_yaw_deg)
        write(self.pot, ppos, _qz(pyaw))

        # --- clear latches ---
        self._l_seat[env_ids] = False
        self._best_err[env_ids] = math.pi

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "disc": self.disc.data.root_state_w[env_ids].clone(),
            "pot": self.pot.data.root_state_w[env_ids].clone(),
            "frypan": self.frypan.data.root_state_w[env_ids].clone(),
            "pan_bay": self.pan_bay[env_ids].clone(),
            "l_seat": self._l_seat[env_ids].clone(),
            "best_err": self._best_err[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.disc.write_root_state_to_sim(state["disc"], env_ids)
        self.pot.write_root_state_to_sim(state["pot"], env_ids)
        self.frypan.write_root_state_to_sim(state["frypan"], env_ids)
        self.pan_bay[env_ids] = state["pan_bay"]
        self._l_seat[env_ids] = state["l_seat"]
        self._best_err[env_ids] = state["best_err"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden counter carries a serving CAROUSEL: a round wooden turntable "
            f"(~{2 * c.slab_half * 100:.0f} cm across) on a fixed pedestal, free to spin "
            f"about its centre. On the turntable sit THREE square walled BAYS "
            f"({c.bay_in * 1000:.0f} mm inner, {c.fence_h * 1000:.0f} mm fences, 120 degrees "
            f"apart) and three BRASS POSTS between them ({c.post_h * 1000:.0f} mm tall) — "
            f"push a post sideways to rotate the carousel. The turntable starts at a "
            f"random angle, and one bay is already occupied by a dark FRYING PAN.\n"
            f"At the BACK of the carousel (its far side, +x from the axis) a dark CANOPY "
            f"on pillars roofs over the turntable: the WARMING STATION, marked by the "
            f"ORANGE LAMP on the canopy roof. The canopy underside is only "
            f"{c.roof_dz * 1000:.0f} mm above the turntable — a bay-riding object passes "
            f"under it, but there is no room to lower an object into a bay or lift one "
            f"over a bay fence anywhere beneath it.\n"
            f"On the counter in front of the carousel stands a silver octagonal MOKA POT "
            f"({c.pot_flats * 1000:.0f} mm across, {c.pot_h * 1000:.0f} mm tall, black side "
            f"handle and lid knob).\n"
            f"Goal: serve the moka pot to the warming station. Put the pot UPRIGHT into "
            f"an OPEN bay while that bay is out in the open (rotate the carousel first "
            f"if no empty bay is accessible), then rotate the carousel so that bay "
            f"carries the pot under the canopy, and stop with the pot's bay centred on "
            f"the station (within {c.az_tol_deg:.0f} degrees of the lamp's azimuth). "
            f"Success is judged at rest: pot upright in its bay at the station, "
            f"turntable stopped, everything settled, hands off. The frying pan parked "
            f"at the station, a pot left outside a bay, a toppled pot, or a "
            f"still-turning carousel does not count. Either rotation direction is fine; "
            f"there is no required order beyond loading before delivering."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the moka pot upright into an empty bay of the carousel, then rotate "
            "the carousel by its posts until that bay sits centred under the orange-"
            "lamp canopy, and stop it there. The pot must end upright in its bay at "
            "the station with the carousel at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def disc_yaw(self) -> torch.Tensor:
        """(N,) turntable yaw about world z (radians, wrapped)."""
        q = self.disc.data.root_quat_w
        return _wrap(2.0 * torch.atan2(q[:, 3], q[:, 0]))

    def pocket_azimuths(self) -> torch.Tensor:
        """(N,3) world azimuth of each bay centre (station = azimuth 0)."""
        return _wrap(self.disc_yaw().unsqueeze(1) + self._phi[None, :])

    def _local_of(self, body) -> torch.Tensor:
        """(N,3) body origin in the TURNTABLE frame."""
        from isaaclab.utils.math import quat_apply_inverse

        dp = body.data.root_pos_w - self.disc.data.root_pos_w
        return quat_apply_inverse(self.disc.data.root_quat_w, dp)

    def _bay_frame_xy(self, lp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bay-frame coords for each bay k: (N,3) x (radial off-centre), (N,3) y."""
        cp, sp = torch.cos(self._phi), torch.sin(self._phi)
        x = lp[:, 0:1] * cp[None] + lp[:, 1:2] * sp[None] - self.cfg.r_bay
        y = -lp[:, 0:1] * sp[None] + lp[:, 1:2] * cp[None]
        return x, y

    def pot_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.pot.data.root_quat_w, ez)[:, 2] >= self.cfg.upright_cos

    def pot_bay(self) -> torch.Tensor:
        """(N,) long: index of the bay holding an UPRIGHT pot at floor level,
        else -1. Judged in the turntable frame, so it holds while riding."""
        c = self.cfg
        lp = self._local_of(self.pot)
        x, y = self._bay_frame_xy(lp)
        z = lp[:, 2] - c.slab_t / 2  # above the turntable top
        zin = (z > c.bay_z_band[0]) & (z < c.bay_z_band[1])
        inside = (x.abs() <= c.bay_xy_bound) & (y.abs() <= c.bay_xy_bound)
        ok = inside & zin.unsqueeze(1) & self.pot_upright().unsqueeze(1)
        return torch.where(ok.any(dim=1), ok.float().argmax(dim=1),
                           torch.full_like(self.pan_bay, -1))

    def bay_err(self) -> torch.Tensor:
        """(N,) azimuth error of the pot's bay to the station (pi if no bay)."""
        k = self.pot_bay()
        az = self.pocket_azimuths().abs()
        err = az.gather(1, k.clamp(min=0).unsqueeze(1)).squeeze(1)
        return torch.where(k >= 0, err, torch.full_like(err, math.pi))

    def still(self) -> torch.Tensor:
        """(N,) bool: pot at rest AND turntable stopped."""
        c = self.cfg
        return (self.pot.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.disc.data.root_ang_vel_w[:, 2].abs() < c.disc_calm_omega) \
            & (self.disc.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def _update_latches(self) -> None:
        k = self.pot_bay()
        in_bay = k >= 0
        speed = self.pot.data.root_lin_vel_w.norm(dim=-1)
        self._l_seat |= in_bay & (speed < self.cfg.latch_speed)
        err = self.bay_err()
        upd = in_bay & self._l_seat
        self._best_err = torch.where(upd, torch.minimum(self._best_err, err),
                                     self._best_err)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, all live: pot upright in a bay, that bay within the
        station window, pot at rest, turntable stopped, finite. The bay-frame
        clause rejects a pot pinned mid-air or parked on the canopy; the calm
        clauses reject a still-turning fly-through of the window."""
        self._update_latches()
        c = self.cfg
        delivered = (self.pot_bay() >= 0) \
            & (self.bay_err() <= math.radians(c.az_tol_deg))
        finite = torch.isfinite(self.pot.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.disc.data.root_pos_w).all(dim=-1)
        return delivered & self.still() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25*seated (latched) + 0.35*progress (latched
        running-min azimuth error, gated on seated; ~0 for doing nothing),
        capped at 0.60 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        prog = (1.0 - self._best_err / math.pi).clamp(0.0, 1.0)
        base = (c.w_seat * self._l_seat.float()
                + c.w_prog * self._l_seat.float() * prog).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="moka_carousel", robot="null"))
