"""SlabEaselScene — erect the display slabs (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet_i53`).

Derived from libero_90/libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet,
but STRATEGICALLY different: the seed is a single pick-and-place — grasp the frying pan
and set it down inside the cabinet-top bounding box; one grasp-and-set-down transport
ends the task. Here the goal is a POSE CHANGE IN PLACE, not a delivery: three colored
display slabs (RED, GREEN, BLUE) lie FLAT on a low studio deck and must each be
PIVOT-ERECTED — hinged up about their own grounded foot edge, through the 30–70° band,
past the tip-over balance point (~83°) — so each comes to rest LEANING flush against its
color-matched tilted backrest panel, ladder-style. A fourth BLANK slab (identity
control) must stay lying flat. The slabs are deliberately UNGRASPABLE as free bodies
(0.24 x 0.20 m faces dwarf an 80 mm parallel jaw; 30 mm thin, no handle): the only
embodied route is nonprehensile — slide a slab to the foot of its bay, get a fingertip
under the front edge through a deck rail gap, and hinge it up until it tips back onto
the panel. No corpus task judges erecting free bodies past an unstable equilibrium; the
seed's "object centre in region" check becomes a face-normal-alignment + lean-pose
predicate plus a RISING-TILT ACCUMULATOR that only credits tilt gained plausibly, in the
bay, through the band — a slab teleported (or written) upright earns nothing.

Judged in the STUDIO's body frame (kinematic rack, xy + free yaw randomized: bay
directions must be read from the scene). success() iff, for each colored slab: it rests
seated against its own bay's panel (face normal aligned with the panel normal within
`seat_normal_deg`, centre in the bay's seat window) AND its erect accumulator crossed
`erect_req_deg` (the tilt was physically swept through the 30–70° band in that bay);
AND the blank slab still lies flat on the work surface; AND everything is settled (a
consecutive-substep pose-delta stillness counter — velocity readings lie for bodies
resting on edges). score() is latched, non-decreasing: per colored slab 0.06 staged +
0.12 x erect-band progress + 0.10 seated-still, capped at 0.85; exactly 1.0 iff
success(). Doing nothing scores ~0 (spawns are outside every staging window); the
seed's strategy (carry the object to the target and set it down flat) leaves the slab
flat in the bay: at most the 0.06 staging latch, never success.

Assets are fully procedural (no external files):
  - studio rack (KINEMATIC compound): a low deck (1.05 x 1.40 x 0.10 m) whose top
    carries, at the back (local -x), three display BAYS at local y = -0.36 / 0 / +0.36:
    each a color-tinted floor plate at the foot of a backrest PANEL (0.02 x 0.26 x
    0.32 m) tilted 10° back, capped by a bright color TRIM bar (the bay's label). In
    front of the bays, six low RAILS (35 mm wide, 20 mm tall) run across the deck with
    30 mm finger gaps between them; rail tops, bay plates and the back apron are flush
    at one work plane (z = +0.02 above the deck top), so slabs slide anywhere on it but
    a fingertip fits UNDER a slab edge only at a rail gap.
  - slabs (DYNAMIC, 4x): 0.24 x 0.20 x 0.03 m boards, 2.2 kg. RED, GREEN, BLUE slabs
    carry a colored tile on BOTH large faces (either face may end up showing); the
    BLANK slab is plain gray. Identical bodies — only color identity distinguishes them.
Moderate friction everywhere (deck 0.35, slabs 0.45) so leaning is stable (a ladder
lean at 80° needs ~0.09), restitution 0, small contact offsets.

Per-episode randomization (verified by readback in smoke): studio xy jitter + FREE yaw
(the bay directions must be read from the scene), and the four slabs spawn on the open
rail field in a PERMUTED assignment of four y-slots with xy jitter and free yaw
(rejection-resampled to a minimum pairwise separation), so a memorized trajectory
fails. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _collide(prim, contact_offset: float, material=None) -> None:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, orient=None) -> None:
    """One collidable box child prim (translate -> orient -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
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
    _collide(seg.GetPrim(), contact_offset, material)


def _box_visual(stage, path: str, size, center, color, orient=None) -> None:
    """One VISUAL-ONLY box child prim (no collision — used for the face color tiles)."""
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


def _spawn_studio(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC studio rack at `prim_path`. Origin = deck-top centre
    (z = 0 at the deck top); local -x = toward the bays/panels. One flush work plane at
    z = cfg.z_work: back apron + three tinted bay plates + six rails, with 30 mm finger
    gaps between the rails."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(80.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    zw = cfg.z_work
    xbp = cfg.x_bp
    beta = math.radians(cfg.panel_beta_deg)
    qp = (math.cos(beta / 2), 0.0, -math.sin(beta / 2), 0.0)  # lean back about +y

    # --- deck ---
    _box(stage, f"{prim_path}/deck", cfg.deck_size,
         (0.0, 0.0, -cfg.deck_size[2] / 2), cfg.deck_color, co, material=mat)

    # --- back apron (work plane behind the panel foot line) ---
    _box(stage, f"{prim_path}/apron", cfg.apron_size,
         (xbp - cfg.apron_size[0] / 2, 0.0, zw - cfg.apron_size[2] / 2),
         cfg.apron_color, co, material=mat)

    # --- rails (work plane in front of the bays, with finger gaps) ---
    for k in range(cfg.n_rails):
        _box(stage, f"{prim_path}/rail_{k}", cfg.rail_size,
             (cfg.rail_x0 + k * cfg.rail_pitch, 0.0, zw - cfg.rail_size[2] / 2),
             cfg.rail_color, co, material=mat)

    # --- three bays: tinted plate + tilted panel + color trim cap ---
    ph, pt = cfg.panel_size[2], cfg.panel_size[0]
    ux, uz = -math.sin(beta), math.cos(beta)          # up-the-panel-face direction
    nx, nz = math.cos(beta), math.sin(beta)           # panel front (outward) normal
    pcx = xbp + (ph / 2) * ux - (pt / 2) * nx         # panel centre (foot at x_bp, z_work)
    pcz = zw + (ph / 2) * uz - (pt / 2) * nz
    tcx = pcx + (ph / 2 + cfg.trim_size[2] / 2) * ux  # trim cap centred on the panel top
    tcz = pcz + (ph / 2 + cfg.trim_size[2] / 2) * uz
    for i, by in enumerate(cfg.bay_ys):
        _box(stage, f"{prim_path}/bay_plate_{i}", cfg.bay_plate_size,
             (xbp + cfg.bay_plate_size[0] / 2, by, zw - cfg.bay_plate_size[2] / 2),
             cfg.bay_colors[i], co, material=mat)
        _box(stage, f"{prim_path}/panel_{i}", cfg.panel_size,
             (pcx, by, pcz), cfg.panel_color, co, material=mat, orient=qp)
        _box(stage, f"{prim_path}/trim_{i}", cfg.trim_size,
             (tcx, by, tcz), cfg.trim_colors[i], co, material=mat, orient=qp)
    return root


def _spawn_slab(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one DYNAMIC slab at `prim_path`. Origin = geometric centre (MassAPI mass
    leaves the CoM at the body origin — centred by construction). Colored slabs carry a
    visual-only tile on BOTH large faces."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.25)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    mat = _friction_material(stage, f"{prim_path}/phys_mat", cfg.mu_static, cfg.mu_dynamic)
    _box(stage, f"{prim_path}/board", cfg.size, (0.0, 0.0, 0.0), cfg.body_color,
         cfg.contact_offset, material=mat)
    if cfg.tile_color is not None:
        zt = cfg.size[2] / 2 + cfg.tile_size[2] / 2
        for tag, sgn in (("top", 1.0), ("bot", -1.0)):
            _box_visual(stage, f"{prim_path}/tile_{tag}", cfg.tile_size,
                        (0.0, 0.0, sgn * zt), cfg.tile_color)
    return root


def _studio_spawner_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "studio" not in _SPAWNER_CACHE:

        @configclass
        class SlabEaselStudioSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_studio)
            deck_size: tuple = (1.05, 1.40, 0.10)
            z_work: float = 0.02
            x_bp: float = -0.24
            apron_size: tuple = (0.12, 1.30, 0.02)
            bay_ys: tuple = (-0.36, 0.0, 0.36)
            bay_plate_size: tuple = (0.14, 0.36, 0.02)
            panel_size: tuple = (0.02, 0.26, 0.32)
            panel_beta_deg: float = 10.0
            trim_size: tuple = (0.022, 0.27, 0.035)
            rail_size: tuple = (0.035, 1.30, 0.02)
            rail_x0: float = -0.0825
            rail_pitch: float = 0.065
            n_rails: int = 6
            mu_static: float = 0.35
            mu_dynamic: float = 0.32
            deck_color: tuple = (0.50, 0.48, 0.45)
            apron_color: tuple = (0.35, 0.34, 0.33)
            rail_color: tuple = (0.40, 0.38, 0.36)
            panel_color: tuple = (0.30, 0.30, 0.32)
            bay_colors: tuple = ((0.55, 0.20, 0.20), (0.20, 0.50, 0.22), (0.20, 0.25, 0.55))
            trim_colors: tuple = ((0.85, 0.10, 0.10), (0.10, 0.65, 0.15), (0.10, 0.20, 0.85))
            contact_offset: float = 0.002

        _SPAWNER_CACHE["studio"] = SlabEaselStudioSpawnerCfg

    return _SPAWNER_CACHE["studio"](
        mass_props=sim_utils.MassPropertiesCfg(mass=80.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        deck_size=c.deck_size, z_work=c.z_work, x_bp=c.x_bp, apron_size=c.apron_size,
        bay_ys=c.bay_ys, bay_plate_size=c.bay_plate_size, panel_size=c.panel_size,
        panel_beta_deg=c.panel_beta_deg, trim_size=c.trim_size, rail_size=c.rail_size,
        rail_x0=c.rail_x0, rail_pitch=c.rail_pitch, n_rails=c.n_rails,
        mu_static=c.deck_mu_static, mu_dynamic=c.deck_mu_dynamic,
        deck_color=c.deck_color, apron_color=c.apron_color, rail_color=c.rail_color,
        panel_color=c.panel_color, bay_colors=c.bay_colors, trim_colors=c.trim_colors,
        contact_offset=c.contact_offset,
    )


def _slab_spawner_cfg(c: Any, body_color: tuple, tile_color) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "slab" not in _SPAWNER_CACHE:

        @configclass
        class SlabEaselSlabSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slab)
            size: tuple = (0.24, 0.20, 0.03)
            tile_size: tuple = (0.19, 0.15, 0.004)
            mass: float = 2.2
            mu_static: float = 0.45
            mu_dynamic: float = 0.42
            body_color: tuple = (0.72, 0.66, 0.55)
            tile_color: Any = None
            contact_offset: float = 0.002

        _SPAWNER_CACHE["slab"] = SlabEaselSlabSpawnerCfg

    return _SPAWNER_CACHE["slab"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.slab_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        size=c.slab_size, tile_size=c.tile_size, mass=c.slab_mass,
        mu_static=c.slab_mu_static, mu_dynamic=c.slab_mu_dynamic,
        body_color=body_color, tile_color=tile_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SlabEaselSceneCfg(BaseCfg):
    """Config for `SlabEaselScene`. Honesty knobs asserted in `__post_init__`: the
    seated slab fits its panel (shorter than the backrest, narrower than its width),
    the slabs are jaw-ungraspable as free bodies while a fingertip fits a rail gap, the
    spawn band is outside every staging window (null policy earns nothing), the seat
    windows of adjacent bays are disjoint, and the erect band sits strictly between the
    staging tilt and the final lean tilt."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_normal_deg: float = tunable(12.0)  # face-normal alignment cone at the panel (deg)
    seat_x_tol: float = tunable(0.035)  # seat window: |x - x_bp| below this (m)
    seat_y_tol: float = tunable(0.10)  # seat window: |y - bay_y| below this (m)
    seat_z_lo: float = tunable(0.08)  # seat window: z - z_work within [lo, hi] (m)
    seat_z_hi: float = tunable(0.14)
    erect_lo_deg: float = tunable(30.0)  # rising-tilt accumulator band (deg)
    erect_hi_deg: float = tunable(70.0)
    erect_req_deg: float = tunable(25.0)  # accumulated rising tilt required in the band
    dtilt_lo: float = tunable(5e-4)  # plausible per-substep tilt rise (rad): (lo, hi)
    dtilt_hi: float = tunable(0.08)
    acc_dpos: float = tunable(0.008)  # max per-substep centre travel while accruing (m)
    stage_r: float = tunable(0.10)  # staging latch: xy within this of the staging point
    stage_tilt_deg: float = tunable(15.0)  # ... and lying flat (tilt below this)
    stage_dx: float = tunable(0.128)  # staging point: x_bp + stage_dx (foot ~8 mm off panel)
    flat_tilt_deg: float = tunable(15.0)  # blank slab: must stay below this tilt
    flat_z_hi: float = tunable(0.07)  # ... resting near the work plane (z - z_work band)
    still_dpos: float = tunable(4e-4)  # stillness: per-substep pose deltas (m, rad) ...
    still_dang: float = tunable(4e-3)
    still_steps: int = tunable(30)  # ... for this many consecutive substeps

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    studio_jitter: float = tunable(0.06)  # uniform +/- xy jitter of the rack at reset (m)
    studio_yaw_deg: float = tunable(180.0)  # uniform +/- rack yaw (free — read the layout)
    slab_x_lo: float = tunable(0.04)  # slab spawn band, rack frame x in [lo, hi]
    slab_x_hi: float = tunable(0.22)
    slab_y_jitter: float = tunable(0.01)  # jitter around the permuted y-slot
    min_sep: float = tunable(0.315)  # min pairwise centre distance (rejection resample)

    # --- info: studio structure (rack frame: z = 0 at deck top, -x = toward the bays) --------
    deck_size: tuple = info((1.05, 1.40, 0.10))
    z_work: float = info(0.02)  # the flush work plane (apron + bay plates + rail tops)
    x_bp: float = info(-0.24)  # panel foot line (panel front face meets the work plane)
    apron_size: tuple = info((0.12, 1.30, 0.02))
    bay_ys: tuple = info((-0.36, 0.0, 0.36))  # RED, GREEN, BLUE bay centres
    bay_plate_size: tuple = info((0.14, 0.36, 0.02))
    panel_size: tuple = info((0.02, 0.26, 0.32))  # thickness, width, height
    panel_beta_deg: float = info(10.0)  # backrest lean-back angle
    trim_size: tuple = info((0.022, 0.27, 0.035))
    rail_size: tuple = info((0.035, 1.30, 0.02))
    rail_x0: float = info(-0.0825)
    rail_pitch: float = info(0.065)  # 35 mm rail + 30 mm finger gap
    n_rails: int = info(6)
    slot_ys: tuple = info((-0.48, -0.16, 0.16, 0.48))  # spawn y-slots (permuted)
    # --- info: slabs -------------------------------------------------------------------------
    slab_size: tuple = info((0.24, 0.20, 0.03))
    slab_mass: float = info(2.2)
    tile_size: tuple = info((0.19, 0.15, 0.004))
    # --- info: friction + contact ------------------------------------------------------------
    deck_mu_static: float = info(0.35)
    deck_mu_dynamic: float = info(0.32)
    slab_mu_static: float = info(0.45)
    slab_mu_dynamic: float = info(0.42)
    contact_offset: float = info(0.002)
    # --- info: colors ------------------------------------------------------------------------
    deck_color: tuple = info((0.50, 0.48, 0.45))
    apron_color: tuple = info((0.35, 0.34, 0.33))
    rail_color: tuple = info((0.40, 0.38, 0.36))
    panel_color: tuple = info((0.30, 0.30, 0.32))
    bay_colors: tuple = info(((0.55, 0.20, 0.20), (0.20, 0.50, 0.22), (0.20, 0.25, 0.55)))
    trim_colors: tuple = info(((0.85, 0.10, 0.10), (0.10, 0.65, 0.15), (0.10, 0.20, 0.85)))
    slab_body_color: tuple = info((0.72, 0.66, 0.55))
    blank_color: tuple = info((0.55, 0.55, 0.55))

    # Derived (filled in __post_init__).
    beta: float = field(default=None, init=False)  # panel lean (rad)
    erect_req: float = field(default=None, init=False)  # accumulator threshold (rad)
    seat_x_exp: float = field(default=None, init=False)  # expected seated centre (rack)
    seat_z_exp: float = field(default=None, init=False)
    lean_tilt_deg: float = field(default=None, init=False)  # final lean tilt (90 - beta)
    balance_deg: float = field(default=None, init=False)  # tip-over angle atan(L/T)

    def __post_init__(self) -> None:
        self.beta = math.radians(self.panel_beta_deg)
        self.erect_req = math.radians(self.erect_req_deg)
        length, width, thick = self.slab_size
        # seated centre: foot edge at (x_bp, z_work), face flush on the panel
        ux, uz = -math.sin(self.beta), math.cos(self.beta)
        nx, nz = math.cos(self.beta), math.sin(self.beta)
        self.seat_x_exp = self.x_bp + (length / 2) * ux + (thick / 2) * nx
        self.seat_z_exp = self.z_work + (length / 2) * uz + (thick / 2) * nz
        self.lean_tilt_deg = 90.0 - self.panel_beta_deg
        self.balance_deg = math.degrees(math.atan2(length, thick))

        # seated slab fits its panel: shorter than the backrest, narrower than its width
        assert length <= self.panel_size[2] - 0.06, "slab must be shorter than the panel"
        assert width <= self.panel_size[1] + 0.001, "slab must not out-span the panel"
        # seat window contains the expected seated centre
        assert abs(self.seat_x_exp - self.x_bp) < self.seat_x_tol - 0.01
        assert self.seat_z_lo + 0.01 < self.seat_z_exp - self.z_work < self.seat_z_hi - 0.01
        # adjacent bays' seat windows are disjoint
        assert 2 * self.seat_y_tol < (self.bay_ys[1] - self.bay_ys[0]) - 0.05
        # erect band strictly between staging tilt and the final lean tilt
        assert (self.stage_tilt_deg < self.erect_lo_deg < self.erect_hi_deg
                < self.lean_tilt_deg < self.balance_deg)
        assert self.erect_req_deg <= (self.erect_hi_deg - self.erect_lo_deg) - 5.0
        # slabs are jaw-ungraspable as free bodies (80 mm parallel jaw), yet thin enough
        # to edge-pinch at a rail gap; rails leave real finger gaps
        assert min(length, width) > 0.075 and thick < 0.075
        assert self.rail_pitch - self.rail_size[0] >= 0.025, "finger gaps must be real"
        # work plane is flush: apron, bay plates, rail tops all at z_work
        assert self.apron_size[2] == self.bay_plate_size[2] == self.rail_size[2]
        assert abs(self.rail_size[2] - self.z_work) < 1e-9
        # spawn band: on the rail field, outside every staging window (null earns nothing)
        stage_x = self.x_bp + self.stage_dx
        assert self.slab_x_lo - stage_x > self.stage_r + 0.04
        half_diag = math.hypot(length / 2, width / 2)
        assert self.min_sep >= 2 * half_diag, "min_sep must exclude any overlap"
        assert self.slab_x_hi + half_diag < self.deck_size[0] / 2
        assert max(self.slot_ys) + self.slab_y_jitter + half_diag < self.deck_size[1] / 2
        assert self.slab_x_hi - self.slab_x_lo >= 0.12, "x band must let resampling work"
        # slabs fit between the panel foot and the deck front while staged
        assert stage_x - length / 2 > self.x_bp + 0.004, "staged foot must clear the panel"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("slab_easel")
class SlabEaselScene(BaseScene):
    cfg: SlabEaselSceneCfg

    COLOR_NAMES = ("RED", "GREEN", "BLUE")

    def __init__(self, cfg: SlabEaselSceneCfg | None = None) -> None:
        super().__init__(cfg or SlabEaselSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        deck_h = c.deck_size[2]
        z0 = deck_h + c.z_work + c.slab_size[2] / 2 + 0.003
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
            "studio": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Studio",
                spawn=_studio_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, deck_h)),
            ),
        }
        tiles = [c.trim_colors[0], c.trim_colors[1], c.trim_colors[2], None]
        bodies = [c.slab_body_color] * 3 + [c.blank_color]
        names = ["slab_red", "slab_green", "slab_blue", "slab_blank"]
        for k, (nm, bc, tc) in enumerate(zip(names, bodies, tiles)):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + nm.title().replace("_", ""),
                spawn=_slab_spawner_cfg(c, bc, tc),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.10, c.slot_ys[k], z0)),
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
        self.studio: RigidObject = env.iscene["studio"]
        self.slabs: list[RigidObject] = [
            env.iscene["slab_red"], env.iscene["slab_green"],
            env.iscene["slab_blue"], env.iscene["slab_blank"]]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.staged_l = torch.zeros(n, 3, device=dev)
        self.erect_acc = torch.zeros(n, 3, device=dev)
        self.seated_l = torch.zeros(n, 3, device=dev)
        self.still_cnt = torch.zeros(n, device=dev)
        self.prev_pos = torch.zeros(n, 4, 3, device=dev)
        self.prev_quat = torch.zeros(n, 4, 4, device=dev)
        self.prev_tilt = torch.zeros(n, 4, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: studio with xy jitter + free yaw; the four slabs flat on the
        rail field in a permuted assignment of the four y-slots (xy jitter, free yaw,
        rejection-resampled pairwise separation); latches, accumulators, stillness
        counter and prev-pose buffers reset."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        deck_h = c.deck_size[2]

        # --- studio: xy jitter + free yaw ---
        sxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.studio_jitter
        syaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.studio_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = sxy
        st[:, 2] = deck_h
        st[:, 3] = torch.cos(syaw / 2)
        st[:, 6] = torch.sin(syaw / 2)
        st[:, 0:3] += origin
        self.studio.write_root_state_to_sim(st, env_ids)
        ch, sh = torch.cos(syaw), torch.sin(syaw)

        # --- slabs: permuted slots + rejection-resampled xy, free yaw, flat on rails ---
        z0 = deck_h + c.z_work + c.slab_size[2] / 2 + 0.003
        loc = torch.zeros(m, 4, 2, device=dev)
        for j in range(m):
            perm = torch.randperm(4, device=dev)
            placed: list[tuple[float, float]] = []
            for k in range(4):
                sy = c.slot_ys[int(perm[k])]
                lx, ly = 0.0, 0.0
                for _try in range(50):
                    lx = c.slab_x_lo + float(torch.rand((), device=dev)) \
                        * (c.slab_x_hi - c.slab_x_lo)
                    ly = sy + (float(torch.rand((), device=dev)) * 2 - 1) * c.slab_y_jitter
                    if all((lx - px) ** 2 + (ly - py) ** 2 >= c.min_sep ** 2
                           for px, py in placed):
                        break
                placed.append((lx, ly))
                loc[j, k, 0], loc[j, k, 1] = lx, ly
        for k, body in enumerate(self.slabs):
            wyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sxy[:, 0] + loc[:, k, 0] * ch - loc[:, k, 1] * sh
            st[:, 1] = sxy[:, 1] + loc[:, k, 0] * sh + loc[:, k, 1] * ch
            st[:, 2] = z0
            st[:, 3] = torch.cos(wyaw / 2)
            st[:, 6] = torch.sin(wyaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)
            self.prev_pos[env_ids, k] = st[:, 0:3]
            self.prev_quat[env_ids, k] = st[:, 3:7]
            self.prev_tilt[env_ids, k] = 0.0

        # --- latches + counters ---
        self.staged_l[env_ids] = 0.0
        self.erect_acc[env_ids] = 0.0
        self.seated_l[env_ids] = 0.0
        self.still_cnt[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {"studio": self.studio.data.root_state_w[env_ids].clone(),
               "staged_l": self.staged_l[env_ids].clone(),
               "erect_acc": self.erect_acc[env_ids].clone(),
               "seated_l": self.seated_l[env_ids].clone(),
               "still_cnt": self.still_cnt[env_ids].clone(),
               "prev_pos": self.prev_pos[env_ids].clone(),
               "prev_quat": self.prev_quat[env_ids].clone(),
               "prev_tilt": self.prev_tilt[env_ids].clone()}
        for k, body in enumerate(self.slabs):
            out[f"slab_{k}"] = body.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.studio.write_root_state_to_sim(state["studio"], env_ids)
        for k, body in enumerate(self.slabs):
            body.write_root_state_to_sim(state[f"slab_{k}"], env_ids)
        self.staged_l[env_ids] = state["staged_l"]
        self.erect_acc[env_ids] = state["erect_acc"]
        self.seated_l[env_ids] = state["seated_l"]
        self.still_cnt[env_ids] = state["still_cnt"]
        self.prev_pos[env_ids] = state["prev_pos"]
        self.prev_quat[env_ids] = state["prev_quat"]
        self.prev_tilt[env_ids] = state["prev_tilt"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low studio deck ({c.deck_size[0] * 100:.0f} x {c.deck_size[1] * 100:.0f} cm, "
            f"{c.deck_size[2] * 100:.0f} cm tall) carries a display rack. Along its back "
            f"edge stand three DISPLAY BAYS, {abs(c.bay_ys[0] - c.bay_ys[1]) * 100:.0f} cm "
            f"apart: each has a color-tinted floor plate at the foot of a dark backrest "
            f"PANEL ({c.panel_size[1] * 100:.0f} cm wide, {c.panel_size[2] * 100:.0f} cm "
            f"tall, leaning back {c.panel_beta_deg:.0f} degrees) capped by a bright color "
            f"TRIM bar: RED, GREEN, BLUE in order. In front of the bays, six low rails "
            f"({c.rail_size[0] * 1000:.0f} mm wide, {c.rail_size[2] * 1000:.0f} mm tall, "
            f"{(c.rail_pitch - c.rail_size[0]) * 1000:.0f} mm gaps) run across the deck; "
            f"their tops, the bay plates and the back apron form one flush work surface. "
            f"On the rail field lie FOUR flat slabs ({c.slab_size[0] * 100:.0f} x "
            f"{c.slab_size[1] * 100:.0f} x {c.slab_size[2] * 100:.0f} cm, "
            f"{c.slab_mass:.1f} kg): three carry a RED / GREEN / BLUE tile on both large "
            f"faces, one is plain gray. Their positions and yaws — and the rack's own "
            f"position and yaw — vary per episode, so read the layout from the scene.\n"
            f"Goal: ERECT each colored slab in its color-matched bay so it stands leaning "
            f"flush against that bay's backrest panel (either colored face showing), and "
            f"leave the plain gray slab lying flat on the work surface. The slabs are too "
            f"wide to grasp: slide one to the foot of its bay (about "
            f"{c.stage_dx * 100:.0f} cm in front of the panel foot, long edge parallel to "
            f"the panel), get a fingertip under its front edge through a rail gap, and "
            f"hinge it up about its grounded back edge — past upright (it tips over its "
            f"foot at ~{c.balance_deg:.0f} degrees) — so it falls back onto the panel and "
            f"rests at ~{c.lean_tilt_deg:.0f} degrees.\n"
            f"Judged when everything is at rest: each colored slab seated against its own "
            f"panel (face flush within {c.seat_normal_deg:.0f} degrees, centre at the "
            f"panel foot), the gray slab flat. The tilt must be GAINED in the bay — a slab "
            f"placed flat in a bay, propped elsewhere, or leaned against the wrong panel "
            f"counts for nothing, and credit only accrues while the slab physically "
            f"rotates up through the {c.erect_lo_deg:.0f}-{c.erect_hi_deg:.0f} degree band "
            f"inside its bay."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the red, green, and blue slabs up in their matching display bays, "
            "each leaning back against its color-matched panel. Leave the plain gray "
            "slab lying flat."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _yaw_of(self, body) -> torch.Tensor:
        q = body.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def rack_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(...,3) world points -> studio body frame (origin = deck-top centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        sq = self.studio.data.root_quat_w
        sp = self.studio.data.root_pos_w
        if p_w.dim() == 3:  # (n, k, 3)
            k = p_w.shape[1]
            return quat_apply_inverse(
                sq.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
                (p_w - sp.unsqueeze(1)).reshape(-1, 3)).view(p_w.shape)
        return quat_apply_inverse(sq, p_w - sp)

    def panel_normal_w(self) -> torch.Tensor:
        """(n,3) world unit vector of the panels' front (outward) normal."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        nl = torch.tensor([math.cos(c.beta), 0.0, math.sin(c.beta)],
                          device=self.env.device)
        return quat_apply(self.studio.data.root_quat_w,
                          nl.unsqueeze(0).expand(self.env.num_envs, 3))

    def _all_pos(self) -> torch.Tensor:
        return torch.stack([b.data.root_pos_w for b in self.slabs], dim=1)  # (n,4,3)

    def _all_quat(self) -> torch.Tensor:
        return torch.stack([b.data.root_quat_w for b in self.slabs], dim=1)  # (n,4,4)

    @staticmethod
    def _tilt_of(q: torch.Tensor) -> torch.Tensor:
        """(...,4) quats -> (...) tilt of the slab's face normal from vertical (rad),
        folded over both faces (|n_z|): flat = 0, leaning on the panel = 90 - beta."""
        nz = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
        return torch.acos(nz.abs().clamp(0.0, 1.0))

    def tilts(self) -> torch.Tensor:
        return self._tilt_of(self._all_quat())  # (n,4)

    def normal_w(self, k: int) -> torch.Tensor:
        """(n,3) world unit face normal of slab k."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device)
        return quat_apply(self.slabs[k].data.root_quat_w,
                          ez.unsqueeze(0).expand(self.env.num_envs, 3))

    # ----- predicates -------------------------------------------------------------------------
    def seated_now(self, k: int) -> torch.Tensor:
        """(n,) bool: colored slab k rests in ITS bay's seat window with its face
        normal aligned to the panel normal (either face — the alignment is folded)."""
        c = self.cfg
        loc = self.rack_local(self.slabs[k].data.root_pos_w)
        align = (self.normal_w(k) * self.panel_normal_w()).sum(dim=-1).abs()
        return ((align > math.cos(math.radians(c.seat_normal_deg)))
                & ((loc[:, 0] - c.x_bp).abs() < c.seat_x_tol)
                & ((loc[:, 1] - c.bay_ys[k]).abs() < c.seat_y_tol)
                & ((loc[:, 2] - c.z_work) > c.seat_z_lo)
                & ((loc[:, 2] - c.z_work) < c.seat_z_hi))

    def staged_now(self, k: int) -> torch.Tensor:
        """(n,) bool: colored slab k lying flat at ITS bay's staging point."""
        c = self.cfg
        loc = self.rack_local(self.slabs[k].data.root_pos_w)
        tilt = self._tilt_of(self.slabs[k].data.root_quat_w)
        dx = loc[:, 0] - (c.x_bp + c.stage_dx)
        dy = loc[:, 1] - c.bay_ys[k]
        return ((tilt < math.radians(c.stage_tilt_deg))
                & (dx ** 2 + dy ** 2 < c.stage_r ** 2)
                & (loc[:, 2] - c.z_work > -0.01) & (loc[:, 2] - c.z_work < 0.05))

    def distractor_flat(self) -> torch.Tensor:
        """(n,) bool: the blank slab lies flat on the work surface, on the deck."""
        c = self.cfg
        loc = self.rack_local(self.slabs[3].data.root_pos_w)
        tilt = self._tilt_of(self.slabs[3].data.root_quat_w)
        return ((tilt < math.radians(c.flat_tilt_deg))
                & (loc[:, 2] - c.z_work > -0.010) & (loc[:, 2] - c.z_work < c.flat_z_hi)
                & (loc[:, 0].abs() < c.deck_size[0] / 2)
                & (loc[:, 1].abs() < c.deck_size[1] / 2))

    def erect_ok(self) -> torch.Tensor:
        """(n,3) bool: the rising-tilt account crossed the requirement per colored slab."""
        return self.erect_acc >= self.cfg.erect_req

    def settled(self) -> torch.Tensor:
        """(n,) bool: all four slabs pose-still for `still_steps` consecutive substeps
        (velocity readings lie for bodies resting on edges — use pose deltas)."""
        return self.still_cnt >= self.cfg.still_steps

    # ----- progress latches (every physics substep) -------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch staging, accrue the RISING-TILT account (only while the slab is in its
        bay and the per-substep motion is physically plausible — a teleported upright
        pose is a radians-scale jump and accrues nothing), latch the seated-still
        state, and run the consecutive-substep stillness counter."""
        c = self.cfg
        pos = self._all_pos()
        quat = self._all_quat()
        tilt = self._tilt_of(quat)  # (n,4)
        dp = (pos - self.prev_pos).norm(dim=-1)  # (n,4)
        qd = (quat * self.prev_quat).sum(dim=-1).abs().clamp(0.0, 1.0)
        dang = 2.0 * torch.acos(qd)  # (n,4)
        dtilt = tilt - self.prev_tilt  # (n,4)
        loc = self.rack_local(pos)  # (n,4,3)

        lo, hi = math.radians(c.erect_lo_deg), math.radians(c.erect_hi_deg)
        for k in range(3):
            self.staged_l[:, k] = torch.maximum(
                self.staged_l[:, k], self.staged_now(k).float())
            in_bay = (((loc[:, k, 0] - c.x_bp) > -0.04)
                      & ((loc[:, k, 0] - c.x_bp) < 0.15)
                      & ((loc[:, k, 1] - c.bay_ys[k]).abs() < 0.12)
                      & ((loc[:, k, 2] - c.z_work) > -0.01)
                      & ((loc[:, k, 2] - c.z_work) < 0.16))
            band = (tilt[:, k] > lo) & (tilt[:, k] < hi)
            plaus = ((dtilt[:, k] > c.dtilt_lo) & (dtilt[:, k] < c.dtilt_hi)
                     & (dp[:, k] < c.acc_dpos))
            gate = (in_bay & band & plaus).float()
            self.erect_acc[:, k] = self.erect_acc[:, k] + gate * dtilt[:, k]
            seated_still = (self.seated_now(k) & (self.erect_acc[:, k] >= c.erect_req)
                            & (dang[:, k] < 5e-3) & (dp[:, k] < 5e-4))
            self.seated_l[:, k] = torch.maximum(
                self.seated_l[:, k], seated_still.float())

        still = ((dp < c.still_dpos).all(dim=1) & (dang < c.still_dang).all(dim=1))
        self.still_cnt = torch.where(still, self.still_cnt + 1.0,
                                     torch.zeros_like(self.still_cnt))
        self.prev_pos = pos.clone()
        self.prev_quat = quat.clone()
        self.prev_tilt = tilt.clone()

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(n,) bool: every colored slab seated against its own panel with its erect
        account earned (the tilt was physically swept up through the band in the bay),
        the blank slab flat on the deck, everything pose-still."""
        ok = self.distractor_flat() & self.settled()
        erect = self.erect_ok()
        for k in range(3):
            ok = ok & self.seated_now(k) & erect[:, k]
        return ok

    def score(self) -> torch.Tensor:
        """(n,) float in [0,1]: per colored slab 0.06 * staged + 0.12 * erect-band
        progress + 0.10 * seated-still, all latched (non-decreasing), capped at 0.85;
        exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy (carry
        the object to the target region and set it down flat) at most stages one slab."""
        prog = (self.erect_acc / self.cfg.erect_req).clamp(0.0, 1.0)
        base = (0.06 * self.staged_l + 0.12 * prog + 0.10 * self.seated_l).sum(dim=1)
        base = base.clamp(0.0, 0.85)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="slab_easel", robot="null"))
