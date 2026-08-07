"""PuddingSiftScene — pour a sealed bin's mixed contents onto a sieve grate so the
small pearls fall through into the hopper, then move the retained pudding ball to the
dish and park the bin (sim_gen task `libero_pick_chocolate_pudding_i46`).

Derived from libero/libero_pick_chocolate_pudding, but STRATEGICALLY different: the
seed's plan is single-object pick-and-place among distractors — grasp the chocolate
pudding, carry it, lower it into an open-topped basket; the distractors are scenery
and one bbox check ends the task. Here NOTHING can be placed into the goal container
from above by hand: the hopper is sealed on every face except its top, which is a
GRATE of square holes ringed by a low lip — the holes pass an 18 mm pearl but not the
42 mm pudding ball and certainly not a gripper finger. The judged set is MIXED and
starts SEALED inside a roofed carry-bin whose only exit is a side pour-port behind a
raised sill: contents leave only when the bin is lifted and deliberately tilted. The
plan is therefore PARTITION BY GEOMETRIC FILTRATION: carry the loaded bin over the
grate, tilt-pour the whole mixed batch through the port onto the grate, let hole
geometry sort the batch (pearls thread the holes under gravity, the pudding ball is
retained on top), then transfer the retained pudding ball to the open dish, and set
the emptied bin down upright on the floor. A solver needs a different PLAN (carry a
loaded container, controlled tilt-pour, filtration with contamination stakes,
retained-item retrieval, tool set-down) and different code structure (per-pearl
containment accounting in the hopper frame, pour control) — not a bbox check on one
transported body.

Judged in body frames (hopper is a kinematic fixture; bin and dish are free bodies).
success() iff, settled:
  - every PRESENT pearl rests INSIDE the hopper basin (hopper-local xy within the
    interior, z between the basin floor and the grate underside — reachable only
    through the grate holes);
  - the pudding ball rests INSIDE the dish (dish-local containment, on its floor);
    pearls poured into the dish are contamination: any pearl in the dish is a pearl
    not in the basin, so success is impossible until it is fished out and re-sifted;
  - the bin is parked on the floor: upright and resting at ground level (not left on
    the grate, not tipped over, not hovering).
score() is latched every physics substep: 0.10 * the pudding ball ever LEFT the bin
+ 0.50 * fraction of present pearls ever IN the basin + 0.20 * the pudding ball ever
in the dish, capped at 0.80; exactly 1.0 iff success(). Doing nothing scores ~0; the
seed's whole strategy (put the target object into the open goal vessel, ignore the
rest) scores <= 0.30 and never succeeds.

Assets are fully procedural (no external files):
  - hopper: ONE kinematic fixture — a sealed 186 x 186 x 220 mm bin whose only
    opening is the top GRATE: 5 x 5 square holes (26 mm) between 6 mm orange bars,
    ringed by a 12 mm dark lip that keeps poured balls from rolling off the edge;
  - bin: DYNAMIC — a roofed 116 mm steel-blue box, interior 100 x 100 x 90 mm, with
    a 72 mm wide pour-port on ONE side wall behind a 28 mm sill plus a matching
    notch in the roof edge above it (nothing escapes a level bin; the 42 mm ball
    clears the L-shaped corner window when tilted past vertical) and a 12 mm
    handle bar bridged 40 mm above the roof for a parallel jaw;
  - dish: DYNAMIC — an open white dish, interior 110 x 110 mm, 30 mm walls;
  - pudding ball: 42 mm brown sphere, 60 g (fits the jaw, NOT the 26 mm holes);
  - pearls: up to six 18 mm white spheres, 10 g (fit the holes with 8 mm slack).
Explicit small contact offsets (1.5 mm): the default would eat the 4 mm hole-vs-ball
margins. Physics materials are bound to every custom spawner body (the unbound
default ~0.5 friction makes pearls hang on the grate bars).

Per-episode randomization (verified by readback in smoke): hopper xy + yaw, a coin
flip for which y-side holds the dish (the bin mirrors it), bin xy + free yaw, dish
xy, and the PRESENT pearl count (3..6; absent pearls park in a far ground depot).
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _bind_material(stage, root_prim, mat_path: str, static: float, dynamic: float) -> None:
    """Author a physics material and bind it at the body root (covers all child
    colliders). Unbound custom-spawner colliders default to ~0.5 friction, which
    makes pearls hang on the grate bars instead of sliding into the holes."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    mp = mat.GetPrim()
    UsdPhysics.MaterialAPI.Apply(mp)
    api = UsdPhysics.MaterialAPI(mp)
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(root_prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC hopper at `prim_path`. Local origin = centre of the
    bottom face on the ground. Sealed floor + four walls; the top is the GRATE
    (6 bars each way leaving 5 x 5 square holes) ringed by a low lip."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    co = cfg.contact_offset
    ih = cfg.basin_half          # interior half-extent (grate span half)
    wt = cfg.wall_t
    oh = ih + wt                 # outer half-extent
    ft = cfg.floor_t
    z_wall0, z_wall1 = ft, cfg.grate_z0
    wall_h = z_wall1 - z_wall0
    bar, t_g = cfg.bar_w, cfg.grate_t
    zg = cfg.grate_z0 + t_g / 2
    lip_h, lip_t = cfg.lip_h, cfg.wall_t
    body_c, grate_c, lip_c = cfg.hopper_color, cfg.grate_color, cfg.lip_color

    _box(stage, f"{prim_path}/floor", (2 * oh, 2 * oh, ft), (0, 0, ft / 2), body_c, co)
    zc = (z_wall0 + z_wall1) / 2
    _box(stage, f"{prim_path}/wall_e", (wt, 2 * oh, wall_h), (ih + wt / 2, 0, zc), body_c, co)
    _box(stage, f"{prim_path}/wall_w", (wt, 2 * oh, wall_h), (-ih - wt / 2, 0, zc), body_c, co)
    _box(stage, f"{prim_path}/wall_n", (2 * ih, wt, wall_h), (0, ih + wt / 2, zc), body_c, co)
    _box(stage, f"{prim_path}/wall_s", (2 * ih, wt, wall_h), (0, -ih - wt / 2, zc), body_c, co)
    # grate bars: 6 each way, leaving 5 x 5 holes of cfg.hole_w
    for k, xb in enumerate(cfg.bar_centers):
        _box(stage, f"{prim_path}/bar_x{k}", (bar, 2 * ih, t_g), (xb, 0, zg), grate_c, co)
        _box(stage, f"{prim_path}/bar_y{k}", (2 * ih, bar, t_g), (0, xb, zg), grate_c, co)
    # retaining lip above the grate
    zl = cfg.grate_z1 + lip_h / 2
    _box(stage, f"{prim_path}/lip_e", (lip_t, 2 * oh, lip_h), (ih + lip_t / 2, 0, zl), lip_c, co)
    _box(stage, f"{prim_path}/lip_w", (lip_t, 2 * oh, lip_h), (-ih - lip_t / 2, 0, zl), lip_c, co)
    _box(stage, f"{prim_path}/lip_n", (2 * ih, lip_t, lip_h), (0, ih + lip_t / 2, zl), lip_c, co)
    _box(stage, f"{prim_path}/lip_s", (2 * ih, lip_t, lip_h), (0, -ih - lip_t / 2, zl), lip_c, co)
    _bind_material(stage, root, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC carry-bin at `prim_path`. Local origin = centre of the
    bottom face. Floor, roof, three solid walls; the +x wall is the pour-port: a
    full-width sill strip below and two pillars flanking the opening; the roof's
    port-side edge carries a matching NOTCH so the exit is an L-shaped corner
    window (a ball pressed against the roof while tilted past vertical would
    otherwise permanently graze the opening's top edge). A handle bar bridges two
    posts above the roof."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(2.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset
    ihx = cfg.bin_inner / 2       # interior half-extent
    wt = cfg.bin_wall_t
    oh = ihx + wt                 # outer half-extent
    hin = cfg.bin_inner_h         # interior height
    z0 = wt                       # interior floor top
    z1 = wt + hin                 # interior ceiling (roof underside)
    col, hcol = cfg.bin_color, cfg.handle_color
    pw2 = cfg.port_w / 2
    sill = cfg.sill_h

    _box(stage, f"{prim_path}/floor", (2 * oh, 2 * oh, wt), (0, 0, wt / 2), col, co)
    # roof: main panel + two side strips, leaving a notch_d x port_w notch at the
    # +x (port) edge — the exit's top clearance
    nd = cfg.notch_d
    x0 = ihx - nd                 # notch inner edge
    _box(stage, f"{prim_path}/roof", (x0 + oh, 2 * oh, wt),
         ((x0 - oh) / 2, 0, z1 + wt / 2), col, co)
    _box(stage, f"{prim_path}/roof_strip_n", (oh - x0, oh - pw2, wt),
         ((x0 + oh) / 2, pw2 + (oh - pw2) / 2, z1 + wt / 2), col, co)
    _box(stage, f"{prim_path}/roof_strip_s", (oh - x0, oh - pw2, wt),
         ((x0 + oh) / 2, -pw2 - (oh - pw2) / 2, z1 + wt / 2), col, co)
    zc = (z0 + z1) / 2
    _box(stage, f"{prim_path}/wall_w", (wt, 2 * oh, hin), (-ihx - wt / 2, 0, zc), col, co)
    _box(stage, f"{prim_path}/wall_n", (2 * ihx, wt, hin), (0, ihx + wt / 2, zc), col, co)
    _box(stage, f"{prim_path}/wall_s", (2 * ihx, wt, hin), (0, -ihx - wt / 2, zc), col, co)
    # +x port wall: sill strip + two pillars flanking the opening
    _box(stage, f"{prim_path}/sill", (wt, 2 * oh, sill),
         (ihx + wt / 2, 0, z0 + sill / 2), col, co)
    pil_w = oh - pw2              # each pillar's y extent
    pil_h = hin - sill
    zp = z0 + sill + pil_h / 2
    _box(stage, f"{prim_path}/pillar_n", (wt, pil_w, pil_h),
         (ihx + wt / 2, pw2 + pil_w / 2, zp), col, co)
    _box(stage, f"{prim_path}/pillar_s", (wt, pil_w, pil_h),
         (ihx + wt / 2, -pw2 - pil_w / 2, zp), col, co)
    # handle: two posts + a bar bridged above the roof, spanning Y so nothing
    # overhangs the roof notch at the +x edge
    ph, bw = cfg.handle_post_h, cfg.handle_bar_w
    z_roof = z1 + wt
    _box(stage, f"{prim_path}/post_n", (bw, bw, ph), (0, 0.030, z_roof + ph / 2), hcol, co)
    _box(stage, f"{prim_path}/post_s", (bw, bw, ph), (0, -0.030, z_roof + ph / 2), hcol, co)
    _box(stage, f"{prim_path}/bar", (bw, 0.072, bw),
         (0, 0, z_roof + ph + bw / 2), hcol, co)
    _bind_material(stage, root, f"{prim_path}/physmat", cfg.mu_static, cfg.mu_dynamic)
    return root


def _spawn_dish(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC open dish at `prim_path`. Local origin = centre of the
    bottom face."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(1.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset
    ihx = cfg.dish_inner / 2
    wt = cfg.dish_wall_t
    oh = ihx + wt
    ft = cfg.dish_floor_t
    wh = cfg.dish_wall_h
    col = cfg.dish_color
    _box(stage, f"{prim_path}/floor", (2 * oh, 2 * oh, ft), (0, 0, ft / 2), col, co)
    zc = ft + wh / 2
    _box(stage, f"{prim_path}/wall_e", (wt, 2 * oh, wh), (ihx + wt / 2, 0, zc), col, co)
    _box(stage, f"{prim_path}/wall_w", (wt, 2 * oh, wh), (-ihx - wt / 2, 0, zc), col, co)
    _box(stage, f"{prim_path}/wall_n", (2 * ihx, wt, wh), (0, ihx + wt / 2, zc), col, co)
    _box(stage, f"{prim_path}/wall_s", (2 * ihx, wt, wh), (0, -ihx - wt / 2, zc), col, co)
    _bind_material(stage, root, f"{prim_path}/physmat", 0.60, 0.50)
    return root


def _make_spawner(key: str, func: Callable, defaults: dict, cfg_vals: dict,
                  *, kinematic: bool, mass: float) -> Any:
    """Build (and cache the class of) a compound-spawner cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:
        ns = {"func": clone(func), **defaults}
        ns["__annotations__"] = {"func": Callable,
                                 **{k: type(v) for k, v in defaults.items()}}
        _SPAWNER_CACHE[key] = configclass(type(f"Spawner_{key}", (RigidObjectSpawnerCfg,), ns))
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True) if kinematic \
        else sim_utils.RigidBodyPropertiesCfg()
    return _SPAWNER_CACHE[key](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass), rigid_props=rigid, **cfg_vals)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PuddingSiftCfg(BaseCfg):
    """Config for `PuddingSiftScene`. Honesty knobs asserted in `__post_init__`: the
    grate holes pass a pearl with real slack and can NEVER pass the pudding ball;
    the port passes the pudding ball only when the bin is tilted past its sill; the
    dish holds the ball; the ball fits the jaw; the lip retains poured balls."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)     # max |lin vel| of judged bodies (m/s)
    basin_margin: float = tunable(0.004)  # in-basin: xy inside the interior by this (m)
    dish_margin: float = tunable(0.004)   # in-dish: xy inside the interior by this (m)
    park_tilt_deg: float = tunable(15.0)  # parked bin: axis within this of world-up
    park_z_tol: float = tunable(0.012)    # parked bin: bottom within this of the ground

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.04)     # uniform +/- xy jitter of the hopper (m)
    fix_yaw_deg: float = tunable(15.0)    # uniform +/- hopper yaw
    side_flip: bool = tunable(True)       # coin-flip which y-side holds the dish
    bin_scatter: float = tunable(0.04)    # +/- xy scatter of the bin spawn (m)
    bin_yaw_deg: float = tunable(180.0)   # +/- free bin yaw (the port heading varies)
    dish_scatter: float = tunable(0.03)   # +/- xy scatter of the dish spawn (m)
    min_present: int = tunable(3)         # lower bound of the sampled pearl count

    # --- tunable: placement (world xy nominals) ----------------------------------------------
    fix_pos: tuple = tunable((0.55, 0.0))    # hopper centre
    bin_pos: tuple = tunable((0.26, -0.28))  # bin nominal (y * side: opposite the dish)
    dish_pos: tuple = tunable((0.28, 0.30))  # dish nominal (y * side per episode)

    # --- info: hopper structure --------------------------------------------------------------
    basin_half: float = info(0.083)   # interior half-extent (166 mm grate span)
    wall_t: float = info(0.010)
    floor_t: float = info(0.010)
    grate_z0: float = info(0.200)     # grate underside (basin ceiling)
    grate_t: float = info(0.008)
    hole_w: float = info(0.026)       # square hole width (5 x 5 holes)
    bar_w: float = info(0.006)        # grate bar width
    lip_h: float = info(0.012)        # retaining lip above the grate
    # --- info: bin (the sealed carrier) ------------------------------------------------------
    bin_inner: float = info(0.100)    # interior square width
    bin_wall_t: float = info(0.008)
    bin_inner_h: float = info(0.090)  # interior height (floor top -> roof underside)
    port_w: float = info(0.072)       # pour-port opening width (+x wall)
    notch_d: float = info(0.030)      # roof notch depth over the port (exit clearance)
    sill_h: float = info(0.028)       # sill above the interior floor: nothing escapes level
    handle_post_h: float = info(0.040)
    handle_bar_w: float = info(0.012)  # 12 mm bar: a comfortable parallel-jaw grasp
    bin_mass: float = info(0.35)
    # --- info: dish --------------------------------------------------------------------------
    dish_inner: float = info(0.110)
    dish_wall_t: float = info(0.008)
    dish_wall_h: float = info(0.030)
    dish_floor_t: float = info(0.006)
    dish_mass: float = info(0.40)
    # --- info: balls -------------------------------------------------------------------------
    pud_r: float = info(0.021)        # 42 mm pudding ball: fits the jaw, NOT the holes
    pud_mass: float = info(0.060)
    pearl_r: float = info(0.009)      # 18 mm pearls: fit the holes with 8 mm slack
    pearl_mass: float = info(0.010)
    n_pearls: int = info(6)           # pearl slots (present subset sampled per episode)
    jaw_span: float = info(0.080)     # the Franka jaw that must hold ball / bar
    depot_pos: tuple = info((1.8, 1.8))  # far ground depot for absent pearls
    # --- info: colors / contact --------------------------------------------------------------
    hopper_color: tuple = info((0.55, 0.55, 0.58))
    grate_color: tuple = info((0.90, 0.45, 0.10))
    lip_color: tuple = info((0.25, 0.25, 0.28))
    bin_color: tuple = info((0.30, 0.40, 0.55))
    handle_color: tuple = info((0.15, 0.16, 0.18))
    dish_color: tuple = info((0.92, 0.92, 0.90))
    pud_color: tuple = info((0.45, 0.26, 0.13))
    pearl_color: tuple = info((0.95, 0.95, 0.97))
    contact_offset: float = info(0.0015)  # default offsets would eat the 4 mm hole margin
    mu_static: float = info(0.30)     # fixture friction: pearls slide off bars into holes
    mu_dynamic: float = info(0.25)

    # Derived (filled in __post_init__).
    bar_centers: tuple = field(default=None, init=False)   # grate bar centre offsets
    hole_centers: tuple = field(default=None, init=False)  # 1-D hole centre offsets
    grate_z1: float = field(default=None, init=False)      # grate top
    bin_h: float = field(default=None, init=False)         # bin outer height incl. handle

    def __post_init__(self) -> None:
        pitch = self.hole_w + self.bar_w  # 32 mm
        # 5 holes + 6 bars must tile the interior span exactly
        assert abs(2 * self.basin_half - (5 * self.hole_w + 6 * self.bar_w)) < 1e-9, \
            "grate does not tile the basin span"
        self.hole_centers = tuple((i - 2) * pitch for i in range(5))
        self.bar_centers = tuple(-self.basin_half + self.bar_w / 2 + i * pitch
                                 for i in range(6))
        self.grate_z1 = self.grate_z0 + self.grate_t
        self.bin_h = 2 * self.bin_wall_t + self.bin_inner_h \
            + self.handle_post_h + self.handle_bar_w

        # --- filtration honesty: the grate passes pearls, retains the pudding ball ---
        assert self.hole_w >= 2 * self.pearl_r + 0.006, "pearls must pass the holes with slack"
        assert 2 * self.pud_r >= self.hole_w + 0.012, "the pudding ball must NEVER pass a hole"
        assert self.hole_w < 0.030, "no finger fits a hole (the basin stays sealed to the jaw)"
        # --- port honesty: sealed level, passes the ball tilted ---
        assert self.sill_h >= self.pud_r + 0.005, "the sill must retain a level pudding ball"
        assert self.sill_h >= self.pearl_r + 0.010, "the sill must retain level pearls"
        # A ball pressed flat against a slot passes only if its CENTRE fits the
        # opening minus its radius: demand a real +/-15 mm lateral window.
        assert self.port_w >= 2 * self.pud_r + 0.0295, \
            "the port must pass the pudding ball with a real lateral window"
        assert self.bin_inner_h - self.sill_h >= 2 * self.pud_r + 0.015, \
            "the port opening must pass the pudding ball"
        assert self.notch_d >= 0.5 * self.pud_r, \
            "the roof notch must clear a ball riding the roof while tilted past vertical"
        assert self.bin_inner / 2 - self.notch_d > 0.0, "the roof must still cover the bin"
        # The notch stays sealed to the JAW: an 80 mm parallel jaw cannot open
        # around the 42 mm ball inside a 72 mm window, and the contents rest
        # ~90 mm below the roof — beyond fingertip reach through it.
        assert self.port_w < self.jaw_span, "the notch must not admit an open jaw"
        # --- retention / placement honesty ---
        assert self.lip_h >= 0.010, "the lip must retain poured balls on the grate"
        assert self.lip_h < self.pud_r, "the lip must not swallow the retained ball"
        assert self.dish_inner >= 2 * self.pud_r + 0.020, "the dish must take the ball easily"
        assert self.dish_floor_t + self.pud_r < self.dish_floor_t + self.dish_wall_h, \
            "the dish walls must retain the ball"
        # --- embodiment honesty: the jaw can hold everything it must hold ---
        assert 2 * self.pud_r <= self.jaw_span - 0.015, "the pudding ball must fit the jaw"
        assert self.handle_bar_w <= self.jaw_span - 0.030, "the handle bar must fit the jaw"
        assert self.handle_post_h >= 0.030, "finger clearance under the handle bar"
        # --- the basin actually separates the two height bands ---
        assert self.grate_z0 - self.floor_t > 4 * self.pearl_r + 0.05, \
            "the basin must be deep enough that in-basin vs on-grate are distinct"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("pudding_sift")
class PuddingSiftScene(BaseScene):
    cfg: PuddingSiftCfg

    def __init__(self, cfg: PuddingSiftCfg | None = None) -> None:
        super().__init__(cfg or PuddingSiftCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def ball_cfg(radius: float, mass: float, color: tuple, mu: float) -> Any:
            return sim_utils.SphereCfg(
                radius=radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.08, angular_damping=0.30,
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu, dynamic_friction=0.8 * mu, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        fx, fy = c.fix_pos
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
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=_make_spawner(
                    "hopper", _spawn_hopper,
                    {"basin_half": 0.083, "wall_t": 0.010, "floor_t": 0.010,
                     "grate_z0": 0.200, "grate_t": 0.008, "hole_w": 0.026,
                     "bar_w": 0.006, "lip_h": 0.012,
                     "bar_centers": (0.0,), "grate_z1": 0.208,
                     "hopper_color": (0.5, 0.5, 0.5), "grate_color": (0.9, 0.4, 0.1),
                     "lip_color": (0.2, 0.2, 0.2), "contact_offset": 0.0015,
                     "mu_static": 0.30, "mu_dynamic": 0.25},
                    {"basin_half": c.basin_half, "wall_t": c.wall_t, "floor_t": c.floor_t,
                     "grate_z0": c.grate_z0, "grate_t": c.grate_t, "hole_w": c.hole_w,
                     "bar_w": c.bar_w, "lip_h": c.lip_h,
                     "bar_centers": c.bar_centers, "grate_z1": c.grate_z1,
                     "hopper_color": c.hopper_color, "grate_color": c.grate_color,
                     "lip_color": c.lip_color, "contact_offset": c.contact_offset,
                     "mu_static": c.mu_static, "mu_dynamic": c.mu_dynamic},
                    kinematic=True, mass=20.0),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, 0.0)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=_make_spawner(
                    "bin", _spawn_bin,
                    {"bin_inner": 0.100, "bin_wall_t": 0.008, "bin_inner_h": 0.090,
                     "port_w": 0.072, "notch_d": 0.030, "sill_h": 0.028,
                     "handle_post_h": 0.040, "handle_bar_w": 0.012, "mass": 0.35,
                     "bin_color": (0.3, 0.4, 0.55), "handle_color": (0.15, 0.16, 0.18),
                     "contact_offset": 0.0015, "mu_static": 0.22, "mu_dynamic": 0.18},
                    {"bin_inner": c.bin_inner, "bin_wall_t": c.bin_wall_t,
                     "bin_inner_h": c.bin_inner_h, "port_w": c.port_w,
                     "notch_d": c.notch_d, "sill_h": c.sill_h,
                     "handle_post_h": c.handle_post_h, "handle_bar_w": c.handle_bar_w,
                     "mass": c.bin_mass, "bin_color": c.bin_color,
                     "handle_color": c.handle_color, "contact_offset": c.contact_offset,
                     # smooth plastic: a rocked pour must SLIDE contents past the
                     # pillar corners (lateral accel tan(rock) must beat mu)
                     "mu_static": 0.22, "mu_dynamic": 0.18},
                    kinematic=False, mass=c.bin_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bin_pos[0], c.bin_pos[1], 0.002)),
            ),
            "dish": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Dish",
                spawn=_make_spawner(
                    "dish", _spawn_dish,
                    {"dish_inner": 0.110, "dish_wall_t": 0.008, "dish_wall_h": 0.030,
                     "dish_floor_t": 0.006, "mass": 0.40,
                     "dish_color": (0.9, 0.9, 0.9), "contact_offset": 0.0015},
                    {"dish_inner": c.dish_inner, "dish_wall_t": c.dish_wall_t,
                     "dish_wall_h": c.dish_wall_h, "dish_floor_t": c.dish_floor_t,
                     "mass": c.dish_mass, "dish_color": c.dish_color,
                     "contact_offset": c.contact_offset},
                    kinematic=False, mass=c.dish_mass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.dish_pos[0], c.dish_pos[1], 0.002)),
            ),
            "pudding": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pudding",
                spawn=ball_cfg(c.pud_r, c.pud_mass, c.pud_color, 0.50),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] - 0.3, c.depot_pos[1], c.pud_r + 0.002)),
            ),
        }
        for i in range(c.n_pearls):
            out[f"pearl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pearl_" + str(i),
                spawn=ball_cfg(c.pearl_r, c.pearl_mass, c.pearl_color, 0.30),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.1 * (i % 3), c.depot_pos[1] + 0.1 * (i // 3),
                         c.pearl_r + 0.002)),
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
        self.hopper: RigidObject = env.iscene["hopper"]
        self.bin: RigidObject = env.iscene["bin"]
        self.dish: RigidObject = env.iscene["dish"]
        self.pudding: RigidObject = env.iscene["pudding"]
        self.pearls: list[RigidObject] = [env.iscene[f"pearl_{i}"] for i in range(c.n_pearls)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.present = torch.ones(n, c.n_pearls, dtype=torch.bool, device=dev)
        self.side = torch.ones(n, device=dev)          # +1: dish on +y side
        self.rel_latch = torch.zeros(n, device=dev)    # pudding ever left the bin
        self.pearl_latch = torch.zeros(n, c.n_pearls, device=dev)  # ever in the basin
        self.pud_latch = torch.zeros(n, device=dev)    # pudding ever in the dish

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hopper with xy jitter + yaw, a coin flip for which y-side
        holds the dish (the bin mirrors it), bin xy scatter + FREE yaw (the port
        heading varies), dish xy scatter; a present pearl subset (min_present..n) is
        sampled — present pearls and the pudding ball are racked INSIDE the sealed
        bin, absent pearls park in the far ground depot; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, xy: torch.Tensor, z, half: torch.Tensor | None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            if half is None:
                st[:, 3] = 1.0
            else:
                st[:, 3], st[:, 6] = torch.cos(half), torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- hopper: xy jitter + yaw ---
        fix_xy = torch.tensor(c.fix_pos, device=dev).expand(m, 2).clone()
        fix_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        write(self.hopper, fix_xy, 0.0, fyaw / 2)

        # --- side coin flip: dish on +/- y, bin mirrored ---
        if c.side_flip:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               -torch.ones(m, device=dev), torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self.side[env_ids] = side

        dish_xy = torch.tensor(c.dish_pos, device=dev).expand(m, 2).clone()
        dish_xy[:, 1] *= side
        dish_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.dish_scatter
        write(self.dish, dish_xy, 0.002, None)

        bin_xy = torch.tensor(c.bin_pos, device=dev).expand(m, 2).clone()
        bin_xy[:, 1] *= side  # bin_pos.y is negative: the bin lands OPPOSITE the dish
        bin_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bin_scatter
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_yaw_deg)
        write(self.bin, bin_xy, 0.002, byaw / 2)

        # --- pearl subset sampling ---
        k = torch.randint(c.min_present, c.n_pearls + 1, (m,), device=dev)
        rank = torch.rand(m, c.n_pearls, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        # --- rack the contents inside the sealed bin (bin-local slots -> world) ---
        cy, sy = torch.cos(byaw), torch.sin(byaw)

        def to_world(lx: float, ly: float) -> torch.Tensor:
            out = torch.zeros(m, 2, device=dev)
            out[:, 0] = bin_xy[:, 0] + lx * cy - ly * sy
            out[:, 1] = bin_xy[:, 1] + lx * sy + ly * cy
            return out

        z_floor = 0.002 + c.bin_wall_t
        jit = 0.003
        pud_xy = to_world(0.012, 0.0) + (torch.rand(m, 2, device=dev) * 2 - 1) * jit
        write(self.pudding, pud_xy, z_floor + c.pud_r + 0.003, None)
        slots = ((-0.030, 0.032), (0.002, 0.036), (0.032, 0.030),
                 (-0.032, -0.030), (0.000, -0.036), (0.030, -0.032))
        depot = torch.tensor(c.depot_pos, device=dev)
        for i, body in enumerate(self.pearls):
            in_xy = to_world(*slots[i]) + (torch.rand(m, 2, device=dev) * 2 - 1) * jit
            park = depot.expand(m, 2).clone()
            park[:, 0] += 0.10 * (i % 3)
            park[:, 1] += 0.10 * (i // 3)
            pres = self.present[env_ids, i].unsqueeze(1)
            xy = torch.where(pres, in_xy, park)
            z = torch.where(self.present[env_ids, i],
                            torch.full((m,), z_floor + c.pearl_r + 0.003, device=dev),
                            torch.full((m,), c.pearl_r + 0.002, device=dev))
            write(body, xy, z, None)

        # --- latches ---
        self.rel_latch[env_ids] = 0.0
        self.pearl_latch[env_ids] = 0.0
        self.pud_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "dish": self.dish.data.root_state_w[env_ids].clone(),
            "pudding": self.pudding.data.root_state_w[env_ids].clone(),
            "pearls": [b.data.root_state_w[env_ids].clone() for b in self.pearls],
            "present": self.present[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "rel_latch": self.rel_latch[env_ids].clone(),
            "pearl_latch": self.pearl_latch[env_ids].clone(),
            "pud_latch": self.pud_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        self.dish.write_root_state_to_sim(state["dish"], env_ids)
        self.pudding.write_root_state_to_sim(state["pudding"], env_ids)
        for b, st in zip(self.pearls, state["pearls"]):
            b.write_root_state_to_sim(st, env_ids)
        self.present[env_ids] = state["present"]
        self.side[env_ids] = state["side"]
        self.rel_latch[env_ids] = state["rel_latch"]
        self.pearl_latch[env_ids] = state["pearl_latch"]
        self.pud_latch[env_ids] = state["pud_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Three vessels stand on the floor of the work area. (1) A grey HOPPER "
            f"({2 * (c.basin_half + c.wall_t) * 1000:.0f} mm square, "
            f"{(c.grate_z1 + c.lip_h) * 1000:.0f} mm tall), sealed on every face except "
            f"the top, which is an ORANGE GRATE: a 5 x 5 grid of "
            f"{c.hole_w * 1000:.0f} mm square holes between {c.bar_w * 1000:.0f} mm "
            f"bars, ringed by a low dark lip — the holes are the ONLY way into the "
            f"hopper. (2) A steel-blue roofed BIN ({(c.bin_inner + 2 * c.bin_wall_t) * 1000:.0f} "
            f"mm square) with a {c.handle_bar_w * 1000:.0f} mm black HANDLE BAR bridged "
            f"above its roof and a single {c.port_w * 1000:.0f} mm wide PORT opening in "
            f"one side wall behind a {c.sill_h * 1000:.0f} mm sill, with a matching "
            f"notch in the roof edge above the port: the bin is sealed while level "
            f"and pours only when lifted and tilted port-down past vertical. Sealed "
            f"inside it are one BROWN pudding ball ({2 * c.pud_r * 1000:.0f} mm — too "
            f"big for the grate holes, small enough for a gripper) and between "
            f"{c.min_present} and {c.n_pearls} WHITE pearls "
            f"({2 * c.pearl_r * 1000:.0f} mm — they fit through the grate holes); count "
            f"what comes out. (3) An open WHITE DISH "
            f"({c.dish_inner * 1000:.0f} mm interior, {c.dish_wall_h * 1000:.0f} mm walls).\n"
            f"Goal: every pearl must end up INSIDE the hopper (they can only get there "
            f"by falling through the grate holes), the brown pudding ball must end up "
            f"resting in the white dish, and the bin must finish parked upright on the "
            f"floor. The intended route: grasp the bin by its handle bar, carry it over "
            f"the grate, tilt it port-down to pour the whole mixed batch onto the grate "
            f"— the holes pass the pearls and retain the pudding ball; nudge any pearl "
            f"that rests on a bar until it drops through; then set the bin down upright "
            f"on the floor, and pick the retained pudding ball off the grate and place "
            f"it in the dish. No fixed order is required, but pearls poured into the "
            f"dish or dropped on the floor do not count until they end up inside the "
            f"hopper, and a pudding ball left on the grate is not done. Judged only "
            f"when everything is settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pour the bin's contents onto the hopper's orange grate so every white "
            "pearl falls through into the hopper, place the brown pudding ball in the "
            "white dish, and set the bin down upright on the floor. Pearls anywhere "
            "but inside the hopper fail the task."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> `body`'s frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(body.data.root_quat_w, p_w - body.data.root_pos_w)

    def _pearl_pos(self) -> torch.Tensor:
        """(N, P, 3) world pearl positions."""
        return torch.stack([b.data.root_pos_w for b in self.pearls], dim=1)

    # ----- predicates -------------------------------------------------------------------------
    def pearls_in_basin(self) -> torch.Tensor:
        """(N, P) bool: pearl centre inside the hopper basin (hopper frame): xy within
        the interior, z between the basin floor and the grate underside. The basin is
        sealed except for the grate, so this is real filtration containment."""
        c = self.cfg
        pos = self._pearl_pos()
        n, p = pos.shape[0], pos.shape[1]
        loc = self._local(self.hopper, pos.view(n * p, 3))
        loc = loc.view(n, p, 3)
        lim = c.basin_half - c.basin_margin
        return ((loc[:, :, 0].abs() < lim) & (loc[:, :, 1].abs() < lim)
                & (loc[:, :, 2] > c.floor_t - 0.002)
                & (loc[:, :, 2] < c.grate_z0 - 0.004))

    def _hopper_local1(self, body) -> torch.Tensor:
        return self._local(self.hopper, body.data.root_pos_w)

    def pud_on_grate(self) -> torch.Tensor:
        """(N,) bool: pudding ball retained ON the grate (above it, within the lip)."""
        c = self.cfg
        loc = self._hopper_local1(self.pudding)
        return ((loc[:, 0].abs() < c.basin_half) & (loc[:, 1].abs() < c.basin_half)
                & (loc[:, 2] > c.grate_z1) & (loc[:, 2] < c.grate_z1 + 2 * c.pud_r))

    def pud_in_dish(self) -> torch.Tensor:
        """(N,) bool: pudding ball centre inside the dish (dish frame), resting near
        its floor — real containment, not proximity."""
        c = self.cfg
        loc = self._local(self.dish, self.pudding.data.root_pos_w)
        lim = c.dish_inner / 2 - c.dish_margin
        return ((loc[:, 0].abs() < lim) & (loc[:, 1].abs() < lim)
                & (loc[:, 2] > c.dish_floor_t - 0.002)
                & (loc[:, 2] < c.dish_floor_t + c.pud_r + 0.020))

    def pud_in_bin(self) -> torch.Tensor:
        """(N,) bool: pudding ball centre inside the bin interior (bin frame)."""
        c = self.cfg
        loc = self._local(self.bin, self.pudding.data.root_pos_w)
        h = c.bin_inner / 2 + 0.004
        return ((loc[:, 0].abs() < h) & (loc[:, 1].abs() < h)
                & (loc[:, 2] > 0.0) & (loc[:, 2] < c.bin_wall_t + c.bin_inner_h + 0.004))

    def pearls_in_bin(self) -> torch.Tensor:
        """(N, P) bool: pearl centres inside the bin interior (bin frame)."""
        c = self.cfg
        pos = self._pearl_pos()
        n, p = pos.shape[0], pos.shape[1]
        loc = self._local(self.bin, pos.view(n * p, 3)).view(n, p, 3)
        h = c.bin_inner / 2 + 0.004
        return ((loc[:, :, 0].abs() < h) & (loc[:, :, 1].abs() < h)
                & (loc[:, :, 2] > 0.0)
                & (loc[:, :, 2] < c.bin_wall_t + c.bin_inner_h + 0.004))

    def bin_parked(self) -> torch.Tensor:
        """(N,) bool: bin upright (within `park_tilt_deg`), resting at ground level
        (root/bottom within `park_z_tol` of the floor — rejects a bin left on the
        grate or perched on a vessel), and slow."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.bin.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.park_tilt_deg))
        z = (self.bin.data.root_pos_w - self.env_origins)[:, 2]
        on_ground = z.abs() < c.park_z_tol
        slow = self.bin.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return upright & on_ground & slow

    def settled(self) -> torch.Tensor:
        """(N,) bool: pudding, dish and every present pearl below `settle_lin`."""
        c = self.cfg
        pv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.pearls], dim=1)
        pearls_still = ((pv < c.settle_lin) | ~self.present).all(dim=1)
        return (pearls_still
                & (self.pudding.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.dish.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch release, per-pearl basin entry, and dish delivery each physics
        substep, so transient progress keeps its credit."""
        self.rel_latch = torch.maximum(self.rel_latch, (~self.pud_in_bin()).float())
        self.pearl_latch = torch.maximum(
            self.pearl_latch, (self.pearls_in_basin() & self.present).float())
        self.pud_latch = torch.maximum(self.pud_latch, self.pud_in_dish().float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: every present pearl inside the hopper basin + the pudding ball
        inside the dish + the bin parked upright on the floor, everything settled.
        (Any pearl in the dish or on the floor is a pearl NOT in the basin.)"""
        pearls_ok = (self.pearls_in_basin() | ~self.present).all(dim=1)
        return (pearls_ok & self.pud_in_dish() & self.bin_parked() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * pudding ever left the bin + 0.50 * latched
        fraction of present pearls ever in the basin + 0.20 * pudding ever in the
        dish, capped at 0.80; exactly 1.0 iff success(). Doing nothing scores ~0; the
        seed's whole strategy (drop the target into the open goal vessel, ignore the
        rest) tops out at 0.30."""
        n_pres = self.present.float().sum(dim=1).clamp(min=1.0)
        frac = (self.pearl_latch * self.present.float()).sum(dim=1) / n_pres
        base = (0.10 * self.rel_latch + 0.50 * frac + 0.20 * self.pud_latch).clamp(0.0, 0.80)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="pudding_sift", robot="null"))
