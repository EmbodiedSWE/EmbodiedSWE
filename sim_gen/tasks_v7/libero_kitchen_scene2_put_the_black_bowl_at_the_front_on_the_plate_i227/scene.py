"""ShutterVaultScene — slide the two captive shutter tiles down the L-channel to uncover the
sunken well, then drop the FRONT black bowl through the aperture onto the plate at the bottom.

Derived from libero_90 kitchen_scene2 "put the black bowl at the front on the plate". The seed
names one of three identical black bowls by table position ("the front one") and solves with a
single free grasp-carry-place onto an openly reachable plate. Here the bowl selection stays
(three identical black bowls in a row, target = the FRONT slot, identity permuted per episode)
but the plate is NOT reachable at reset and the path to it runs through a SLIDING-BLOCK PUZZLE:

  * the white plate lies at the bottom of a sunken WELL (50 mm deep, 120 x 120 mm aperture)
    recessed into a vault box on the counter. It is flush in the well (4 mm radial slack, no
    rim proud of the pocket) — it cannot be extracted, only served INTO;
  * the well aperture starts COVERED by a captive shutter tile (tile RED, cell A). A second
    tile (BLUE) occupies the middle cell B of an L-shaped three-cell channel (A - B - C);
    cell C is the only free cell. Overhanging rails cap the whole channel 4 mm above the
    flanges: the tiles can SLIDE cell-to-cell but can never be lifted out (a 140 mm flange
    under a 120 mm rail slot with 4 mm of headroom cannot tilt free — geometric proof in
    TASK.md);
  * the ORDER is forced by geometry, not by the rubric: RED can only leave the well cell
    into B, and B is occupied — so BLUE must first slide B -> C, then RED A -> B, and only
    then is the aperture open. success() has no order clause at all;
  * the goal state is unreachable by the seed plan: putting the front bowl "on the plate"
    where the plate is means lowering it INTO the well, impossible while the shutter covers
    the aperture; and the served bowl stands 8 mm PROUD of the deck, so the shutter can
    never be closed back over a served bowl (the out-of-order end state does not exist);
  * the two slides are physical pushes of free (captive) bodies through friction contact —
    there is no motor, no joint, no rubric hand-holding.

Success (pure settled-state predicate, all simultaneous):
  * the episode's FRONT bowl (front slot of the three-bowl row at reset) sits upright ON the
    plate (bottom at the plate top, centred within `bowl_xy_tol` in the plate frame);
  * the plate lies seated at the bottom of the well (vault-local xy within `plate_xy_tol`,
    at seat height, upright);
  * everything settled (tiles, plate, bowls slow).

Rubric (graded 0..1, latched in post_step, additive and monotone along the demonstrated
solution; 1.0 iff success()):
  0.00  nothing happened (reset state, null policy, and the seed-strategy end state)
  +0.25 UNLOCKED: BLUE tile parked in cell C (within `tile_latch_tol`), slow;
  +0.25 OPENED: RED tile parked in cell B (within `tile_latch_tol`), slow — geometrically
        requires BLUE to have vacated B first;
  +0.20 SERVED: the FRONT bowl deep inside the well (vault-local xy within `serve_xy_latch`
        of the well axis, base below `serve_z_latch`), slow — geometrically requires the
        aperture open;
  1.00  success() (overrides the 0.70 partial sum).

Honesty of the gates:
  * the seed-strategy end state (front bowl set down over the well ON the closed shutter) is
    constructible and rejected: the bowl base rests at deck+tile height (0.082), far above
    `serve_z_latch` (0.050) — no latch, no success;
  * z near-miss: a bowl standing on the open deck (base at 0.070) is rejected by the same
    gate; a bowl IN the well but on a missing/shifted plate fails the bowl-on-plate z band;
  * wrong bowl: only the FRONT-slot bowl counts (identity latched at reset, permuted per
    episode); an outer bowl served through the honest procedure scores the tile credits but
    never SERVED/success;
  * upright gates reject an inverted or tipped bowl that fell through the aperture;
  * `plate_in_well` ties the plate to its seat — a plate somehow removed and served on the
    counter (the literal seed goal geometry) is rejected;
  * every latch carries a velocity gate so fly-through states latch nothing.

Assets are fully procedural (no external files):
  * vault: ONE kinematic compound (well floor + four well walls + two deck plates + skirt +
    six channel walls + six overhanging rails) — recessed well under cell A, L-channel
    A(0,0) -> B(P,0) -> C(P,P), pitch P = 150 mm, rails 4 mm above the tile flanges;
  * tiles: two dynamic compounds (140 x 140 x 12 mm flange + a 22 mm knob post, the pinch
    handle), RED and BLUE, explicit MassAPI;
  * plate: plain white cylinder (112 x 10 mm) seated at the bottom of the well;
  * bowls: three identical BLACK octagonal bowls (outer ~84 mm, 48 mm tall, 6 mm rim wall —
    the rim is the pinch feature), one rigid body each, explicit MassAPI;
  * counter: kinematic slab (top at `surface_z`).

Per-episode randomization (verified by readback in the smoke): the vault's xy position
(+-`box_jitter`) and yaw (+-`box_yaw_deg`), and which physical bowl body occupies the FRONT
slot (permutation) plus per-bowl xy jitter and free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ explicit MassAPI on dynamics —
# overlapping child shapes would double-count density), child collider shapes. Authored through
# `clone()` so per-env replication is idempotent (no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(stage, prim_path: str, translation, orientation, *, kinematic: bool,
                mass: float | None = None, ang_damp: float = 0.0, lin_damp: float = 0.0):
    """Root Xform + RigidBodyAPI (+ MassAPI / damping / depenetration cap on dynamics)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    else:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px.CreateMaxDepenetrationVelocityAttr(0.5)
        px.CreateLinearDampingAttr(lin_damp)
        px.CreateAngularDampingAttr(ang_damp)
        px.CreateSolverPositionIterationCountAttr(16)
        # 4 velocity iterations: 1 leaves GPU TGS contacts with a constant phantom creep
        px.CreateSolverVelocityIterationCountAttr(4)
    return root


def _box(stage, prim_path: str, name: str, tx, ty, tz, sx, sy, sz, color,
         contact_offset: float | None, yaw: float = 0.0):
    """Child box; collision only when contact_offset is given (None -> visual only).

    xformOp order is translate -> orient -> scale, so the unit cube is scaled in its OWN
    frame first, then yawed (needed for the octagonal bowl rim segments)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
    if yaw:
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw / 2))))
    sxf.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
    return seg


def _cyl(stage, prim_path: str, name: str, tx, ty, tz, radius, height, color,
         contact_offset: float):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    cxf = UsdGeom.Xformable(cyl.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return cyl


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The kinematic vault: a recessed well under cell A of an L-shaped three-cell channel
    (cells A(0,0), B(P,0), C(P,P), pitch P), decked at `deck_top`, walled all around, with
    overhanging hold-down rails `rail_clear` above the tile flanges. Root origin at the cell-A
    centre on the counter top; local +x runs A -> B, local +y runs B -> C."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=True)
    co = float(cfg.contact_offset)
    P = cfg.pitch
    H = P / 2
    ap = cfg.aperture / 2

    # --- well (under cell A): floor + four walls, wall tops flush with the deck
    _box(stage, prim_path, "well_floor", 0.0, 0.0, cfg.well_floor_t / 2,
         P, P, cfg.well_floor_t, cfg.well_color, co)
    wz = (cfg.well_floor_t + cfg.deck_top) / 2
    wh = cfg.deck_top - cfg.well_floor_t
    wc = ap + cfg.well_wall_t / 2
    _box(stage, prim_path, "well_w", -wc, 0.0, wz, cfg.well_wall_t, P, wh, cfg.well_color, co)
    _box(stage, prim_path, "well_e", wc, 0.0, wz, cfg.well_wall_t, P, wh, cfg.well_color, co)
    _box(stage, prim_path, "well_s", 0.0, -wc, wz, P, cfg.well_wall_t, wh, cfg.well_color, co)
    _box(stage, prim_path, "well_n", 0.0, wc, wz, P, cfg.well_wall_t, wh, cfg.well_color, co)

    # --- decks over cells B and C (+ solid skirt below, down to the counter)
    dz = cfg.deck_top - cfg.deck_t / 2
    _box(stage, prim_path, "deck_b", P, 0.0, dz, P, P, cfg.deck_t, cfg.deck_color, co)
    _box(stage, prim_path, "deck_c", P, P, dz, P, P, cfg.deck_t, cfg.deck_color, co)
    sk_h = cfg.deck_top - cfg.deck_t
    _box(stage, prim_path, "skirt", P, H, sk_h / 2, P, 2 * P, sk_h, cfg.body_color, co)

    # --- channel walls (bounding the L; rise `chan_wall_h` from the counter, above the tiles)
    t = cfg.chan_wall_t
    ch = cfg.chan_wall_h
    cz = ch / 2
    walls = [
        ("cw_west",   -(H + t / 2), 0.0,          t,           P + 2 * t),
        ("cw_south",  H,            -(H + t / 2), 2 * P + 2 * t, t),
        ("cw_east",   P + H + t / 2, H,           t,           2 * P + 2 * t),
        ("cw_northc", P,            P + H + t / 2, P + 2 * t,   t),
        ("cw_notchn", -t / 2,       H + t / 2,    P + t,       t),
        ("cw_notchw", H - t / 2,    P + t / 2,    t,           P + t),
    ]
    for name, cx, cyy, sx, sy in walls:
        _box(stage, prim_path, name, cx, cyy, cz, sx, sy, ch, cfg.body_color, co)

    # --- hold-down rails: `rail_w` wide, overhanging the channel inward, underside
    # `rail_clear` above the flange tops — the tiles are CAPTIVE (slide only)
    rz = cfg.deck_top + cfg.tile_t + cfg.rail_clear + cfg.rail_t / 2
    ro = H + t - cfg.rail_w / 2  # rail centreline: outer edge flush with the wall outer face
    rails = [
        ("rail_west",   -ro,          0.0,         cfg.rail_w,  P + 2 * t),
        ("rail_south",  H,            -ro,         2 * P + 2 * t, cfg.rail_w),
        ("rail_east",   P + ro,       H,           cfg.rail_w,  2 * P + 2 * t),
        ("rail_northc", P,            P + ro,      P + 2 * t,   cfg.rail_w),
        ("rail_notchn", -t / 2,       H - t + cfg.rail_w / 2 + t / 2, P + t, cfg.rail_w),
        ("rail_notchw", H - t + cfg.rail_w / 2 + t / 2, P + t / 2,   cfg.rail_w, P + t),
    ]
    for name, cx, cyy, sx, sy in rails:
        _box(stage, prim_path, name, cx, cyy, rz, sx, sy, cfg.rail_t, cfg.rail_color, co)
    return root


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A shutter tile: square flange + a knob post (the pinch handle). Root origin at the
    flange centre."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.10, lin_damp=0.05)
    co = float(cfg.contact_offset)
    _box(stage, prim_path, "flange", 0.0, 0.0, 0.0, cfg.tile_s, cfg.tile_s, cfg.tile_t,
         cfg.color, co)
    _cyl(stage, prim_path, "knob", 0.0, 0.0, cfg.tile_t / 2 + cfg.knob_h / 2,
         cfg.knob_r, cfg.knob_h, cfg.knob_color, co)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A black octagonal bowl: floor disc + 8 vertical rim-wall boxes. Root origin at the
    floor slab centre (CoM low: MassAPI puts the mass at the body origin)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = _apply_root(stage, prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass, ang_damp=0.10, lin_damp=0.05)
    co = float(cfg.contact_offset)
    _cyl(stage, prim_path, "floor", 0.0, 0.0, 0.0, cfg.floor_r, cfg.floor_h, cfg.color, co)
    n_w = 8
    rw = cfg.floor_r - cfg.wall_t / 2 + 0.002  # wall mid-line; outer face ~ floor_r + t/2
    seg_l = 2 * rw * math.tan(math.pi / n_w) + 0.003
    for k in range(n_w):
        a = 2 * math.pi * k / n_w
        _box(stage, prim_path, f"wall_{k}", rw * math.cos(a), rw * math.sin(a),
             cfg.floor_h / 2 + cfg.wall_h / 2, cfg.wall_t, seg_l, cfg.wall_h,
             cfg.color, co, yaw=a)
    return root


def _mk_spawner(key: str, func: Callable, defaults: dict) -> Callable:
    """Build (once) and cache a RigidObjectSpawnerCfg subclass around `func`."""
    import isaaclab.sim as sim_utils  # noqa: F401  (kept for parity with sibling scenes)
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:
        ns = {"func": clone(func), **defaults}
        ns["__annotations__"] = {"func": Callable,
                                 **{k: type(v).__name__ for k, v in defaults.items()}}
        _SPAWNER_CACHE[key] = configclass(type(f"{key.title()}SpawnerCfg",
                                               (RigidObjectSpawnerCfg,), ns))
    return _SPAWNER_CACHE[key]


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class ShutterVaultSceneCfg(BaseCfg):
    """Config for `ShutterVaultScene`. Gate honesty margins are derived in the module
    docstring."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    tile_latch_tol: float = tunable(0.030)  # tile centre to cell centre (vault frame, m) for
    # the UNLOCKED / OPENED latches. Channel slack is +-5 mm, so a parked tile reads ~0.
    serve_xy_latch: float = tunable(0.045)  # front bowl centre to well axis (vault frame) for
    # SERVED. The well caps the true offset at 18 mm — loose on purpose; the z gate does the work.
    serve_z_latch: float = tunable(0.050)  # SERVED needs the bowl BASE below this (vault-local
    # z). On the plate: 0.030. On the closed shutter: 0.082. On the open deck: 0.070.
    bowl_xy_tol: float = tunable(0.035)  # bowl centre to plate axis, plate frame (m)
    bowl_z_tol: float = tunable(0.010)  # |bowl bottom - plate top| below this (m)
    bowl_tilt_max_deg: float = tunable(12.0)  # bowl local +z within this of world-up
    plate_xy_tol: float = tunable(0.020)  # plate centre to well axis (vault frame, m).
    # Seated slack is 4 mm; a plate anywhere but the well seat fails by construction.
    plate_z_tol: float = tunable(0.008)  # |plate centre - seat height| below this (m)
    plate_tilt_max_deg: float = tunable(10.0)  # plate axis within this of world-up
    settle_speed: float = tunable(0.05)  # max |lin v| of tiles/plate/bowls when judging (m/s)
    latch_speed_tile: float = tunable(0.05)  # tile latches only while the tile is this slow
    latch_speed_bowl: float = tunable(0.10)  # SERVED latches only while the bowl is this slow

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    box_jitter: float = tunable(0.025)  # uniform +- xy jitter of the vault at reset
    box_yaw_deg: float = tunable(20.0)  # uniform +- yaw of the vault at reset
    bowl_jitter: float = tunable(0.015)  # uniform +- xy jitter per bowl at reset
    shuffle_bowls: bool = tunable(True)  # per-episode slot permutation (target identity)

    # --- tunable: placement (counter frame; intended arm base at (-0.05, 0, surface_z)) ----------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    box_xy: tuple = tunable((0.32, 0.14))  # nominal vault root (cell-A centre) on the counter
    slot_x: tuple = tunable((0.16, 0.31, 0.46))  # bowl row slot x; FRONT = smallest x
    row_y: float = tunable(-0.16)  # bowl row y

    # --- info: structure --------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    pitch: float = info(0.150)  # cell pitch: A(0,0) -> B(P,0) -> C(P,P), vault-local
    aperture: float = info(0.120)  # well aperture (inner span, both axes)
    well_floor_t: float = info(0.020)  # well floor slab; floor top = 0.020
    well_wall_t: float = info(0.015)  # well wall; wall tops flush with the deck
    deck_top: float = info(0.070)  # deck height above the counter (= well wall tops)
    deck_t: float = info(0.010)
    chan_wall_t: float = info(0.012)
    chan_wall_h: float = info(0.092)  # channel walls: counter -> 10 mm above the flange tops
    rail_t: float = info(0.006)
    rail_w: float = info(0.027)  # rails overhang 15 mm past the channel wall inner face
    rail_clear: float = info(0.004)  # rail underside above the flange top: slide yes, lift no
    tile_s: float = info(0.140)  # flange side (140 in a 150 channel: 10 mm slack)
    tile_t: float = info(0.012)
    knob_r: float = info(0.011)  # knob post: the pinch handle (22 mm dia, 45 mm tall)
    knob_h: float = info(0.045)
    tile_mass: float = info(0.35)
    plate_r: float = info(0.056)  # plate: 4 mm radial slack inside the 120 mm aperture
    plate_h: float = info(0.010)
    plate_mass: float = info(0.20)
    plate_seat_z: float = info(0.025)  # seated plate CENTRE, vault-local (floor top + h/2)
    n_bowls: int = info(3)
    bowl_floor_r: float = info(0.040)
    bowl_floor_h: float = info(0.008)
    bowl_wall_t: float = info(0.006)
    bowl_wall_h: float = info(0.040)  # bowl total 48 mm: a served bowl stands 8 mm PROUD of
    # the deck, so the shutter cannot close back over it (out-of-order end state impossible)
    bowl_mass: float = info(0.15)
    contact_offset: float = info(0.0015)  # 2 x 0.0015 < the 4 mm rail gap and 4 mm plate slack
    body_color: tuple = info((0.42, 0.30, 0.18))
    deck_color: tuple = info((0.55, 0.42, 0.26))
    well_color: tuple = info((0.30, 0.30, 0.34))
    rail_color: tuple = info((0.60, 0.61, 0.64))
    tile0_color: tuple = info((0.75, 0.15, 0.12))  # RED: starts over the well (cell A)
    tile1_color: tuple = info((0.12, 0.25, 0.75))  # BLUE: starts in the middle cell (B)
    knob_color: tuple = info((0.85, 0.85, 0.88))
    plate_color: tuple = info((0.93, 0.92, 0.88))
    bowl_color: tuple = info((0.06, 0.06, 0.07))
    bench_color: tuple = info((0.35, 0.35, 0.38))


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("shutter_vault")
class ShutterVaultScene(BaseScene):
    cfg: ShutterVaultSceneCfg

    def __init__(self, cfg: ShutterVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or ShutterVaultSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, the kinematic vault, the two shutter tiles, the plate
        (nominally seated in the well) and the three bowls (reset() re-places everything and
        samples the randomization)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        bx, by = c.box_xy

        vault_cfg = _mk_spawner("vault", _spawn_vault, {
            "pitch": 0.150, "aperture": 0.120, "well_floor_t": 0.020, "well_wall_t": 0.015,
            "deck_top": 0.070, "deck_t": 0.010, "chan_wall_t": 0.012, "chan_wall_h": 0.092,
            "rail_t": 0.006, "rail_w": 0.027, "rail_clear": 0.004, "tile_t": 0.012,
            "contact_offset": 0.0015, "body_color": (0.42, 0.30, 0.18),
            "deck_color": (0.55, 0.42, 0.26), "well_color": (0.30, 0.30, 0.34),
            "rail_color": (0.60, 0.61, 0.64)})
        tile0_cfg = _mk_spawner("tile_red", _spawn_tile, {
            "tile_s": 0.140, "tile_t": 0.012, "knob_r": 0.011, "knob_h": 0.045,
            "contact_offset": 0.0015, "color": (0.75, 0.15, 0.12),
            "knob_color": (0.85, 0.85, 0.88)})
        tile1_cfg = _mk_spawner("tile_blue", _spawn_tile, {
            "tile_s": 0.140, "tile_t": 0.012, "knob_r": 0.011, "knob_h": 0.045,
            "contact_offset": 0.0015, "color": (0.12, 0.25, 0.75),
            "knob_color": (0.85, 0.85, 0.88)})
        bowl_cfg = _mk_spawner("bowl", _spawn_bowl, {
            "floor_r": 0.040, "floor_h": 0.008, "wall_t": 0.006, "wall_h": 0.040,
            "contact_offset": 0.0015, "color": (0.06, 0.06, 0.07)})

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, z0 / 2)),
            ),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=vault_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(bx, by, z0)),
            ),
            "tile_0": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TileRed",
                spawn=tile0_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx, by, z0 + c.deck_top + c.tile_t / 2 + 0.0005)),
            ),
            "tile_1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TileBlue",
                spawn=tile1_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + c.pitch, by, z0 + c.deck_top + c.tile_t / 2 + 0.0005)),
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.0015, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx, by, z0 + c.plate_seat_z + 0.0005)),
            ),
        }
        for i in range(c.n_bowls):
            out[f"bowl_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(i),
                spawn=bowl_cfg(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_x[i], c.row_y, z0 + c.bowl_floor_h / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, allocate the episode identity readbacks and the progress latches.
        No joints anywhere: the tiles are free bodies held captive by the rail geometry, so
        the vault CAN be re-posed per episode (no world-fixed joint anchors)."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.vault: RigidObject = env.iscene["vault"]
        self.tiles: list[RigidObject] = [env.iscene["tile_0"], env.iscene["tile_1"]]
        self.plate: RigidObject = env.iscene["plate"]
        self.bowls: list[RigidObject] = [env.iscene[f"bowl_{i}"] for i in range(c.n_bowls)]
        self.env_origins = env.iscene.env_origins
        # episode identity / randomization readback
        self.front_idx = torch.zeros(n, dtype=torch.long, device=dev)  # bowl body in FRONT slot
        self.slot_of = torch.zeros(n, c.n_bowls, dtype=torch.long, device=dev)
        self.box_pos_xy = torch.zeros(n, 2, device=dev)  # vault root xy (counter frame)
        self.box_yaw = torch.zeros(n, device=dev)  # vault yaw (rad, signed)
        # progress latches (post_step; cleared per reset)
        self.ever_unlocked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_served = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the vault pose (xy jitter + yaw), park RED over the well (A)
        and BLUE in the middle cell (B) in the vault frame, seat the plate at the bottom of
        the well, deal the three bowls over the row slots by a random permutation with jitter
        + free yaw, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z

        # --- vault pose: xy jitter + yaw
        bx = c.box_xy[0] + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
        by = c.box_xy[1] + (torch.rand(m, device=dev) * 2 - 1) * c.box_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.box_yaw_deg)
        self.box_pos_xy[env_ids] = torch.stack([bx, by], dim=1)
        self.box_yaw[env_ids] = yaw
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)

        def write(body, lx, ly, wz, with_yaw=True):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = bx + cy * lx - sy * ly
            st[:, 1] = by + sy * lx + cy * ly
            st[:, 2] = wz
            if with_yaw:
                st[:, 3] = qw
                st[:, 6] = qz
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.vault, 0.0, 0.0, torch.full((m,), z0, device=dev))
        tz = torch.full((m,), z0 + c.deck_top + c.tile_t / 2 + 0.0005, device=dev)
        write(self.tiles[0], 0.0, 0.0, tz)          # RED over the well (cell A)
        write(self.tiles[1], c.pitch, 0.0, tz)      # BLUE in the middle cell (B)
        write(self.plate, 0.0, 0.0,
              torch.full((m,), z0 + c.plate_seat_z + 0.0005, device=dev), with_yaw=False)

        # --- bowls: random slot permutation + jitter + free yaw (counter frame)
        if c.shuffle_bowls:
            slot_of = torch.rand(m, c.n_bowls, device=dev).argsort(dim=1)  # bowl -> slot
        else:
            slot_of = torch.arange(c.n_bowls, device=dev).expand(m, c.n_bowls).clone()
        self.slot_of[env_ids] = slot_of
        self.front_idx[env_ids] = (slot_of == 0).float().argmax(dim=1)
        slot_x = torch.tensor(c.slot_x, device=dev)
        for b, bowl in enumerate(self.bowls):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot_x[slot_of[:, b]]
            st[:, 1] = c.row_y
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            st[:, 2] = z0 + c.bowl_floor_h / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            bowl.write_root_state_to_sim(st, env_ids)

        for latch in (self.ever_unlocked, self.ever_opened, self.ever_served):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress milestones at sim rate. Velocity-gated so states flown through latch
        nothing; latched credit survives later mishaps, so along a correct trajectory the
        printed score never decreases."""
        c = self.cfg
        P = c.pitch
        loc0 = self._vault_local(self.tiles[0].data.root_pos_w)
        loc1 = self._vault_local(self.tiles[1].data.root_pos_w)
        cell_b = torch.tensor([P, 0.0], device=loc0.device)
        cell_c = torch.tensor([P, P], device=loc0.device)
        slow0 = self.tiles[0].data.root_lin_vel_w.norm(dim=-1) < c.latch_speed_tile
        slow1 = self.tiles[1].data.root_lin_vel_w.norm(dim=-1) < c.latch_speed_tile
        self.ever_unlocked |= ((loc1[:, :2] - cell_c).norm(dim=-1) < c.tile_latch_tol) & slow1
        self.ever_opened |= ((loc0[:, :2] - cell_b).norm(dim=-1) < c.tile_latch_tol) & slow0
        fpos = self._front_state()[0]
        locf = self._vault_local(fpos)
        in_xy = locf[:, :2].norm(dim=-1) < c.serve_xy_latch
        deep = (locf[:, 2] - c.bowl_floor_h / 2) < c.serve_z_latch
        fvel = torch.stack([b.data.root_lin_vel_w for b in self.bowls], dim=1)
        fspeed = fvel.norm(dim=-1).gather(1, self.front_idx.unsqueeze(1)).squeeze(1)
        self.ever_served |= in_xy & deep & (fspeed < c.latch_speed_bowl)

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "tiles": [t.data.root_state_w[env_ids].clone() for t in self.tiles],
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "bowls": [b.data.root_state_w[env_ids].clone() for b in self.bowls],
            "front_idx": self.front_idx[env_ids].clone(),
            "slot_of": self.slot_of[env_ids].clone(),
            "box_pos_xy": self.box_pos_xy[env_ids].clone(),
            "box_yaw": self.box_yaw[env_ids].clone(),
            "latches": torch.stack([self.ever_unlocked[env_ids], self.ever_opened[env_ids],
                                    self.ever_served[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        for t, st in zip(self.tiles, state["tiles"]):
            t.write_root_state_to_sim(st, env_ids)
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for b, st in zip(self.bowls, state["bowls"]):
            b.write_root_state_to_sim(st, env_ids)
        self.front_idx[env_ids] = state["front_idx"]
        self.slot_of[env_ids] = state["slot_of"]
        self.box_pos_xy[env_ids] = state["box_pos_xy"]
        self.box_yaw[env_ids] = state["box_yaw"]
        lat = state["latches"]
        self.ever_unlocked[env_ids] = lat[:, 0]
        self.ever_opened[env_ids] = lat[:, 1]
        self.ever_served[env_ids] = lat[:, 2]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A kitchen counter. Nearest you, THREE identical BLACK bowls "
            f"(~{2 * (c.bowl_floor_r + c.bowl_wall_t) * 1000:.0f} mm across, "
            f"{(c.bowl_floor_h + c.bowl_wall_h) * 1000:.0f} mm tall, "
            f"{c.bowl_wall_t * 1000:.0f} mm rim wall) stand in a row pointing away from you. "
            f"Beyond them sits a wooden VAULT box with an L-shaped channel of three square "
            f"cells on top, bounded by walls and capped along every edge by overhanging steel "
            f"RAILS. The corner cell nearest you is a sunken WELL "
            f"({c.aperture * 1000:.0f} x {c.aperture * 1000:.0f} mm aperture, "
            f"{(c.deck_top - c.well_floor_t) * 1000:.0f} mm deep) with a round WHITE plate "
            f"({2 * c.plate_r * 1000:.0f} mm across) lying flat at the bottom — the plate is "
            f"sunk flush and cannot be picked out. The well is COVERED by a RED shutter tile; "
            f"a BLUE shutter tile occupies the middle cell; the far corner cell is empty. "
            f"Each tile is a flat square plaque with an upright knob post "
            f"({2 * c.knob_r * 1000:.0f} mm across) in its centre. The rails overhang the "
            f"channel: the tiles can SLIDE from cell to cell but can never be lifted out. "
            f"The vault's position and heading vary per episode, and which bowl stands where "
            f"in the row also varies.\n"
            f"Goal: put the FRONT bowl of the row — the one nearest you — on the plate. The "
            f"plate is at the bottom of the well, so the well must be uncovered first: slide "
            f"the BLUE tile from the middle cell into the empty far corner cell (push or drag "
            f"it by its knob), then slide the RED tile off the well into the vacated middle "
            f"cell, then lower the FRONT bowl through the open aperture and set it upright, "
            f"centred on the plate. Only the front bowl counts; a bowl set on a closed "
            f"shutter, on the deck beside the well, a wrong (rear) bowl, or a tipped bowl "
            f"does not count. Everything must come to rest."
        )

    def instruction(self) -> str:
        return (
            "Slide the BLUE shutter tile into the empty corner cell, slide the RED shutter "
            "tile off the well into the vacated middle cell, then place the FRONT black bowl "
            "of the three down through the open aperture, upright and centred on the white "
            "plate at the bottom of the well."
        )

    # ----- geometric predicates ------------------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (...,) for tilt gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def _vault_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world position -> vault-local frame (origin: cell-A centre at counter top)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.vault.data.root_quat_w,
                                  pos_w - self.vault.data.root_pos_w)

    def _front_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,3),(N,4): world pos and quat of the episode's FRONT bowl."""
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        idx = self.front_idx
        n = idx.shape[0]
        return pos[torch.arange(n, device=idx.device), idx], \
            quat[torch.arange(n, device=idx.device), idx]

    def bowls_on_plate(self) -> torch.Tensor:
        """(N, B) bool: bowl upright, its bottom at the plate top, centred within
        `bowl_xy_tol` of the plate axis (plate frame), plate upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, nb = self.env.num_envs, c.n_bowls
        pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1)  # (N,B,3)
        quat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        pq = self.plate.data.root_quat_w[:, None, :].expand(n, nb, 4).reshape(-1, 4)
        pp = self.plate.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(pq, (pos - pp).reshape(-1, 3)).reshape(n, nb, 3)
        near = loc[:, :, :2].norm(dim=-1) < c.bowl_xy_tol
        upright = self._up_z(quat).clamp(-1, 1) >= math.cos(math.radians(c.bowl_tilt_max_deg))
        bowl_bottom = pos[:, :, 2] - c.bowl_floor_h / 2
        plate_top = (self.plate.data.root_pos_w[:, 2] + c.plate_h / 2).unsqueeze(1)
        on_top = (bowl_bottom - plate_top).abs() < c.bowl_z_tol
        plate_up = self._up_z(self.plate.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.plate_tilt_max_deg))
        return near & upright & on_top & plate_up.unsqueeze(1)

    def front_on_plate(self) -> torch.Tensor:
        """(N,) bool: the episode's FRONT bowl sits on the plate."""
        return self.bowls_on_plate().gather(1, self.front_idx.unsqueeze(1)).squeeze(1)

    def plate_in_well(self) -> torch.Tensor:
        """(N,) bool: plate upright, seated at the bottom of the well (vault frame)."""
        c = self.cfg
        loc = self._vault_local(self.plate.data.root_pos_w)
        near_xy = loc[:, :2].norm(dim=-1) < c.plate_xy_tol
        near_z = (loc[:, 2] - c.plate_seat_z).abs() < c.plate_z_tol
        upright = self._up_z(self.plate.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.plate_tilt_max_deg))
        return near_xy & near_z & upright

    def settled(self) -> torch.Tensor:
        """(N,) bool: tiles, plate and every bowl slow."""
        c = self.cfg
        still = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for t in self.tiles:
            still &= t.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        for b in self.bowls:
            still &= b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still

    def success(self) -> torch.Tensor:
        """(N,) bool: front bowl on the plate + plate seated in the well + everything
        settled. Pure state predicate — the ORDER is enforced by the geometry, not here."""
        return self.front_on_plate() & self.plate_in_well() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along the demonstrated
        solution): +0.25 BLUE parked in C, +0.25 RED parked in B, +0.20 the front bowl deep
        in the well; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.25 * self.ever_unlocked.float()
        s = s + 0.25 * self.ever_opened.float()
        s = s + 0.20 * self.ever_served.float()
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="shutter_vault", robot="null"))
