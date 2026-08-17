"""MilkShuntYardScene — route the milk crate through a walled shunt yard to the correct drop hole.

Derived from libero/libero_pick_milk but strategically different. The seed is one
prehensile transport: grasp the free-standing milk carton, carry it over open space,
release it into the basket. Here NOTHING is ever grasped or carried:

  - the MILK CRATE (100 mm cube, wider than the 80 mm Franka jaw) sits on an elevated
    walled DECK and can only be PUSHED, planar, along walled lanes;
  - the direct route north out of the spawn bay is blocked by an ORANGE BLOCKER crate
    parked at the junction; the blocker fits the narrow SIDING pocket (94 mm) that the
    milk crate (100 mm) does not, so the natural move is to bulldoze the blocker
    straight ahead into the siding, parking it out of the way;
  - the cross-lane ends in TWO floor apertures, one per side. Under one (the GREEN-tab
    side, randomized per episode) an enclosed under-deck cell holds the delivery
    basket; under the other (RED tab) is a bare walled trap cell. The crate must be
    pushed along the lane to the green side until it tips through the aperture and
    free-falls into the basket. The wrong hole is terminal: the trap cell is fully
    enclosed (walls to the deck, deck roof, 150 mm aperture as the only opening, and
    the crate is ungraspable), so the crate can never be recovered.

So the task is push-only routing + a binary committed choice, versus the seed's
grasp-carry-release. Success is gated by an APERTURE-TRANSIT CREDENTIAL: the crate's
CoM must be seen inside the goal aperture volume moving downward. A crate placed into
the basket by any other route never earns it.

Rubric (graded [0, 1]; latched stages + a current-state terminal conjunction):
  0.00   null policy (nothing moves; all latches start False)
  0.15   latched: blocker shunted into the siding pocket
  +0.20  latched: crate committed onto the goal half of the cross-lane
  +0.30  latched: crate fell through the goal aperture (transit credential)
  (cap 0.65)
  1.00   iff success(): credential fired AND the crate rests settled inside the
         upright basket AND the blocker is not in the basket.

Honesty by construction (asserted in __post_init__):
  - the siding admits the blocker but refuses the crate;
  - the blocker fully gates the bay mouth (no slip-past corridor);
  - the crate cannot bridge either aperture in any yaw (diagonal < both spans);
  - the basket mouth strictly contains the aperture footprint (a straight fall lands
    inside) and the crate fits the basket interior at any yaw;
  - the basket (outer) fits the cell, its rim sits well under the deck;
  - the crate exceeds the jaw span (push-only for the goal object) and both crates
    stand proud of the lane walls (top faces reachable for fingertip pushes);
  - the whole yard stays inside Franka reach at worst-case jitter.

Assets are fully procedural: a single kinematic YARD compound (deck tiles with two
aperture cutouts, lane/bay/siding walls, two enclosed under-deck cells, plinth), a
dynamic milk crate, a dynamic blocker crate, a dynamic open basket, and two
collider-less marker tabs (green/red) teleported to the dealt sides each episode.
Per-episode randomization: yard xy jitter + yaw, goal side L/R, crate and blocker
jitter + yaw. Heavy imports (isaaclab, pxr) are deferred.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float,
                max_depen: float = 0.5, live: bool = False):
    """Root Xform + rigid-body APIs, xform ops authored fresh."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(max_depen)
    if live:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)
    return stage, root


def _box_part(stage, prim_path: str, name: str, size, center, color,
              contact_offset: float | None, yaw_deg: float = 0.0) -> None:
    """One box child (optionally yawed about z); collider iff contact_offset is not None."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        h = math.radians(yaw_deg) / 2
        sxf.AddOrientOp().Set(Gf.Quatf(math.cos(h), Gf.Vec3f(0.0, 0.0, math.sin(h))))
    sxf.AddScaleOp().Set(Gf.Vec3f(*size))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)


def _spawn_yard(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC shunt yard, root at GROUND level under the deck center.
    Local frame: bay opens south (-y), siding pocket north (+y), the cross-lane runs
    east-west with one floor aperture per end. Under each aperture an enclosed cell.
    """
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=20.0)
    co = cfg.contact_offset
    zt, dt = cfg.deck_z, cfg.deck_t
    zc_deck = zt - dt / 2
    wt, wh = cfg.wall_t, cfg.wall_h
    zc_wall = zt + wh / 2
    dk, wc = cfg.deck_color, cfg.wall_color

    # --- deck tiles (the two apertures are the gaps between them) ---
    parts = [
        # center tile: everything |x| <= hole_x0
        ("deck_c", (2 * cfg.hole_x0, cfg.yard_ny + cfg.yard_py, dt),
         (0.0, (cfg.yard_py - cfg.yard_ny) / 2, zc_deck), dk),
    ]
    for s, tag in ((-1.0, "w"), (1.0, "e")):
        parts += [
            # strips beside the lane at aperture x-range
            (f"deck_n_{tag}", (cfg.hole_x1 - cfg.hole_x0, cfg.yard_py - cfg.lane_y1, dt),
             (s * (cfg.hole_x0 + cfg.hole_x1) / 2, (cfg.lane_y1 + cfg.yard_py) / 2, zc_deck), dk),
            (f"deck_s_{tag}", (cfg.hole_x1 - cfg.hole_x0, cfg.lane_y0 - (-cfg.yard_ny), dt),
             (s * (cfg.hole_x0 + cfg.hole_x1) / 2, (cfg.lane_y0 - cfg.yard_ny) / 2, zc_deck), dk),
            # end rim beyond the aperture, under the end wall
            (f"deck_r_{tag}", (cfg.lane_x1 + wt - cfg.hole_x1, cfg.yard_ny + cfg.yard_py, dt),
             (s * (cfg.hole_x1 + cfg.lane_x1 + wt) / 2, (cfg.yard_py - cfg.yard_ny) / 2,
              zc_deck), dk),
        ]

    # --- walls on the deck ---
    bx, by0, by1 = cfg.bay_x, cfg.bay_y0, cfg.bay_y1  # bay inner half-x, y-span
    sx, sy0, sy1 = cfg.siding_x, cfg.lane_y1, cfg.siding_y1
    parts += [
        ("bay_wall_w", (wt, by1 - by0 + wt, wh), (-(bx + wt / 2), (by0 + by1 - wt) / 2, zc_wall), wc),
        ("bay_wall_e", (wt, by1 - by0 + wt, wh), (bx + wt / 2, (by0 + by1 - wt) / 2, zc_wall), wc),
        ("bay_wall_s", (2 * bx + 2 * wt, wt, wh), (0.0, by0 - wt / 2, zc_wall), wc),
        ("sid_wall_w", (wt, sy1 - sy0 + wt, wh), (-(sx + wt / 2), (sy0 + sy1 + wt) / 2, zc_wall), wc),
        ("sid_wall_e", (wt, sy1 - sy0 + wt, wh), (sx + wt / 2, (sy0 + sy1 + wt) / 2, zc_wall), wc),
        ("sid_wall_n", (2 * sx + 2 * wt, wt, wh), (0.0, sy1 + wt / 2, zc_wall), wc),
    ]
    for s, tag in ((-1.0, "w"), (1.0, "e")):
        parts += [
            ("lane_s_" + tag, (cfg.lane_x1 + wt - bx, wt, wh),
             (s * (bx + cfg.lane_x1 + wt) / 2, cfg.lane_y0 - wt / 2, zc_wall), wc),
            ("lane_n_" + tag, (cfg.lane_x1 + wt - sx, wt, wh),
             (s * (sx + cfg.lane_x1 + wt) / 2, cfg.lane_y1 + wt / 2, zc_wall), wc),
            ("lane_end_" + tag, (wt, cfg.lane_y1 - cfg.lane_y0 + 2 * wt, wh),
             (s * (cfg.lane_x1 + wt / 2), (cfg.lane_y0 + cfg.lane_y1) / 2, zc_wall), wc),
        ]

    # --- under-deck cells (walls ground -> deck bottom), one per side ---
    ch = zt - dt
    cxh, cyh = cfg.cell_xh, cfg.cell_yh
    cy = cfg.hole_cy
    for s, tag in ((-1.0, "w"), (1.0, "e")):
        cx = s * cfg.hole_cx
        parts += [
            (f"cell_{tag}_xn", (wt, 2 * cyh + 2 * wt, ch), (cx - cxh - wt / 2, cy, ch / 2), wc),
            (f"cell_{tag}_xp", (wt, 2 * cyh + 2 * wt, ch), (cx + cxh + wt / 2, cy, ch / 2), wc),
            (f"cell_{tag}_yn", (2 * cxh, wt, ch), (cx, cy - cyh - wt / 2, ch / 2), wc),
            (f"cell_{tag}_yp", (2 * cxh, wt, ch), (cx, cy + cyh + wt / 2, ch / 2), wc),
        ]

    # --- plinth (visual support for the deck center) ---
    parts.append(("plinth", (0.09, 0.09, ch), (0.0, -0.05, ch / 2), (0.30, 0.30, 0.34)))

    for name, size, center, color in parts:
        _box_part(stage, prim_path, name, size, center, color, co)

    # --- 45-deg funnel chamfers at the siding mouth: square up an incoming yawed
    # blocker by contact (a bare corner would jam it) ---
    fl = 0.036  # funnel wall length
    r = math.sqrt(0.5)
    for s in (-1.0, 1.0):
        ux, uy = s * r, -r  # along the funnel wall, away from the mouth corner
        nx, ny = s * r, r  # outward normal (thickness offset)
        cxf = s * cfg.siding_x + ux * fl / 2 + nx * wt / 2
        cyf = cfg.lane_y1 + uy * fl / 2 + ny * wt / 2
        _box_part(stage, prim_path, f"funnel_{'e' if s > 0 else 'w'}",
                  (fl, wt, wh), (cxf, cyf, zc_wall), wc, co, yaw_deg=-s * 45.0)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Dynamic open basket, root at the BASE BOTTOM CENTER: floor + 4 walls."""
    stage, root = _apply_root(prim_path, translation, orientation, kinematic=False,
                              mass=cfg.mass, live=True)
    from pxr import PhysxSchema

    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.10)
    ih, wt, wh = cfg.inner_half, cfg.wall_t, cfg.wall_h
    oh = ih + wt
    co, color = cfg.contact_offset, cfg.color
    _box_part(stage, prim_path, "floor", (2 * oh, 2 * oh, cfg.bot_t),
              (0.0, 0.0, cfg.bot_t / 2), color, co)
    for s, tag in ((-1.0, "yn"), (1.0, "yp")):
        _box_part(stage, prim_path, f"wall_{tag}", (2 * oh, wt, wh),
                  (0.0, s * (ih + wt / 2), cfg.bot_t + wh / 2), color, co)
    for s, tag in ((-1.0, "xn"), (1.0, "xp")):
        _box_part(stage, prim_path, f"wall_{tag}", (wt, 2 * ih, wh),
                  (s * (ih + wt / 2), 0.0, cfg.bot_t + wh / 2), color, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "yard" not in _SPAWNER_CACHE:

        @configclass
        class YardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_yard)
            deck_z: float = 0.200
            deck_t: float = 0.012
            wall_t: float = 0.012
            wall_h: float = 0.060
            yard_ny: float = 0.272
            yard_py: float = 0.182
            bay_x: float = 0.080
            bay_y0: float = -0.260
            bay_y1: float = -0.100
            lane_x1: float = 0.345
            lane_y0: float = -0.100
            lane_y1: float = 0.060
            siding_x: float = 0.047
            siding_y1: float = 0.170
            hole_x0: float = 0.165
            hole_x1: float = 0.315
            hole_cx: float = 0.240
            hole_cy: float = -0.020
            cell_xh: float = 0.095
            cell_yh: float = 0.095
            contact_offset: float = 0.002
            deck_color: tuple = (0.52, 0.50, 0.46)
            wall_color: tuple = (0.36, 0.40, 0.46)

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            mass: float = 0.50
            inner_half: float = 0.085
            wall_t: float = 0.007
            wall_h: float = 0.085
            bot_t: float = 0.008
            contact_offset: float = 0.002
            color: tuple = (0.62, 0.45, 0.22)

        _SPAWNER_CACHE.update(yard=YardSpawnerCfg, basket=BasketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class MilkShuntYardSceneCfg(BaseCfg):
    """Config for `MilkShuntYardScene`. Interlock margins asserted in __post_init__."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    shunt_y_min: float = tunable(0.075)  # blocker CoM past this (yard y) = in the siding
    commit_x_min: float = tunable(0.090)  # crate |yard x| past this on goal side = committed
    in_basket_margin: float = tunable(0.008)  # xy margin inside the basket interior
    in_basket_zmax: float = tunable(0.080)  # CoM below this (basket frame) = contained
    basket_tilt_max_deg: float = tunable(15.0)  # basket "upright" gate
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (task-family knobs) --------------------------------------------
    yard_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the yard pose
    yard_yaw_deg: float = tunable(8.0)  # uniform +/- yard yaw
    crate_jitter: float = tunable(0.012)  # crate xy jitter inside the bay
    crate_yaw_deg: float = tunable(6.0)
    blocker_jitter: float = tunable(0.006)  # blocker xy jitter at the junction
    blocker_yaw_deg: float = tunable(3.0)

    # --- info: yard structure (yard-local frame; bay south, siding north) ----------------------
    yard_pos: tuple = info((0.42, 0.0))  # nominal yard root (world xy)
    deck_z: float = info(0.200)  # deck TOP height
    deck_t: float = info(0.012)
    wall_t: float = info(0.012)
    wall_h: float = info(0.055)  # lane walls above the deck
    bay_x: float = info(0.080)  # bay inner half-width (x)
    bay_y0: float = info(-0.260)  # bay south inner face
    bay_y1: float = info(-0.100)  # bay mouth = lane south inner face
    lane_x1: float = info(0.345)  # lane end inner face (each side)
    lane_y0: float = info(-0.100)
    lane_y1: float = info(0.060)  # lane north inner face = siding mouth line
    siding_x: float = info(0.047)  # siding inner half-width — refuses the crate
    siding_y1: float = info(0.170)  # siding north inner face
    hole_x0: float = info(0.165)  # aperture near edge (|x|)
    hole_x1: float = info(0.315)  # aperture far edge (|x|)
    hole_cy: float = info(-0.020)  # aperture/cell y center (= lane center)
    cell_xh: float = info(0.095)  # under-deck cell inner half-extents
    cell_yh: float = info(0.095)
    crate_s: float = info(0.100)  # milk crate (cube) — exceeds the jaw AND the siding
    crate_mass: float = info(0.35)
    blocker_s: float = info(0.076)  # blocker (cube) — fits the siding with yaw headroom
    blocker_mass: float = info(0.25)
    crate_spawn_y: float = info(-0.185)  # crate bay seat (yard y)
    blocker_spawn_y: float = info(-0.020)  # blocker junction seat (yard y)
    basket_inner_half: float = info(0.085)
    basket_wall_t: float = info(0.007)
    basket_wall_h: float = info(0.085)
    basket_bot_t: float = info(0.008)
    basket_mass: float = info(0.50)
    tab_s: float = info(0.060)  # marker tab plate (visual only)
    tab_y: float = info(0.115)  # tab seat on the north strip (yard y)
    contact_offset: float = info(0.002)
    deck_color: tuple = info((0.52, 0.50, 0.46))
    wall_color: tuple = info((0.36, 0.40, 0.46))
    crate_color: tuple = info((0.92, 0.94, 0.97))  # milk white
    blocker_color: tuple = info((0.90, 0.45, 0.10))  # orange
    basket_color: tuple = info((0.62, 0.45, 0.22))
    green: tuple = info((0.10, 0.65, 0.15))
    red: tuple = info((0.75, 0.10, 0.10))

    # Derived (filled in __post_init__).
    hole_cx: float = field(default=None, init=False)  # aperture x center (|x|)
    deck_bot: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.hole_cx = (self.hole_x0 + self.hole_x1) / 2
        self.deck_bot = self.deck_z - self.deck_t

        cs, bs = self.crate_s, self.blocker_s
        # siding admits the blocker, refuses the crate (worst blocker yaw included)
        b_half = bs / 2 * (math.cos(math.radians(self.blocker_yaw_deg))
                           + math.sin(math.radians(self.blocker_yaw_deg)))
        assert b_half + 0.003 < self.siding_x, "siding must admit the blocker"
        assert 2 * self.siding_x < cs - 0.004, "siding must refuse the crate"
        # blocker fully gates the bay mouth: crate (center |x| <= bay_x - cs/2) always
        # overlaps the blocker footprint in x
        assert self.bay_x - cs / 2 < bs / 2 + cs / 2 - self.blocker_jitter - 0.004, \
            "blocker must gate the bay mouth at every legal crate x"
        # crate cannot bridge either aperture span at any yaw
        diag = cs * math.sqrt(2.0)
        assert diag + 0.004 < self.hole_x1 - self.hole_x0, "no bridging (x span)"
        assert diag + 0.004 < self.lane_y1 - self.lane_y0, "no bridging (y = lane width)"
        # a straight fall lands inside the basket mouth; the crate fits at any yaw
        bi = self.basket_inner_half
        assert self.hole_cx - self.hole_x0 + 0.008 < bi, "aperture inside basket mouth (x)"
        # worst-case CoM y (hugging a lane wall) still lands within the basket footprint
        assert (self.lane_y1 - self.lane_y0) / 2 < bi + self.basket_wall_t, \
            "aperture y-span lands within the basket footprint"
        assert diag / 2 + 0.004 < bi - self.in_basket_margin, "crate fits the basket at any yaw"
        # basket fits the cell; rim well under the deck
        bo = bi + self.basket_wall_t
        assert bo + 0.002 < self.cell_xh and bo + 0.002 < self.cell_yh, "basket fits the cell"
        assert self.basket_bot_t + self.basket_wall_h + 0.03 < self.deck_bot, \
            "basket rim clears the deck bottom"
        # cell strictly contains the aperture footprint (deck rims cover the cell walls)
        assert self.hole_x1 - self.hole_x0 < 2 * self.cell_xh, "cell wider than aperture (x)"
        assert self.lane_y1 - self.lane_y0 < 2 * self.cell_yh + 0.05, "cell covers lane width"
        # embodiment: the GOAL object is push-only (exceeds the 80 mm jaw); both tops
        # stand proud of the lane walls for fingertip pushes
        assert cs > 0.082, "the milk crate must exceed the 80 mm jaw"
        assert cs > self.wall_h + 0.030, "crate top >= 30 mm proud of the walls"
        assert bs > self.wall_h + 0.020, "blocker top >= 20 mm proud of the walls"
        # reach: farthest REQUIRED contact (crate south face at spawn; crate west face
        # at goal tip-over) at worst jitter stays inside ~0.78 m of the base. The arm
        # never needs the lane end walls — the crate falls once its CoM passes hole_x0.
        r_contact = max(-self.crate_spawn_y + cs / 2 + self.crate_jitter,
                        self.hole_x0 - cs / 2 + 0.02)
        far = math.hypot(self.yard_pos[0], self.yard_pos[1]) + self.yard_jitter * math.sqrt(2) \
            + r_contact
        assert far < 0.78, "yard inside Franka reach"
        # spawn clearances off the walls (reset-band rule: >= 5 mm)
        assert self.bay_x - cs / 2 - self.crate_jitter > 0.005, "crate bay clearance"
        assert self.bay_y0 + cs / 2 + 0.005 < self.crate_spawn_y - self.crate_jitter, \
            "crate off the bay south wall"
        assert self.lane_y0 + bs / 2 + 0.005 < self.blocker_spawn_y - self.blocker_jitter, \
            "blocker off the lane south wall"
        assert self.blocker_spawn_y + self.blocker_jitter + bs / 2 + 0.005 < self.lane_y1, \
            "blocker off the lane north wall"


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("milk_shunt_yard")
class MilkShuntYardScene(BaseScene):
    cfg: MilkShuntYardSceneCfg

    def __init__(self, cfg: MilkShuntYardSceneCfg | None = None) -> None:
        super().__init__(cfg or MilkShuntYardSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        spawners = _spawner_classes()
        yard_cls, basket_cls = spawners["yard"], spawners["basket"]

        def cube(size, mass, color):
            return sim_utils.CuboidCfg(
                size=(size, size, size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.08,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.25, dynamic_friction=0.22, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        def tab(color):
            return sim_utils.CuboidCfg(
                size=(c.tab_s, c.tab_s, 0.003),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )  # NO collision_props: visual-only marker

        yx, yy = c.yard_pos
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
            "yard": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Yard",
                spawn=yard_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    deck_z=c.deck_z, deck_t=c.deck_t, wall_t=c.wall_t, wall_h=c.wall_h,
                    bay_x=c.bay_x, bay_y0=c.bay_y0, bay_y1=c.bay_y1,
                    lane_x1=c.lane_x1, lane_y0=c.lane_y0, lane_y1=c.lane_y1,
                    siding_x=c.siding_x, siding_y1=c.siding_y1,
                    hole_x0=c.hole_x0, hole_x1=c.hole_x1, hole_cx=c.hole_cx,
                    hole_cy=c.hole_cy, cell_xh=c.cell_xh, cell_yh=c.cell_yh,
                    contact_offset=c.contact_offset,
                    deck_color=c.deck_color, wall_color=c.wall_color),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(yx, yy, 0.0)),
            ),
            "crate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crate",
                spawn=cube(c.crate_s, c.crate_mass, c.crate_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(yx, yy + c.crate_spawn_y, c.deck_z + c.crate_s / 2 + 0.002)),
            ),
            "blocker": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Blocker",
                spawn=cube(c.blocker_s, c.blocker_mass, c.blocker_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(yx, yy + c.blocker_spawn_y, c.deck_z + c.blocker_s / 2 + 0.002)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_cls(
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.basket_mass, inner_half=c.basket_inner_half,
                    wall_t=c.basket_wall_t, wall_h=c.basket_wall_h,
                    bot_t=c.basket_bot_t, contact_offset=c.contact_offset,
                    color=c.basket_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(yx + c.hole_cx, yy + c.hole_cy, 0.002)),
            ),
            "green_tab": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GreenTab",
                spawn=tab(c.green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(yx + c.hole_cx, yy + c.tab_y, c.deck_z + 0.0016)),
            ),
            "red_tab": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedTab",
                spawn=tab(c.red),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(yx - c.hole_cx, yy + c.tab_y, c.deck_z + 0.0016)),
            ),
        }

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
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.yard: RigidObject = env.iscene["yard"]
        self.crate: RigidObject = env.iscene["crate"]
        self.blocker: RigidObject = env.iscene["blocker"]
        self.basket: RigidObject = env.iscene["basket"]
        self.green_tab: RigidObject = env.iscene["green_tab"]
        self.red_tab: RigidObject = env.iscene["red_tab"]
        self.env_origins = env.iscene.env_origins
        # Episode state.
        self.goal_side = torch.ones(n, device=dev)  # +1 east / -1 west
        self._shunted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._committed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._trapped = torch.zeros(n, dtype=torch.bool, device=dev)
        # External drive inputs (world frame; solve/smoke write, post_step consumes
        # + owns both bodies' external-wrench slots).
        self.crate_force = torch.zeros(n, 3, device=dev)
        self.blocker_force = torch.zeros(n, 3, device=dev)

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: yard dealt with xy jitter + yaw; goal side sampled L/R
        (uniform comparison — first-randint degeneracy); crate seated in the bay and
        blocker at the junction (jitter + yaw, yard frame); basket teleported into the
        goal cell; marker tabs dealt to their sides; latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(8, device=dev)  # burn: FIRST post-seed draws are biased
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yard_yaw_deg)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.yard_jitter
        yx = c.yard_pos[0] + jit[:, 0]
        yy = c.yard_pos[1] + jit[:, 1]
        gs = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0).to(dev)
        self.goal_side[env_ids] = gs
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write(body, lx, ly, lz, lyaw) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = yx + cy * lx - sy * ly
            st[:, 1] = yy + sy * lx + cy * ly
            st[:, 2] = lz
            w = yaw + lyaw
            st[:, 3] = torch.cos(w / 2)
            st[:, 6] = torch.sin(w / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        write(self.yard, zero, zero, zero, zero)
        cj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.crate_jitter
        write(self.crate, cj[:, 0], c.crate_spawn_y + cj[:, 1],
              torch.full((m,), c.deck_z + c.crate_s / 2 + 0.002, device=dev),
              (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.crate_yaw_deg))
        bj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.blocker_jitter
        write(self.blocker, bj[:, 0], c.blocker_spawn_y + bj[:, 1],
              torch.full((m,), c.deck_z + c.blocker_s / 2 + 0.002, device=dev),
              (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.blocker_yaw_deg))
        write(self.basket, gs * c.hole_cx, torch.full((m,), c.hole_cy, device=dev),
              torch.full((m,), 0.002, device=dev), zero)
        ztab = torch.full((m,), c.deck_z + 0.0016, device=dev)
        ty = torch.full((m,), c.tab_y, device=dev)
        write(self.green_tab, gs * c.hole_cx, ty, ztab, zero)
        write(self.red_tab, -gs * c.hole_cx, ty, ztab, zero)

        self._shunted[env_ids] = False
        self._committed[env_ids] = False
        self._delivered[env_ids] = False
        self._trapped[env_ids] = False
        self.crate_force[env_ids] = 0.0
        self.blocker_force[env_ids] = 0.0

    # ----- step-coupled mechanics (every substep) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the push forces (owned wrench slots), then latch the rubric stages
        (NaN-guarded — a diverged frame earns no progress)."""
        n, dev = self.env.num_envs, self.env.device
        zt = torch.zeros(n, 1, 3, device=dev)
        self.crate.set_external_force_and_torque(
            torch.nan_to_num(self.crate_force).reshape(n, 1, 3), zt)
        self.blocker.set_external_force_and_torque(
            torch.nan_to_num(self.blocker_force).reshape(n, 1, 3), zt)

        good = (torch.isfinite(self.crate.data.root_pos_w).all(dim=-1)
                & torch.isfinite(self.blocker.data.root_pos_w).all(dim=-1))
        self._shunted |= self.shunted_now() & good
        self._committed |= self.committed_now() & good
        self._delivered |= self.transit_now(goal=True) & good
        self._trapped |= self.transit_now(goal=False) & good

    # ----- state (full, restorable) -----------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"yard": self.yard, "crate": self.crate, "blocker": self.blocker,
                "basket": self.basket, "green_tab": self.green_tab, "red_tab": self.red_tab}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("goal_side", "_shunted", "_committed", "_delivered",
                               "_trapped", "crate_force", "blocker_force")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, st in state["bodies"].items():
            bodies[nm].write_root_state_to_sim(st, env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An elevated SHUNT YARD deck (top {c.deck_z * 100:.0f} cm up) with low walled "
            f"lanes. A white MILK CRATE ({c.crate_s * 1000:.0f} mm cube — wider than a "
            f"parallel gripper can span, push-only) sits in the south spawn bay. The bay "
            f"opens onto an east-west CROSS-LANE, but an ORANGE BLOCKER crate "
            f"({c.blocker_s * 1000:.0f} mm) is parked at the junction and "
            f"gates the mouth. Straight across the lane a narrow SIDING pocket "
            f"({2 * c.siding_x * 1000:.0f} mm wide) admits the blocker but refuses the milk "
            f"crate. Each end of the cross-lane has a floor APERTURE: under one — marked by "
            f"the GREEN tab on the deck beside it — an enclosed under-deck cell holds the "
            f"delivery basket; under the other (RED tab) is a bare walled trap cell. Which "
            f"side is green is randomized. Crate tops stand proud of the walls for "
            f"fingertip pushes.\n"
            f"Goal: shunt the blocker out of the junction (pushing it straight north parks "
            f"it in the siding), route the milk crate into the cross-lane and push it along "
            f"the GREEN side until it tips through that aperture and lands inside the "
            f"basket below. The red aperture is terminal: the trap cell is fully enclosed "
            f"and the crate cannot be grasped, so a crate dropped there is lost. Pushing "
            f"the blocker toward the green aperture risks dropping IT into the basket, "
            f"which also fails. Lowering anything into a cell by hand does not count — "
            f"only a fall through the green aperture is credited."
        )

    def instruction(self) -> str:
        return (
            "Push the orange blocker out of the junction (it fits the narrow siding "
            "straight ahead), then push the white milk crate along the cross-lane toward "
            "the GREEN tab side until it falls through that floor aperture into the basket "
            "in the cell below. Dropping the milk crate through the red aperture loses it; "
            "dropping the blocker into the basket, or placing anything into the basket by "
            "any route other than the green aperture, fails the task."
        )

    # ----- predicates / rubric ----------------------------------------------------------------------
    def _to_yard(self, p: torch.Tensor) -> torch.Tensor:
        """(N,3) world -> yard-local."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.yard.data.root_quat_w, p - self.yard.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def shunted_now(self) -> torch.Tensor:
        """(N,) bool: blocker CoM inside the siding pocket, on the deck."""
        c = self.cfg
        p = self._to_yard(self.blocker.data.root_pos_w)
        return ((p[:, 0].abs() < c.siding_x + 0.006) & (p[:, 1] > c.shunt_y_min)
                & (p[:, 1] < c.siding_y1 + 0.01)
                & (p[:, 2] > c.deck_z) & (p[:, 2] < c.deck_z + 0.10))

    def committed_now(self) -> torch.Tensor:
        """(N,) bool: crate on the deck, inside the cross-lane band, past commit_x_min
        toward the goal side."""
        c = self.cfg
        p = self._to_yard(self.crate.data.root_pos_w)
        in_lane = (p[:, 1] > c.lane_y0 - 0.005) & (p[:, 1] < c.lane_y1 + 0.005)
        on_deck = (p[:, 2] > c.deck_z) & (p[:, 2] < c.deck_z + 0.12)
        return in_lane & on_deck & (p[:, 0] * self.goal_side > c.commit_x_min)

    def transit_now(self, *, goal: bool) -> torch.Tensor:
        """(N,) bool: crate CoM inside the (goal|trap) aperture volume MOVING DOWN —
        the transit credential. Only a fall through that hole produces it."""
        c = self.cfg
        p = self._to_yard(self.crate.data.root_pos_w)
        side = self.goal_side if goal else -self.goal_side
        in_x = (p[:, 0] * side > c.hole_x0 - 0.01) & (p[:, 0] * side < c.hole_x1 + 0.01)
        in_y = (p[:, 1] > c.lane_y0 - 0.012) & (p[:, 1] < c.lane_y1 + 0.012)
        # z band hugs the APERTURE PLANE (not the whole under-deck column): a falling
        # crate's CoM sweeps it over several substeps, but a crate set down inside the
        # cell / basket below never re-enters it. Only a true through-the-hole fall
        # earns the credential.
        in_z = (p[:, 2] > c.deck_z - 0.02) & (p[:, 2] < c.deck_z + 0.02)
        down = self.crate.data.root_lin_vel_w[:, 2] < -0.05
        return in_x & in_y & in_z & down

    def in_basket(self, body) -> torch.Tensor:
        """(N,) bool: body CoM inside the basket interior (basket frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.basket.data.root_quat_w,
                                 body.data.root_pos_w - self.basket.data.root_pos_w)
        lim = c.basket_inner_half - c.in_basket_margin
        return ((loc[:, 0].abs() < lim) & (loc[:, 1].abs() < lim)
                & (loc[:, 2] > c.basket_bot_t - 0.005) & (loc[:, 2] < c.in_basket_zmax))

    def in_trap(self) -> torch.Tensor:
        """(N,) bool: crate CoM inside the trap cell (below deck, wrong side)."""
        c = self.cfg
        p = self._to_yard(self.crate.data.root_pos_w)
        return ((p[:, 0] * (-self.goal_side) > c.hole_cx - c.cell_xh)
                & (p[:, 0] * (-self.goal_side) < c.hole_cx + c.cell_xh)
                & ((p[:, 1] - c.hole_cy).abs() < c.cell_yh)
                & (p[:, 2] < c.deck_bot))

    def settled(self, body) -> torch.Tensor:
        return body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def outcome_ok(self) -> torch.Tensor:
        """(N,) bool, PURE current-state geometry (latch-free; smoke tests this):
        crate contained in the upright basket, blocker NOT in the basket, settled."""
        upright = self._up_z(self.basket) >= math.cos(math.radians(self.cfg.basket_tilt_max_deg))
        return (self.in_basket(self.crate) & ~self.in_basket(self.blocker)
                & upright & self.settled(self.crate))

    def success(self) -> torch.Tensor:
        """(N,) bool: outcome_ok AND the crate carries the goal-aperture transit
        credential (it got there by falling through the green hole — not by hand)."""
        return self.outcome_ok() & self._delivered

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched 0.15 shunted + 0.20 committed + 0.30
        delivered (cap 0.65); exactly 1.0 iff success(); ~0 for the null policy
        (all latches start False and nothing moves by itself)."""
        s = 0.15 * (self._shunted | self.shunted_now()).float() \
            + 0.20 * (self._committed | self.committed_now()).float() \
            + 0.30 * self._delivered.float()
        return torch.where(self.success(), torch.ones_like(s), s.clamp(max=0.65))


register_env("simgen", lambda: EnvCfg(scene="milk_shunt_yard", robot="null"))
