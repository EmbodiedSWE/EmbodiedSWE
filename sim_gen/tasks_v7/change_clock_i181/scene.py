"""CardClockScene — set the station card-clock by EXCHANGING its hour card.
Derived from rlbench/change_clock but the entire plan is different: the seed's
solver ROTATES an articulated crown/knob (continuous single-DOF angle regulation,
no free objects). Here there is NO rotary DOF anywhere — the displayed time is a
free rigid CARD seated in a snug display slot, and changing the clock is a
discrete select / extract / discard / insert exchange:

- A CONSOLE (one kinematic compound on the floor) carries four fixtures:
  a crimson DISPLAY STAND whose vertical slot holds the currently-displayed hour
  card; a gray 3-slot RACK holding the three spare hour cards; a blue open
  DISCARD TRAY; and a green INDICATOR pedestal whose recess holds a small tile
  showing the TARGET hour.
- Four identical-looking ivory CARDS differ only in their pip code: the hour is
  written as ROWS OF THREE dark pips near the card top (1 row = 3 o'clock,
  2 rows = 6, 3 rows = 9, 4 rows = 12), embossed on BOTH faces. The indicator
  TILE shows the target hour in the same code on its top face.
- The display slot admits exactly ONE card (two card thicknesses do not fit), so
  the execution order is geometry-forced: the wrong card must be extracted
  before the right one can be seated. A funnel mouth flares the slot entry.
- Goal: the card whose pip rows match the indicator tile seated fully in the
  display slot, pips upright (a flipped card hides its pips inside the slot and
  does not count), AND the old displayed card lying inside the blue tray.

So a solver needs a different PLAN (read a symbolic target, pick the matching
free body among distractors, one-out-one-in exchange through a snug slot,
container deposit — no angle servoing at all) and different CODE STRUCTURE
(grasp/extract/place primitives instead of a knob-turning controller).

Assets are fully procedural (compound spawners; child colliders of one body
never self-collide). Pips are visual-only cylinders (no collision).

Per-episode randomization (readback-verifiable): whole-console yaw + xy offset;
WHICH card is displayed, WHICH hour is the target (tile teleported onto the
indicator; unused tiles parked off-console), and the rack permutation.

Rubric (0..1; latched partial credit that does not evaporate):
  0.15 * extracted — the displayed card ever clear of the display stand (latched)
  0.20 * discarded — the displayed card ever settled inside the tray (latched)
  0.45 * seated    — the TARGET card ever seated in the display slot, upright,
                     pips up, still (latched)
  1.0 iff success() — target card seated NOW + old card in the tray NOW, both
  still. Non-success cap 0.80.

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

HOURS = (3, 6, 9, 12)  # the four hour cards; pip rows = hour / 3

# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             orient=None) -> None:
    """Author one box (collider unless collide is None; orient = wxyz quaternion)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())


def _add_pip(stage, path: str, *, center, radius: float, height: float, axis: str,
             color) -> None:
    """Author one VISUAL-ONLY pip cylinder (no collision)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(axis)
    r, h = float(radius), float(height) / 2
    if axis == "Y":
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -h, -r), Gf.Vec3f(r, h, r)])
    else:  # "Z"
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _root_xform(prim_path: str, translation, orientation):
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


def _dyn_body(root, mass: float, *, lin_damp: float, ang_damp: float) -> None:
    """Author dynamic rigid-body physics on a compound root (custom spawners apply
    NO cfg schemas — MassAPI + damping + solver iterations authored here)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)


def _spawn_console(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the console: ONE KINEMATIC compound. Base plate + display stand
    (slot + funnel mouth) + 3-slot rack + walled tray + indicator pedestal."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    bz = c.base_top  # base plate top

    # --- base plate ---
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.03, bz / 2),
             size=(0.72, 0.54, bz), color=c.base_color, collide=collide)

    def slot_fixture(tag: str, sx: float, sy: float, floor_z: float, mouth_z: float,
                     pil_w: float, color, funnel: bool) -> None:
        """One card slot: pillar to floor_z, y-walls and x-walls to mouth_z,
        optional 4-plate 45-degree funnel mouth. Interior x +/-hx, y +/-hy."""
        hx, hy = c.slot_hx, c.slot_hy
        wall_h = mouth_z - floor_z
        wz = (floor_z + mouth_z) / 2
        # pillar (slot floor)
        _add_box(stage, f"{prim_path}/{tag}_pillar",
                 center=(sx, sy, (bz + floor_z) / 2),
                 size=(pil_w, 0.07, floor_z - bz), color=color, collide=collide)
        # y-walls (front/back of the card's thin axis)
        for sgn, nm in ((1.0, "f"), (-1.0, "b")):
            _add_box(stage, f"{prim_path}/{tag}_ywall_{nm}",
                     center=(sx, sy + sgn * (hy + 0.011), wz),
                     size=(pil_w, 0.022, wall_h), color=color, collide=collide)
        # x-walls (slot ends)
        for sgn, nm in ((1.0, "l"), (-1.0, "r")):
            _add_box(stage, f"{prim_path}/{tag}_xwall_{nm}",
                     center=(sx + sgn * (hx + 0.013), sy, wz),
                     size=(0.026, 2 * hy, wall_h), color=color, collide=collide)
        if funnel:
            ln, t = c.funnel_len, 0.006
            off_c = 0.7071 * (ln + t) / 2  # plate-centre offset along the flare
            off_z = 0.7071 * (ln - t) / 2
            h22 = math.cos(math.pi / 8), math.sin(math.pi / 8)  # cos/sin 22.5 deg
            for sgn, nm in ((1.0, "f"), (-1.0, "b")):
                q = (h22[0], sgn * h22[1], 0.0, 0.0)  # rot about x by +/-45 deg
                _add_box(stage, f"{prim_path}/{tag}_funnel_y{nm}",
                         center=(sx, sy + sgn * (hy + off_c), mouth_z + off_z),
                         size=(pil_w, ln, t), color=color, collide=collide,
                         orient=q)
            for sgn, nm in ((1.0, "l"), (-1.0, "r")):
                q = (h22[0], 0.0, -sgn * h22[1], 0.0)  # rot about y
                _add_box(stage, f"{prim_path}/{tag}_funnel_x{nm}",
                         center=(sx + sgn * (hx + off_c), sy, mouth_z + off_z),
                         size=(ln, 2 * hy, t), color=color, collide=collide,
                         orient=q)

    # --- display stand (crimson) with funnel mouth ---
    slot_fixture("stand", c.stand_x, c.stand_y, c.disp_floor, c.disp_mouth,
                 0.13, c.stand_color, funnel=True)

    # --- rack (gray), 3 slots ---
    for j, rx in enumerate(c.rack_xs):
        slot_fixture(f"rack{j}", rx, c.rack_y, c.rack_floor, c.rack_mouth,
                     0.09, c.rack_color, funnel=False)

    # --- discard tray (blue): floor + 4 walls ---
    tx, ty = c.tray_x, c.tray_y
    ihx, ihy, wt, wh = c.tray_ihx, c.tray_ihy, 0.010, c.tray_wall_h
    _add_box(stage, f"{prim_path}/tray_floor",
             center=(tx, ty, bz + 0.004),
             size=(2 * ihx + 2 * wt, 2 * ihy + 2 * wt, 0.008),
             color=c.tray_color, collide=collide)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/tray_xwall_{nm}",
                 center=(tx + sgn * (ihx + wt / 2), ty, bz + 0.008 + wh / 2),
                 size=(wt, 2 * ihy + 2 * wt, wh), color=c.tray_color,
                 collide=collide)
        _add_box(stage, f"{prim_path}/tray_ywall_{nm}",
                 center=(tx, ty + sgn * (ihy + wt / 2), bz + 0.008 + wh / 2),
                 size=(2 * ihx, wt, wh), color=c.tray_color, collide=collide)

    # --- indicator pedestal (green) with a lipped recess for the target tile ---
    ex, ey = c.easel_x, c.easel_y
    ped_h = c.easel_top - bz
    _add_box(stage, f"{prim_path}/easel_ped", center=(ex, ey, bz + ped_h / 2),
             size=(0.07, 0.07, ped_h), color=c.easel_color, collide=collide)
    for sgn, nm in ((1.0, "l"), (-1.0, "r")):
        _add_box(stage, f"{prim_path}/easel_xlip_{nm}",
                 center=(ex + sgn * 0.031, ey, c.easel_top + 0.006),
                 size=(0.008, 0.07, 0.012), color=c.easel_color, collide=collide)
        _add_box(stage, f"{prim_path}/easel_ylip_{nm}",
                 center=(ex, ey + sgn * 0.031, c.easel_top + 0.006),
                 size=(0.054, 0.008, 0.012), color=c.easel_color, collide=collide)
    return root


def _pip_rows(n_rows: int) -> list[int]:
    return list(range(int(n_rows)))


def _spawn_card(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one hour card: DYNAMIC ivory box, pip rows embossed on BOTH faces
    (visual-only cylinders). Local +z = card top (pips live in the top half)."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dyn_body(root, cfg.mass_props.mass, lin_damp=0.05, ang_damp=0.10)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(c.width, c.thick, c.height), color=c.color, collide=collide)
    cols = (-c.pip_dx, 0.0, c.pip_dx)
    for r in _pip_rows(c.n_rows):
        z = c.pip_z_top - c.pip_dz * r
        for k, x in enumerate(cols):
            for sgn, face in ((1.0, "f"), (-1.0, "b")):
                _add_pip(stage, f"{prim_path}/pip_{face}_{r}_{k}",
                         center=(x, sgn * (c.thick / 2 + 0.00075), z),
                         radius=c.pip_r, height=0.0015, axis="Y",
                         color=c.pip_color)
    return root


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one indicator tile: DYNAMIC plate, pip rows on the TOP face."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dyn_body(root, cfg.mass_props.mass, lin_damp=0.10, ang_damp=0.20)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0),
             size=(c.side, c.side, c.thick), color=c.color, collide=collide)
    cols = (-c.pip_dx, 0.0, c.pip_dx)
    for r in _pip_rows(c.n_rows):
        y = (c.n_rows - 1) * c.pip_dy / 2 - c.pip_dy * r
        for k, x in enumerate(cols):
            _add_pip(stage, f"{prim_path}/pip_{r}_{k}",
                     center=(x, y, c.thick / 2 + 0.00075),
                     radius=c.pip_r, height=0.0015, axis="Z", color=c.pip_color)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "console" not in _SPAWNER_CACHE:

        @configclass
        class ConsoleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_console)
            base_top: float = 0.008
            slot_hx: float = 0.036
            slot_hy: float = 0.008
            funnel_len: float = 0.020
            stand_x: float = 0.0
            stand_y: float = 0.12
            disp_floor: float = 0.060
            disp_mouth: float = 0.110
            rack_xs: tuple = (-0.11, 0.0, 0.11)
            rack_y: float = -0.12
            rack_floor: float = 0.038
            rack_mouth: float = 0.078
            tray_x: float = -0.26
            tray_y: float = 0.12
            tray_ihx: float = 0.08
            tray_ihy: float = 0.06
            tray_wall_h: float = 0.050
            easel_x: float = 0.26
            easel_y: float = 0.12
            easel_top: float = 0.058
            base_color: tuple = (0.45, 0.45, 0.48)
            stand_color: tuple = (0.55, 0.10, 0.10)
            rack_color: tuple = (0.30, 0.32, 0.36)
            tray_color: tuple = (0.10, 0.25, 0.60)
            easel_color: tuple = (0.10, 0.45, 0.20)
            contact_offset: float = 0.002

        @configclass
        class CardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_card)
            width: float = 0.060
            thick: float = 0.010
            height: float = 0.100
            n_rows: int = 1
            pip_r: float = 0.004
            pip_dx: float = 0.013
            pip_dz: float = 0.011
            pip_z_top: float = 0.043
            color: tuple = (0.92, 0.90, 0.84)
            pip_color: tuple = (0.08, 0.08, 0.10)
            contact_offset: float = 0.002

        @configclass
        class TileSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tile)
            side: float = 0.040
            thick: float = 0.006
            n_rows: int = 1
            pip_r: float = 0.003
            pip_dx: float = 0.009
            pip_dy: float = 0.009
            color: tuple = (0.92, 0.90, 0.84)
            pip_color: tuple = (0.08, 0.08, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(console=ConsoleSpawnerCfg, card=CardSpawnerCfg,
                              tile=TileSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CardClockSceneCfg(BaseCfg):
    """Config for `CardClockScene`. The interlocks are metric: the display slot's
    16 mm gap admits ONE 10 mm card (two do not fit, asserted), so extraction
    must precede insertion; a flipped card hides its pips inside the slot and is
    rejected by the pips-up gate; all four cards are ivory boxes distinguishable
    ONLY by their pip rows."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(1.0)  # max |ang vel| when judging (rad/s)
    seat_x_tol: float = tunable(0.020)  # seated card |x - stand_x| (slot caps at 6 mm)
    seat_y_tol: float = tunable(0.012)  # seated card |y - stand_y| (slot caps at 3 mm)
    seat_z_tol: float = tunable(0.008)  # seated card centre |z - seat_z|
    upright_max_deg: float = tunable(15.0)  # card +z within this of world up (pips-up)
    tray_margin: float = tunable(0.018)  # xy inset from the tray walls that counts

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_max_deg: float = tunable(25.0)  # whole-console yaw (+/- deg)
    xy_jitter: float = tunable(0.05)  # whole-console xy offset (+/- m)

    # --- info: console layout (console frame; base plate top base_z) -----------------------------
    base_z: float = info(0.008)
    stand_x: float = info(0.0)
    stand_y: float = info(0.12)
    disp_floor: float = info(0.060)  # display slot floor top
    disp_mouth: float = info(0.110)  # display slot mouth (wall top)
    funnel_len: float = info(0.020)  # 45-deg funnel plates above the mouth
    slot_hx: float = info(0.036)  # slot interior half width (72 mm for the 60 mm card)
    slot_hy: float = info(0.008)  # slot interior half gap (16 mm for the 10 mm card)
    rack_xs: tuple = info((-0.11, 0.0, 0.11))
    rack_y: float = info(-0.12)
    rack_floor: float = info(0.038)
    rack_mouth: float = info(0.078)
    tray_x: float = info(-0.26)
    tray_y: float = info(0.12)
    tray_ihx: float = info(0.08)  # tray interior half extents
    tray_ihy: float = info(0.06)
    tray_wall_h: float = info(0.050)  # rim z = base_z + 0.008 + wall_h
    easel_x: float = info(0.26)
    easel_y: float = info(0.12)
    easel_top: float = info(0.058)  # recess floor for the indicator tile

    # --- info: cards / tiles ---------------------------------------------------------------------
    card_w: float = info(0.060)
    card_t: float = info(0.010)
    card_h: float = info(0.100)
    card_mass: float = info(0.020)
    tile_side: float = info(0.040)
    tile_t: float = info(0.006)
    tile_mass: float = info(0.008)
    hours: tuple = info(HOURS)  # pip rows = hour / 3 (1..4 rows of three)
    tile_park: tuple = info(((0.85, -0.25), (0.85, -0.05), (0.85, 0.15),
                             (0.85, 0.35)))  # off-console ground parking (local)

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.20 + 0.45 = 0.80 = the non-success cap)
    w_extract: float = info(0.15)
    w_tray: float = info(0.20)
    w_seat: float = info(0.45)

    def __post_init__(self) -> None:
        assert 2 * self.card_t > 2 * self.slot_hy, \
            "two cards must NOT fit the display slot together (order interlock)"
        assert self.card_t + 0.004 < 2 * self.slot_hy, \
            "one card must fit the display slot with clearance"
        assert len(set(self.hours)) == len(self.hours), "hours must be distinct"
        # seated card CENTRE heights (card resting on the slot floor)
        self.disp_seat_z: float = self.disp_floor + self.card_h / 2  # 0.110
        self.rack_seat_z: float = self.rack_floor + self.card_h / 2  # 0.088
        self.tray_rim_z: float = self.base_z + 0.008 + self.tray_wall_h


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("card_clock")
class CardClockScene(BaseScene):
    cfg: CardClockSceneCfg

    def __init__(self, cfg: CardClockSceneCfg | None = None) -> None:
        super().__init__(cfg or CardClockSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        console_spawn = spawners["console"](
            mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            base_top=c.base_z, slot_hx=c.slot_hx, slot_hy=c.slot_hy,
            funnel_len=c.funnel_len, stand_x=c.stand_x, stand_y=c.stand_y,
            disp_floor=c.disp_floor, disp_mouth=c.disp_mouth,
            rack_xs=c.rack_xs, rack_y=c.rack_y, rack_floor=c.rack_floor,
            rack_mouth=c.rack_mouth, tray_x=c.tray_x, tray_y=c.tray_y,
            tray_ihx=c.tray_ihx, tray_ihy=c.tray_ihy, tray_wall_h=c.tray_wall_h,
            easel_x=c.easel_x, easel_y=c.easel_y, easel_top=c.easel_top,
            contact_offset=c.contact_offset,
        )
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
            "console": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Console",
                spawn=console_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
        }
        for i, hour in enumerate(c.hours):
            out[f"card_{hour}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Card" + str(hour),
                spawn=spawners["card"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.card_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    width=c.card_w, thick=c.card_t, height=c.card_h,
                    n_rows=hour // 3, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_xs[i - 1] if i > 0 else c.stand_x,
                         c.rack_y if i > 0 else c.stand_y,
                         c.rack_seat_z if i > 0 else c.disp_seat_z)),
            )
            out[f"tile_{hour}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile" + str(hour),
                spawn=spawners["tile"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    side=c.tile_side, thick=c.tile_t, n_rows=hour // 3,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tile_park[i][0], c.tile_park[i][1], c.tile_t / 2 + 0.002)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.console: RigidObject = env.iscene["console"]
        self.cards: list[RigidObject] = [env.iscene[f"card_{h}"] for h in c.hours]
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{h}"] for h in c.hours]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.displayed_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.target_idx = torch.ones(n, dtype=torch.long, device=dev)
        # latches: partial progress survives transient achievements
        self._extracted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tray_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: console re-posed (yaw + xy), a random card seated in the
        display, the other three permuted into the rack, a random OTHER hour's
        tile put on the indicator (rest parked off-console), latches cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- console (kinematic): yaw + xy offset ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = (torch.rand(m, 2, device=dev) * 2 - 1) * c.xy_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.console.write_root_state_to_sim(st, env_ids)
        s_pos, s_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- identities: displayed card, rack permutation, target hour ---
        # (torch.rand-based draws — first-randint-after-seed is degenerate)
        perm = torch.rand(m, 4, device=dev).argsort(dim=1)  # (m, 4)
        displayed = perm[:, 0]
        t_sel = (torch.rand(m, device=dev) * 3).long().clamp(max=2)
        target = perm.gather(1, (t_sel + 1).unsqueeze(1)).squeeze(1)
        self.displayed_idx[env_ids] = displayed
        self.target_idx[env_ids] = target

        # --- cards: displayed -> display slot; perm[:,1+j] -> rack slot j ---
        rack_xs = torch.tensor(c.rack_xs, device=dev)
        for i in range(4):
            loc = torch.zeros(m, 3, device=dev)
            is_disp = displayed == i
            # rack slot of card i (position of i within perm[:,1:4])
            slot_j = (perm[:, 1:4] == i).float().argmax(dim=1)
            loc[:, 0] = torch.where(is_disp,
                                    torch.full((m,), c.stand_x, device=dev),
                                    rack_xs[slot_j])
            loc[:, 1] = torch.where(is_disp,
                                    torch.full((m,), c.stand_y, device=dev),
                                    torch.full((m,), c.rack_y, device=dev))
            loc[:, 2] = torch.where(is_disp,
                                    torch.full((m,), c.disp_seat_z, device=dev),
                                    torch.full((m,), c.rack_seat_z, device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
            st[:, 3:7] = s_quat
            self.cards[i].write_root_state_to_sim(st, env_ids)

        # --- tiles: target hour's tile on the indicator; others parked ---
        for i in range(4):
            on_easel = target == i
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = torch.where(on_easel,
                                    torch.full((m,), c.easel_x, device=dev),
                                    torch.full((m,), c.tile_park[i][0], device=dev))
            loc[:, 1] = torch.where(on_easel,
                                    torch.full((m,), c.easel_y, device=dev),
                                    torch.full((m,), c.tile_park[i][1], device=dev))
            loc[:, 2] = torch.where(on_easel,
                                    torch.full((m,), c.easel_top + c.tile_t / 2
                                               + 0.001, device=dev),
                                    torch.full((m,), c.tile_t / 2 + 0.002,
                                               device=dev))
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = s_pos + quat_apply(s_quat, loc)
            st[:, 3:7] = s_quat
            self.tiles[i].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._extracted_ever[env_ids] = False
        self._tray_ever[env_ids] = False
        self._seated_ever[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "console": self.console.data.root_state_w[env_ids].clone(),
            "cards": [b.data.root_state_w[env_ids].clone() for b in self.cards],
            "tiles": [b.data.root_state_w[env_ids].clone() for b in self.tiles],
            "displayed_idx": self.displayed_idx[env_ids].clone(),
            "target_idx": self.target_idx[env_ids].clone(),
            "extracted_ever": self._extracted_ever[env_ids].clone(),
            "tray_ever": self._tray_ever[env_ids].clone(),
            "seated_ever": self._seated_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.console.write_root_state_to_sim(state["console"], env_ids)
        for b, s in zip(self.cards, state["cards"]):
            b.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.tiles, state["tiles"]):
            b.write_root_state_to_sim(s, env_ids)
        self.displayed_idx[env_ids] = state["displayed_idx"]
        self.target_idx[env_ids] = state["target_idx"]
        self._extracted_ever[env_ids] = state["extracted_ever"]
        self._tray_ever[env_ids] = state["tray_ever"]
        self._seated_ever[env_ids] = state["seated_ever"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A station CARD CLOCK console stands on the floor: at its back-centre "
            "a CRIMSON display stand holds the currently-displayed hour card in a "
            "vertical slot (the card sticks up out of the slot; a flared funnel "
            "rims the slot mouth). At the front a GRAY three-slot rack holds the "
            "three spare hour cards, also sticking up. Back-left sits a BLUE open "
            "discard tray; back-right a GREEN indicator pedestal whose recess "
            "holds a small tile. All four cards are identical ivory plates "
            f"({c.card_w * 1000:.0f} x {c.card_h * 1000:.0f} x "
            f"{c.card_t * 1000:.0f} mm) EXCEPT for their pip code: near the top, "
            "rows of three dark pips (on both faces) encode the hour — one row = "
            "3 o'clock, two rows = 6, three rows = 9, four rows = 12. The "
            "indicator tile shows the TARGET hour in the same code (rows of "
            "three pips on its top face).\n"
            "Goal: make the clock display the target hour. Count the pip rows on "
            "the green indicator's tile, then: (1) lift the wrong card out of "
            "the crimson display slot and drop it INSIDE the blue tray; (2) take "
            "the card with the MATCHING number of pip rows from the gray rack "
            "and seat it fully down in the crimson display slot, pips at the "
            "top (a card inserted upside-down hides its pips inside the slot "
            "and does not count). The slot only admits one card at a time, so "
            "the old card must come out before the new one can go in. Success = "
            "the matching card seated upright in the display slot AND the old "
            "displayed card lying inside the blue tray, everything at rest. A "
            "wrong-hour card in the display, a card left leaning or lying on "
            "the stand instead of seated, or the old card anywhere but inside "
            "the tray — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Set the card clock to the hour shown on the green indicator tile: "
            "lift the current card out of the crimson display slot and drop it "
            "into the blue tray, then take the spare card whose pip rows match "
            "the tile from the gray rack and seat it fully in the display slot "
            "with its pips upright."
        )

    # ----- readings / rubric ---------------------------------------------------------------------
    def _card_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(loc (N,4,3) console-frame positions, up_z (N,4) world-up component of
        each card's local +z, still (N,4))."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        pos = torch.stack([b.data.root_pos_w for b in self.cards], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.cards], dim=1)
        rel = (pos - self.console.data.root_pos_w[:, None, :]).reshape(n * 4, 3)
        cq = self.console.data.root_quat_w[:, None, :].expand(n, 4, 4).reshape(n * 4, 4)
        loc = quat_apply_inverse(cq, rel).reshape(n, 4, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * 4, 3)
        up_z = quat_apply(quat.reshape(n * 4, 4), ez).reshape(n, 4, 3)[:, :, 2]
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.cards],
                          dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.cards],
                          dim=1)
        still = (lin < c.settle_speed) & (ang < c.settle_omega)
        return loc, up_z, still

    def _seated_display(self) -> torch.Tensor:
        """(N,4) bool: card seated in the DISPLAY slot — centre on the slot within
        tolerance, at seat height, upright with pips up (local +z up), still."""
        c = self.cfg
        loc, up_z, still = self._card_tensors()
        return ((loc[:, :, 0] - c.stand_x).abs() < c.seat_x_tol) \
            & ((loc[:, :, 1] - c.stand_y).abs() < c.seat_y_tol) \
            & ((loc[:, :, 2] - c.disp_seat_z).abs() < c.seat_z_tol) \
            & (up_z > math.cos(math.radians(c.upright_max_deg))) \
            & still

    def _in_display_zone(self) -> torch.Tensor:
        """(N,4) bool: card anywhere in/over the display stand (the extraction
        latch fires when the OLD card leaves this zone)."""
        c = self.cfg
        loc, _up, _still = self._card_tensors()
        return ((loc[:, :, 0] - c.stand_x).abs() < 0.055) \
            & ((loc[:, :, 1] - c.stand_y).abs() < 0.035) \
            & (loc[:, :, 2] < 0.18)

    def _in_tray(self) -> torch.Tensor:
        """(N,4) bool: card inside the tray volume — xy inside the walls minus
        margin, centre BELOW the rim minus margin (containment below the
        aperture), still."""
        c = self.cfg
        loc, _up, still = self._card_tensors()
        return ((loc[:, :, 0] - c.tray_x).abs() < c.tray_ihx - c.tray_margin) \
            & ((loc[:, :, 1] - c.tray_y).abs() < c.tray_ihy - c.tray_margin) \
            & (loc[:, :, 2] > c.base_z + 0.004) \
            & (loc[:, :, 2] < c.tray_rim_z - 0.011) \
            & still

    def _gather(self, mat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return mat.gather(1, idx.unsqueeze(1)).squeeze(1)

    def _update_latches(self) -> None:
        self._extracted_ever |= ~self._gather(self._in_display_zone(),
                                              self.displayed_idx)
        self._tray_ever |= self._gather(self._in_tray(), self.displayed_idx)
        self._seated_ever |= self._gather(self._seated_display(), self.target_idx)

    # ----- step-coupled bookkeeping --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the console is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the TARGET card seated in the display slot NOW (upright,
        pips up, still) AND the old displayed card inside the tray NOW (still).
        Physical outcome only — settled poses, real containment."""
        self._update_latches()
        return self._gather(self._seated_display(), self.target_idx) \
            & self._gather(self._in_tray(), self.displayed_idx)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*extracted + 0.20*discarded + 0.45*seated —
        all latched, ~0 for doing nothing, non-success cap 0.80 — and exactly
        1.0 iff success() holds."""
        c = self.cfg
        self._update_latches()
        base = (c.w_extract * self._extracted_ever.float()
                + c.w_tray * self._tray_ever.float()
                + c.w_seat * self._seated_ever.float()).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="card_clock", robot="null"))
