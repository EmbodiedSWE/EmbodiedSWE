"""UnderpinSwapScene — swap the support column under a raised deck WITHOUT letting the
deck drop (sim_gen task `approach_grasp_knife_i25`).

Derived from pick_place/approach_grasp_knife, but STRATEGICALLY different: the seed is
a pure approach-grasp-lift — servo the gripper to a knife on a table, close the jaw,
and lift; the checker is gripper-object distance held for a few frames, so the
manipulated object IS the judged object and picking it up IS the task. Here the judged
object — a raised grey DECK — must never be picked up, lifted, or tipped at all: it
rests on a fixed pedestal at one end and on a removable RED column under its other,
blue-banded end, and success requires UNDERPINNING: slide the BLUE replacement column
in under the banded end FIRST, then drag the loaded RED column out from under the deck
sideways by its handle (a real friction-loaded extraction with the load transferring
onto the blue column through contact), and park the red column on the green pad. The
plan is insert-before-remove with an irreversible negative invariant (a latched spoil
flag fires the moment the deck ever drops, lifts, or tips beyond tolerance — the
seed's own strategy, grasping and lifting the judged object, is exactly what spoils
it), and the code structure is a support-transfer sequence with an always-on collapse
monitor, not an approach-grasp servo. The seed's end state — the object grasped and
hoisted — is expressible here (deck lifted) and is a smoke control: latched failure
even after the deck is put back perfectly.

A YELLOW decoy column has the same handle but is visibly too short: seating IT and
removing the red column drops the deck end far past the spoil threshold — the decoy is
rejected by physics, not just by the rubric.

Judged in the FIXTURE body frame (pedestal + rails + lip is one kinematic body whose
xy and yaw are randomized, so every target must be read from the scene). success() iff
NOT spoiled AND the blue column stands seated in the slot under the deck's blue band
AND the deck still sits level in its pedestal rails with its free-end underside in the
supported height band AND the red column is clear of the deck footprint and standing
on the green pad AND deck + both columns are settled. score() is graded and latched
every physics substep: 0.10 * best blue approach to the slot + 0.25 * blue seated +
0.30 * red extraction progress COUNTED ONLY WHILE BLUE IS SEATED (the ordering
credit), 0.85 once the swap is complete (red clear, deck riding the blue column), 1.0
iff success(); a spoiled episode is capped at 0.10. Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - fixture: one KINEMATIC compound body — pedestal block (top at 58 mm) carrying two
    guide rails and an end lip that register the deck; local +x runs from the pedestal
    end toward the swappable end.
  - deck: grey slab 300 x 160 x 10 mm (0.45 kg) with a BLUE painted band on top over
    x in [50, 90] mm marking where the replacement must bear.
  - columns (red 58 mm / blue 55 mm / yellow decoy 40 mm tall): dynamic compound
    bodies — 40 x 40 mm bearing block + a ground-level foot bar carrying a 20 mm
    square handle post rising to ~112 mm, so a parallel jaw grasps the post far from
    any overhang. The red column spawns bearing the deck, handle pointing out the -y
    side; the blue column's handle points +y, so it slides in from the opposite side
    and the two never collide during the swap.
  - pad: green kinematic plate, the required parking spot for the extracted column.
Contact offsets are explicit and small (1 mm): the blue column passes under the deck
with 3 mm of clearance, and the default ~2 cm offset would weld that shut.

Per-episode randomization: fixture xy jitter + yaw (deck, red column, slot, band and
both handle directions all move with it), blue column xy + free yaw, decoy xy + yaw,
pad xy. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float,
         collide: bool = True) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(seg.GetPrim(), contact_offset)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC pedestal fixture at `prim_path`. Origin = deck-centre xy at
    table level, local +x toward the swappable deck end:
      - pedestal block whose top carries the deck's -x end;
      - two guide rails along x that register the deck's y position;
      - an end lip beyond the deck's -x edge that registers its x position."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(10.0)

    co, color = cfg.contact_offset, cfg.color
    # pedestal block: top = the deck ride height
    _box(stage, f"{prim_path}/pedestal", (cfg.ped_len, cfg.ped_w, cfg.ped_h),
         (cfg.ped_x, 0.0, cfg.ped_h / 2), color, co)
    # guide rails on the pedestal top, along x, y-registering the deck
    for tag, sgn in (("rail_yp", 1.0), ("rail_yn", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.ped_len, cfg.rail_w, cfg.rail_h),
             (cfg.ped_x, sgn * (cfg.rail_inner_y + cfg.rail_w / 2),
              cfg.ped_h + cfg.rail_h / 2), color, co)
    # end lip beyond the deck's -x edge, x-registering the deck
    _box(stage, f"{prim_path}/lip", (cfg.lip_t, cfg.lip_w, cfg.lip_h),
         (cfg.lip_x, 0.0, cfg.lip_h / 2), color, co)
    return root


def _spawn_deck(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC deck at `prim_path`: origin = slab centre; grey slab plus a
    visual-only blue band on top over the replacement bearing slot."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))

    _box(stage, f"{prim_path}/slab", (cfg.deck_len, cfg.deck_w, cfg.deck_t),
         (0.0, 0.0, 0.0), cfg.color, cfg.contact_offset)
    # blue band: visual marker only (no collision), sitting on the slab top over the slot
    _box(stage, f"{prim_path}/band", (cfg.band_len, cfg.deck_w, 0.0016),
         (cfg.band_x, 0.0, cfg.deck_t / 2 + 0.0008), cfg.band_color,
         cfg.contact_offset, collide=False)
    return root


def _spawn_column(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC support column at `prim_path`. Origin = bearing-block centre at
    GROUND level, handle along local +y: bearing block + ground-level foot bar + a
    square handle post at the bar's far end (the graspable feature). The foot bar rests
    on the ground so the support polygon spans block-to-post and the column stands."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))

    co, color = cfg.contact_offset, cfg.color
    bh = cfg.block_w / 2
    _box(stage, f"{prim_path}/block", (cfg.block_w, cfg.block_w, cfg.height),
         (0.0, 0.0, cfg.height / 2), color, co)
    _box(stage, f"{prim_path}/foot", (cfg.foot_w, cfg.foot_len, cfg.foot_h),
         (0.0, bh + cfg.foot_len / 2, cfg.foot_h / 2), color, co)
    post_y = bh + cfg.foot_len - cfg.post_w / 2
    _box(stage, f"{prim_path}/post", (cfg.post_w, cfg.post_w, cfg.post_h),
         (0.0, post_y, cfg.foot_h + cfg.post_h / 2), color, co)
    return root


def _fixture_spawner_cfg(*, ped_x: float, ped_len: float, ped_w: float, ped_h: float,
                         rail_inner_y: float, rail_w: float, rail_h: float,
                         lip_x: float, lip_t: float, lip_w: float, lip_h: float,
                         color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            ped_x: float = -0.115
            ped_len: float = 0.060
            ped_w: float = 0.160
            ped_h: float = 0.058
            rail_inner_y: float = 0.083
            rail_w: float = 0.008
            rail_h: float = 0.022
            lip_x: float = -0.157
            lip_t: float = 0.008
            lip_w: float = 0.180
            lip_h: float = 0.080
            color: tuple = (0.28, 0.28, 0.30)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["fixture"] = FixtureSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ped_x=ped_x, ped_len=ped_len, ped_w=ped_w, ped_h=ped_h,
        rail_inner_y=rail_inner_y, rail_w=rail_w, rail_h=rail_h,
        lip_x=lip_x, lip_t=lip_t, lip_w=lip_w, lip_h=lip_h,
        color=color, contact_offset=contact_offset,
    )


def _deck_spawner_cfg(*, deck_len: float, deck_w: float, deck_t: float, mass: float,
                      band_x: float, band_len: float, color: tuple, band_color: tuple,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "deck" not in _SPAWNER_CACHE:

        @configclass
        class DeckSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_deck)
            deck_len: float = 0.300
            deck_w: float = 0.160
            deck_t: float = 0.010
            mass: float = 0.45
            band_x: float = 0.070
            band_len: float = 0.040
            color: tuple = (0.55, 0.55, 0.58)
            band_color: tuple = (0.15, 0.35, 0.85)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["deck"] = DeckSpawnerCfg

    return _SPAWNER_CACHE["deck"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            linear_damping=0.05, angular_damping=0.10,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        deck_len=deck_len, deck_w=deck_w, deck_t=deck_t, mass=mass,
        band_x=band_x, band_len=band_len, color=color, band_color=band_color,
        contact_offset=contact_offset,
    )


def _column_spawner_cfg(*, key: str, height: float, block_w: float, foot_len: float,
                        foot_w: float, foot_h: float, post_w: float, post_h: float,
                        mass: float, color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if key not in _SPAWNER_CACHE:

        @configclass
        class ColumnSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_column)
            height: float = 0.058
            block_w: float = 0.040
            foot_len: float = 0.115
            foot_w: float = 0.020
            foot_h: float = 0.012
            post_w: float = 0.020
            post_h: float = 0.100
            mass: float = 0.15
            color: tuple = (0.85, 0.15, 0.12)
            contact_offset: float = 0.001

        _SPAWNER_CACHE[key] = ColumnSpawnerCfg

    return _SPAWNER_CACHE[key](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5,
            linear_damping=0.05, angular_damping=0.10,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        height=height, block_w=block_w, foot_len=foot_len, foot_w=foot_w,
        foot_h=foot_h, post_w=post_w, post_h=post_h, mass=mass,
        color=color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class UnderpinSwapSceneCfg(BaseCfg):
    """Config for `UnderpinSwapScene`. Honesty knobs asserted in `__post_init__`: the
    blue column passes under the deck with real clearance and holds the deck end inside
    the supported band; the decoy is short enough that riding on it drops the deck end
    past the spoil threshold; the handle posts fit a parallel jaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.05)  # max |lin vel| (deck + columns) when judging (m/s)
    slot_x_tol: float = tunable(0.025)  # blue block centre within this of slot_x (m)
    slot_y_tol: float = tunable(0.030)  # blue block centre within this of the deck axis (m)
    end_band: tuple = tunable((0.048, 0.062))  # supported free-end underside height band (m)
    fall_end_z: float = tunable(0.046)  # SPOIL: deck free-end underside ever below this (m)
    lift_center_z: float = tunable(0.100)  # SPOIL: deck centre ever above this (m)
    spoil_tilt_deg: float = tunable(12.0)  # SPOIL: deck ever tilts beyond this (deg)
    deck_tilt_deg: float = tunable(6.0)  # success: deck within this of level (deg)
    deck_yaw_tol_deg: float = tunable(8.0)  # success: deck yaw within this of the fixture (deg)
    deck_xy_tol: float = tunable(0.020)  # success: deck centre within this of nominal (m)
    pad_xy_tol: float = tunable(0.045)  # red block centre within this of the pad centre (m)
    upright_tol_deg: float = tunable(15.0)  # columns judged standing within this of vertical

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    asm_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the fixture at reset (m)
    asm_yaw_deg: float = tunable(45.0)  # uniform +/- fixture yaw at reset (deg)
    blue_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the blue column (m)
    blue_yaw_deg: float = tunable(180.0)  # uniform +/- blue column yaw (free)
    decoy_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the decoy (m)
    decoy_yaw_deg: float = tunable(30.0)  # uniform +/- decoy yaw (handle stays +y-ish)
    pad_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the parking pad (m)

    # --- tunable: placement ------------------------------------------------------------------
    asm_pos: tuple = tunable((-0.05, 0.02))  # fixture origin (deck centre), nominal
    blue_pos: tuple = tunable((0.14, 0.38))  # blue replacement column, nominal
    decoy_pos: tuple = tunable((0.46, 0.14))  # yellow decoy column, nominal
    pad_pos: tuple = tunable((0.44, -0.16))  # green parking pad, nominal

    # --- info: structure ---------------------------------------------------------------------
    deck_len: float = info(0.300)  # local x in [-150, 150] mm
    deck_w: float = info(0.160)
    deck_t: float = info(0.010)
    deck_mass: float = info(0.45)
    ped_x: float = info(-0.115)  # pedestal block centre (top = deck ride height)
    ped_len: float = info(0.060)
    ped_w: float = info(0.160)
    rail_inner_y: float = info(0.083)  # rail inner faces at +/- this (deck half-w 80 + 3 play)
    rail_w: float = info(0.008)
    rail_h: float = info(0.022)
    lip_x: float = info(-0.157)  # end lip centre (inner face 3 mm beyond the deck end)
    lip_t: float = info(0.008)
    lip_w: float = info(0.180)
    lip_h: float = info(0.080)
    red_x: float = info(0.120)  # red column bearing station (block spans [100, 140])
    slot_x: float = info(0.070)  # replacement bearing slot (block spans [50, 90])
    h_red: float = info(0.058)  # red column height = pedestal top = deck ride height
    h_blue: float = info(0.055)  # 3 mm shorter: slides under, deck settles 3 mm onto it
    h_decoy: float = info(0.040)  # far too short: riding on it spoils the deck
    block_w: float = info(0.040)
    foot_len: float = info(0.115)
    foot_w: float = info(0.020)
    foot_h: float = info(0.012)
    post_w: float = info(0.020)  # 20 mm square post: fits the 80 mm parallel jaw
    post_h: float = info(0.100)  # post top ~112 mm above ground
    col_mass: float = info(0.15)
    pad_size: float = info(0.120)
    pad_t: float = info(0.008)
    fixture_color: tuple = info((0.28, 0.28, 0.30))
    deck_color: tuple = info((0.55, 0.55, 0.58))
    band_color: tuple = info((0.15, 0.35, 0.85))
    red_color: tuple = info((0.85, 0.15, 0.12))
    blue_color: tuple = info((0.15, 0.35, 0.85))
    decoy_color: tuple = info((0.95, 0.85, 0.10))
    pad_color: tuple = info((0.10, 0.55, 0.20))
    # Explicit small offsets: the blue column passes under the deck with 3 mm clearance,
    # and the ~2 cm default offset would weld that shut.
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    deck_center_z: float = field(default=None, init=False)  # slab centre at rest on red
    end_lever: float = field(default=None, init=False)  # pedestal edge -> deck free end (m)
    slot_lever: float = field(default=None, init=False)  # pedestal edge -> slot centre (m)

    def __post_init__(self) -> None:
        self.deck_center_z = self.h_red + self.deck_t / 2
        ped_edge = self.ped_x + self.ped_len / 2  # pivot when the far support vanishes
        self.end_lever = self.deck_len / 2 - ped_edge
        self.slot_lever = self.slot_x - ped_edge

        # geometry registration
        assert self.rail_inner_y - self.deck_w / 2 >= 0.002, "deck must fit between the rails"
        assert -self.deck_len / 2 - (self.lip_x + self.lip_t / 2) >= 0.002, (
            "end lip must clear the deck end")
        assert self.slot_x + self.block_w / 2 + 0.005 <= self.red_x - self.block_w / 2, (
            "slot must clear the red bearing station")
        # swap clearances and support bands
        assert self.h_red - self.h_blue >= 0.003, (
            "blue must slide under the deck with real clearance")
        end_drop_blue = (self.h_red - self.h_blue) * self.end_lever / self.slot_lever
        assert self.h_red - end_drop_blue >= self.end_band[0] + 0.002, (
            "deck riding the blue column must sit inside the supported band")
        assert self.h_red - end_drop_blue >= self.fall_end_z + 0.006, (
            "deck riding the blue column must stay clear of the spoil threshold")
        end_drop_decoy = (self.h_red - self.h_decoy) * self.end_lever / self.slot_lever
        assert self.h_red - end_drop_decoy <= self.fall_end_z - 0.006, (
            "deck riding the DECOY must fall past the spoil threshold (physics rejects it)")
        # embodiment: the handle post is the grasp feature
        assert self.post_w <= 0.060, "handle post must fit the 80 mm parallel jaw"
        assert self.foot_h + self.post_h >= 0.090, "handle post must be graspable well off the ground"
        assert self.block_w / 2 + self.foot_len - self.post_w > self.deck_w / 2 + 0.030, (
            "handle post must stay outside the deck footprint while the block is seated")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("underpin_swap")
class UnderpinSwapScene(BaseScene):
    cfg: UnderpinSwapSceneCfg

    def __init__(self, cfg: UnderpinSwapSceneCfg | None = None) -> None:
        super().__init__(cfg or UnderpinSwapSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        ax, ay = c.asm_pos
        col_common = dict(block_w=c.block_w, foot_len=c.foot_len, foot_w=c.foot_w,
                          foot_h=c.foot_h, post_w=c.post_w, post_h=c.post_h,
                          mass=c.col_mass, contact_offset=c.contact_offset)
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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=_fixture_spawner_cfg(
                    ped_x=c.ped_x, ped_len=c.ped_len, ped_w=c.ped_w, ped_h=c.h_red,
                    rail_inner_y=c.rail_inner_y, rail_w=c.rail_w, rail_h=c.rail_h,
                    lip_x=c.lip_x, lip_t=c.lip_t, lip_w=c.lip_w, lip_h=c.lip_h,
                    color=c.fixture_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, 0.0)),
            ),
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=_deck_spawner_cfg(
                    deck_len=c.deck_len, deck_w=c.deck_w, deck_t=c.deck_t,
                    mass=c.deck_mass, band_x=c.slot_x, band_len=c.block_w,
                    color=c.deck_color, band_color=c.band_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(ax, ay, c.deck_center_z)),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedColumn",
                spawn=_column_spawner_cfg(key="col_red", height=c.h_red,
                                          color=c.red_color, **col_common),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(ax + c.red_x, ay, 0.0005), rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            "blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlueColumn",
                spawn=_column_spawner_cfg(key="col_blue", height=c.h_blue,
                                          color=c.blue_color, **col_common),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.blue_pos[0], c.blue_pos[1], 0.0005)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DecoyColumn",
                spawn=_column_spawner_cfg(key="col_decoy", height=c.h_decoy,
                                          color=c.decoy_color, **col_common),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.decoy_pos[0], c.decoy_pos[1], 0.0005)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_size, c.pad_size, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
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
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.fixture: RigidObject = env.iscene["fixture"]
        self.deck: RigidObject = env.iscene["deck"]
        self.red: RigidObject = env.iscene["red"]
        self.blue: RigidObject = env.iscene["blue"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.d0 = torch.full((n,), 0.40, device=dev)  # blue-spawn -> slot distance
        self.stage_latch = torch.zeros(n, device=dev)
        self.seat_latch = torch.zeros(n, device=dev)
        self.extract_latch = torch.zeros(n, device=dev)
        self.swap_latch = torch.zeros(n, device=dev)
        self.spoiled = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture with xy jitter + yaw; deck registered in its rails and
        the red column bearing its free end, both written in the SAME sampled fixture
        frame; blue column, decoy and pad independently jittered; latches zeroed and
        the blue approach baseline `d0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- fixture ---
        asm_xy = torch.tensor(c.asm_pos, device=dev).expand(m, 2).clone()
        asm_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.asm_jitter
        asm_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.asm_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = asm_xy
        st[:, 3] = torch.cos(asm_yaw / 2)
        st[:, 6] = torch.sin(asm_yaw / 2)
        st[:, 0:3] += origin
        self.fixture.write_root_state_to_sim(st, env_ids)

        # --- deck: registered on the pedestal + red column, same frame ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = asm_xy
        st[:, 2] = c.deck_center_z + 0.0005
        st[:, 3] = torch.cos(asm_yaw / 2)
        st[:, 6] = torch.sin(asm_yaw / 2)
        st[:, 0:3] += origin
        self.deck.write_root_state_to_sim(st, env_ids)

        # --- red column: bearing station, handle pointing -y (yaw = asm_yaw + pi) ---
        ca, sa = torch.cos(asm_yaw), torch.sin(asm_yaw)
        half = (asm_yaw + math.pi) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = asm_xy[:, 0] + ca * c.red_x
        st[:, 1] = asm_xy[:, 1] + sa * c.red_x
        st[:, 2] = 0.0005
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.red.write_root_state_to_sim(st, env_ids)

        # --- blue column: free spawn ---
        blue_xy = torch.tensor(c.blue_pos, device=dev).expand(m, 2).clone()
        blue_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.blue_jitter
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.blue_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = blue_xy
        st[:, 2] = 0.0005
        st[:, 3] = torch.cos(byaw / 2)
        st[:, 6] = torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.blue.write_root_state_to_sim(st, env_ids)

        # --- decoy ---
        dec_xy = torch.tensor(c.decoy_pos, device=dev).expand(m, 2).clone()
        dec_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.decoy_jitter
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.decoy_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = dec_xy
        st[:, 2] = 0.0005
        st[:, 3] = torch.cos(dyaw / 2)
        st[:, 6] = torch.sin(dyaw / 2)
        st[:, 0:3] += origin
        self.decoy.write_root_state_to_sim(st, env_ids)

        # --- pad ---
        pad_xy = torch.tensor(c.pad_pos, device=dev).expand(m, 2).clone()
        pad_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pad_xy
        st[:, 2] = c.pad_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        # --- baselines + latches ---
        slot_xy = asm_xy.clone()
        slot_xy[:, 0] += ca * c.slot_x
        slot_xy[:, 1] += sa * c.slot_x
        self.d0[env_ids] = (blue_xy - slot_xy).norm(dim=-1).clamp(min=0.05)
        self.stage_latch[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0
        self.extract_latch[env_ids] = 0.0
        self.swap_latch[env_ids] = 0.0
        self.spoiled[env_ids] = False

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fixture": self.fixture.data.root_state_w[env_ids].clone(),
            "deck": self.deck.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "stage_latch": self.stage_latch[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
            "extract_latch": self.extract_latch[env_ids].clone(),
            "swap_latch": self.swap_latch[env_ids].clone(),
            "spoiled": self.spoiled[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fixture.write_root_state_to_sim(state["fixture"], env_ids)
        self.deck.write_root_state_to_sim(state["deck"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.d0[env_ids] = state["d0"]
        self.stage_latch[env_ids] = state["stage_latch"]
        self.seat_latch[env_ids] = state["seat_latch"]
        self.extract_latch[env_ids] = state["extract_latch"]
        self.swap_latch[env_ids] = state["swap_latch"]
        self.spoiled[env_ids] = state["spoiled"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey rectangular DECK ({c.deck_len * 1000:.0f} x {c.deck_w * 1000:.0f} mm) "
            f"lies raised {c.h_red * 1000:.0f} mm above the ground, level. One of its short "
            f"ends rests on a dark fixed pedestal with guide rails; its other end rests on a "
            f"removable RED support column standing under it. A BLUE painted band across the "
            f"deck's top, a little inboard of the red column, marks the bearing slot where a "
            f"replacement support must go. Every support column has a {c.block_w * 1000:.0f} "
            f"mm square bearing block plus a low foot bar ending in a vertical square handle "
            f"post (~{(c.foot_h + c.post_h) * 1000:.0f} mm tall) that always sticks out from "
            f"under the deck — the red column's handle sticks out one side of the deck. Near "
            f"the deck stand two spare columns, each with the same handle: a BLUE one whose "
            f"block is just under deck height (the correct replacement), and a YELLOW one "
            f"that is visibly too short (a decoy). A flat GREEN pad lies on the ground "
            f"further away.\n"
            f"Goal: replace the deck's red support with the blue column WITHOUT ever letting "
            f"the deck drop, tip, or be lifted — the deck itself must not be picked up or "
            f"displaced, and once it falls or tips the task is failed for good, even if "
            f"everything is rebuilt afterwards. First slide the BLUE column in under the "
            f"deck (from the side its handle trails on) until its bearing block stands "
            f"directly beneath the blue band; only then pull the RED column out sideways "
            f"from under the deck by its handle post, letting the deck settle onto the blue "
            f"block; finally stand the red column on the GREEN pad. Success requires: deck "
            f"still level in its rails at supported height, blue block seated under the "
            f"band, red column standing on the pad clear of the deck — all at rest. Seating "
            f"the short YELLOW column instead lets the deck sag past the failure threshold; "
            f"a blue block left outside the band does not count; the red column dropped "
            f"anywhere but the pad does not count."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the blue support column under the blue-banded end of the raised deck, "
            "then pull the red column out from under the deck by its handle post and stand "
            "it on the green pad. Never let the deck drop, tip, or lift — if the deck falls "
            "the task is failed permanently."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _to_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> fixture body frame (origin = deck centre at ground)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.fixture.data.root_quat_w,
                                  p_w - self.fixture.data.root_pos_w)

    @staticmethod
    def _yaw(q: torch.Tensor) -> torch.Tensor:
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _up(self, body) -> torch.Tensor:
        """(N,3) body local +z in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def deck_end_z(self) -> torch.Tensor:
        """(N,) world z (above the env origin) of the deck FREE-END underside centre —
        the swap side, i.e. the deck-frame point (+len/2, 0, -t/2)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        off = torch.tensor([c.deck_len / 2, 0.0, -c.deck_t / 2],
                           device=self.env.device).expand(self.env.num_envs, 3)
        p = self.deck.data.root_pos_w + quat_apply(self.deck.data.root_quat_w, off)
        return p[:, 2] - self.env_origins[:, 2]

    def deck_tilt_deg(self) -> torch.Tensor:
        up = self._up(self.deck)
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))

    def blue_local(self) -> torch.Tensor:
        return self._to_local(self.blue.data.root_pos_w)

    def red_local(self) -> torch.Tensor:
        return self._to_local(self.red.data.root_pos_w)

    # ----- predicates -------------------------------------------------------------------------
    def blue_seated(self) -> torch.Tensor:
        """(N,) bool: blue bearing block standing upright on the ground with its centre
        inside the slot tolerance under the blue band."""
        c = self.cfg
        loc = self.blue_local()
        up = self._up(self.blue)
        upright = up[:, 2] >= math.cos(math.radians(c.upright_tol_deg))
        on_ground = loc[:, 2] < 0.020
        return ((loc[:, 0] - c.slot_x).abs() < c.slot_x_tol) \
            & (loc[:, 1].abs() < c.slot_y_tol) & upright & on_ground

    def red_clear(self) -> torch.Tensor:
        """(N,) bool: red bearing block outside the deck footprint (fixture frame)."""
        c = self.cfg
        loc = self.red_local()
        inside = (loc[:, 0].abs() < c.deck_len / 2 + c.block_w / 2 + 0.005) \
            & (loc[:, 1].abs() < c.deck_w / 2 + c.block_w / 2 + 0.005)
        return ~inside

    def red_on_pad(self) -> torch.Tensor:
        """(N,) bool: red column standing upright with its block centred on the pad."""
        c = self.cfg
        d = self.red.data.root_pos_w[:, :2] - self.pad.data.root_pos_w[:, :2]
        up = self._up(self.red)
        upright = up[:, 2] >= math.cos(math.radians(c.upright_tol_deg))
        z = self.red.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return (d.norm(dim=-1) < c.pad_xy_tol) & upright & (z < 0.060) & (z > 0.002)

    def deck_ok(self) -> torch.Tensor:
        """(N,) bool: deck still registered in its rails, level, free end in the
        supported band."""
        c = self.cfg
        loc = self._to_local(self.deck.data.root_pos_w)
        yaw_d = self._yaw(self.deck.data.root_quat_w) - self._yaw(self.fixture.data.root_quat_w)
        yaw_d = torch.atan2(torch.sin(yaw_d), torch.cos(yaw_d)).abs()
        ez = self.deck_end_z()
        return (loc[:, 0].abs() < c.deck_xy_tol) & (loc[:, 1].abs() < c.deck_xy_tol) \
            & (yaw_d < math.radians(c.deck_yaw_tol_deg)) \
            & (self.deck_tilt_deg() < c.deck_tilt_deg) \
            & (ez > c.end_band[0]) & (ez < c.end_band[1])

    def settled(self) -> torch.Tensor:
        v = self.cfg.settle_lin
        return (self.deck.data.root_lin_vel_w.norm(dim=-1) < v) \
            & (self.blue.data.root_lin_vel_w.norm(dim=-1) < v) \
            & (self.red.data.root_lin_vel_w.norm(dim=-1) < v)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch blue approach / seat / ordering-gated extraction / swap each substep,
        and monitor the SPOIL invariant: the deck must never drop, lift, or tip."""
        c = self.cfg
        # spoil monitor (irreversible)
        deck_cz = self.deck.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        self.spoiled = self.spoiled | (self.deck_end_z() < c.fall_end_z) \
            | (deck_cz > c.lift_center_z) \
            | (self.deck_tilt_deg() > c.spoil_tilt_deg)
        # blue approach toward the slot (normalized by the episode's own spawn distance)
        from isaaclab.utils.math import quat_apply

        slot_off = torch.tensor([c.slot_x, 0.0, 0.0],
                                device=self.env.device).expand(self.env.num_envs, 3)
        slot_w = self.fixture.data.root_pos_w + quat_apply(self.fixture.data.root_quat_w, slot_off)
        d = (self.blue.data.root_pos_w[:, :2] - slot_w[:, :2]).norm(dim=-1)
        self.stage_latch = torch.maximum(self.stage_latch, (1.0 - d / self.d0).clamp(0.0, 1.0))
        # seat + ordering-gated extraction + swap
        seat_now = self.blue_seated().float()
        self.seat_latch = torch.maximum(self.seat_latch, seat_now)
        red_off = (self.red_local()[:, 1].abs() - c.block_w / 2).clamp(min=0.0)
        extract = (red_off / 0.100).clamp(0.0, 1.0) * seat_now
        self.extract_latch = torch.maximum(self.extract_latch, extract)
        ez = self.deck_end_z()
        swap_now = ((seat_now > 0.5) & self.red_clear()
                    & (ez > c.end_band[0]) & (ez < c.end_band[1])).float()
        self.swap_latch = torch.maximum(self.swap_latch, swap_now)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: never spoiled; blue seated under the band; deck level, registered
        and riding at supported height; red clear of the deck and standing on the pad;
        everything settled."""
        return (~self.spoiled) & self.blue_seated() & self.deck_ok() \
            & self.red_clear() & self.red_on_pad() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * latched blue approach + 0.25 * blue seated +
        0.30 * ordering-gated red extraction (max 0.65), 0.85 once the swap is complete
        (red clear, deck riding the blue column), 1.0 iff success; a spoiled episode is
        capped at 0.10. Doing nothing scores ~0; the seed's grasp-and-lift strategy
        spoils the deck and caps at 0.10."""
        base = (0.10 * self.stage_latch + 0.25 * self.seat_latch
                + 0.30 * self.extract_latch).clamp(0.0, 0.65)
        s = torch.where(self.swap_latch > 0.5, torch.maximum(base, base.new_tensor(0.85)), base)
        s = torch.where(self.success(), s.new_tensor(1.0), s)
        return torch.where(self.spoiled, s.clamp(max=0.10), s)


register_env("simgen", lambda: EnvCfg(scene="underpin_swap", robot="null"))
