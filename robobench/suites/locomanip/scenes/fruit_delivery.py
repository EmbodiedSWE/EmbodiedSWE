"""FruitDeliveryScene — fruits_on_plate stretched across a table no arm can span.

Registered as `fruit_delivery`; runnable presets are `locomanip.fruit_delivery` (NullRobot
oracle) and `locomanip.fruit_delivery.g1.{loco_joint,loco_pink_ik}` (the mobile G1).

THE TASK. A wooden kitchen table (authored after the one in RoboLab's `fruits_out_of_basket`
scene: warm oak top at 0.70 m on thin steel legs) carries the RoboLab produce tableau on the
robot's side: fruit to pick, mixed among produce and kitchenware that must stay put. The large
round clay plate stands ON THE SAME TABLE but across it, inset from the FAR edge. **Goal: pick
up every fruit and carry it around the table to the plate — the plate is out of reach from the
pick side, so the robot must walk.**

WHY THIS GEOMETRY MAKES IT LOCO-MANIPULATION, in one number each way: the table top is 1.00 m
deep and the plate's counted centre sits 0.20 m in from the far edge, so from the closest
pick-side stance the plate is >= 0.93 m from the shoulder — more than twice the G1's 0.44 m
fingertip ceiling, and unreachable BY CONSTRUCTION for any fixed stance on the pick side (the
robot cannot climb the table; `_assert_reach_separation` enforces the margin at build). The
transit around either table end is ~2.5 m of walking per delivery, done while carrying fruit
in a swaying hand.

The identification half is INHERITED VERBATIM from `packing.fruits_on_plate` (this class
subclasses it): fruit vs not-fruit, with produce distractors — pumpkins and a red onion — that
FAIL the task if they reach the plate, the plate-body-frame footprint+settle rubric, the same
score() (per-fruit progress to 90, -10 per misplaced non-fruit) and success() (every present
fruit placed, nothing else on the plate). What changes is the table, the layout, and that the
distance is now the hard part: pick-side geometry reproduces the fixed-base tier's measured
grasp cell, so the NEW skill this task grades on top of it is the carry.

LAYOUT (table-relative; the table centre is the origin, top 1.40 x 1.00 m at 0.70 m):
  * PICK ZONE (robot's side, y < 0): the fruit at the calibrated stance spot, the produce
    distractors around it, the long utensils along the near edge.
  * CENTRE BAND: serving bowl and storage box — the colourful mid-table clutter of the
    original scene, and a physical wall against reaching across.
  * FAR SIDE (y > 0): the clay plate, centred at `plate_pos`, inset from the far edge so a
    placing stance on the far side reproduces the fixed-base tier's plate reach.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from robobench.core.assets import asset_path
from typing import ClassVar

from robobench.core import SCENES

from robobench.suites.packing.scenes.fruits_on_plate import (
    FruitsOnPlateScene,
    FruitsOnPlateSceneCfg,
)


@dataclass
class FruitDeliverySceneCfg(FruitsOnPlateSceneCfg):
    """`FruitsOnPlateSceneCfg` re-pointed at the kitchen table with a cross-table layout.

    The G1-tier manifest keeps the fruits this hand measurably grasps (the fixed-base tier's
    findings carry over: the lemon is dependable; near-spheres roll out of the three-finger
    pinch) and ALL SEVEN non-fruit distractors, because the crowd is what makes the scene read
    like the original — and what makes the identification non-trivial.
    """

    # --- the walk (difficulty dial): shoulder-to-plate from the closest pick-side stance ---
    #: Minimum allowed distance (m) from the closest pick-side shoulder position to the
    #: plate's counted centre. Asserted at build. 0.85 is ~2x the G1's fingertip ceiling.
    min_cross_reach: float = 0.85

    # --- overrides of the parent's geometry --------------------------------------------
    table: str = "kitchen"
    surface_z: float | None = 0.70
    #: plate on the FAR side (+y), its counted centre 0.30 m past the table's midline; its
    #: rim stops ~0.05 m short of the far edge (top half-depth 0.50), and the cross-table
    #: separation from the closest pick-side shoulder is 0.93 m (asserted below). The inset
    #: is chosen so a FAR-side stance with the pelvis 0.13 m off the far edge (torso clear)
    #: sees the dish centre ~0.38 m from its shoulder, comfortably inside the arm's
    #: measured 0.20-0.46 m release annulus; at 0.24 the same stance was 0.44 m out.
    plate_pos: tuple = (0.30, 0.30)
    #: pick zone on the NEAR side: the fruit line sits 0.28 m ahead of a shoulder whose
    #: pelvis stands clear of the near edge — the fixed-base tier's calibrated grasp
    #: distance, reproduced at the walking stance.
    scatter_center: tuple = (0.0, -0.37)
    scatter_span: tuple = (0.44, 0.10)
    scatter_cols: int = 3
    #: long utensils hug the near corners (309/332 mm long, lying along table x), keeping
    #: the centre of the near edge clear for the grasp corridor; bowl and box mid-table;
    #: the produce distractors between the fruit line and the midline.
    slot_override: tuple = (
        ("wooden_spoons", -0.44, -0.42, 90.0),
        ("spatula", 0.42, -0.43, 0.0),
        ("serving_bowl", -0.42, 0.05, 0.0),
        # clear of the plate's counted footprint (plate spans x 0.15..0.46, y 0.09..0.40):
        # at (0.44, 0.08) the box settled onto the plate rim, which both reads wrong and
        # scores as a misplaced non-fruit before the robot has done anything.
        ("storage_box", 0.56, -0.08, 20.0),
        ("pumpkinlarge", -0.18, -0.12, 0.0),
        ("redonion", 0.15, -0.14, 0.0),
        ("pumpkinsmall", 0.26, -0.33, 0.0),
    )
    #: G1 tier: ONE fruit — the dependable lemon — and the full distractor set. The task
    #: is one complete delivery: pick, carry around the table, place from the far side.
    #: Near-spherical fruit stays excluded — measured to roll out of the pinch — and the
    #: small lemon (40 mm, under the hand's ~50 mm minimum) with it.
    exclude: tuple = ("lemon_02", "lime01", "lime01_01", "orange_01", "orange_02",
                      "pomegranate01")
    #: (instance -> spawn-scale multiplier) applied over the parent MANIFEST's scale. Empty
    #: for the one-lemon tier; kept as the hook for re-admitting the small lemon (x1.3 puts
    #: its 40 mm short axis at ~52 mm, inside the hand's measured 50-64 mm window).
    rescale: ClassVar[dict[str, float]] = {}
    #: brighter than the parent's 2500: the walk-around cell is ~4 m across and the warm
    #: tableau should read like the original's daylight kitchen, not a dim workshop.
    surface_light: float = 3400.0
    #: Reset regime of the fixed-base G1 tier (`_fruits_on_plate_g1_cfg`), whose measured grasp
    #: this task inherits: the fruit's long axis along +y (narrow axis at the hand's azimuth
    #: 180, dead centre of its reachable band) with +/-30 deg of yaw and +/-15 mm of xy
    #: jitter, spawned essentially at rest so the commanded orientation survives settling.
    #: One fruit in the pool, so no subset sampling and nothing to shuffle.
    subset_sample: bool = False
    reset_pos_jitter: float = 0.015
    reset_yaw_center_deg: float = 90.0
    reset_yaw_deg: float = 60.0
    shuffle_slots: bool = False
    drop_lift: float = 0.002

    def __post_init__(self) -> None:
        # The kitchen table is suite-local (authored by scripts/author_fruit_delivery_table.py).
        table_usd = (asset_path(Path(__file__).resolve().parents[1] / "assets") / "fruit_delivery"
                     / "kitchen_table" / "main.usda")
        self.TABLES = dict(self.TABLES)
        self.TABLES["kitchen"] = {
            "usd": None, "scale": 1.0, "orient": (1.0, 0.0, 0.0, 0.0),
            "surface_z": 0.70, "pos": (0.0, 0.0),
            # top_offset == height == the authored surface height: the ground-standing
            # branch in the parent's assets() then spawns the table at z=0 on a ground at
            # z=0 with a z-scale of exactly 1 — no sinking, no squashing, one shared floor
            # for the table and the walking robot. (top_offset=0 was tried first and takes
            # the OTHER branch: the table spawned 0.70 m up and every item fell to the
            # floor beneath it.)
            "top_offset": 0.70, "height": 0.70, "kinematic": True,
        }
        # The parent stretches the packing bench 1.5x in depth; the kitchen table is
        # authored at its intended proportions.
        self.table_depth_scale = 1.0
        self.workbench_usd = str(table_usd)
        super().__post_init__()
        self.manifest = tuple(
            (n, k, f, s * self.rescale.get(n, 1.0), m) for n, k, f, s, m in self.manifest
        )
        self._assert_reach_separation()

    def _assert_reach_separation(self) -> None:
        """The plate must be unreachable from every pick-side stance, by margin, at build."""
        half_depth = 0.50   # authored top is 1.00 deep
        # The closest a pick-side shoulder can get to the table plane: pelvis clears the
        # near edge (torso half-depth ~0.13) and the shoulder rides ~0.0 ahead of the pelvis.
        closest_shoulder_y = -half_depth - 0.13
        px, py = self.plate_pos
        # worst case: shoulder slides freely in x to face the plate
        gap = py - closest_shoulder_y
        if gap < self.min_cross_reach:
            raise ValueError(
                f"plate at y={py:+.2f} is only {gap:.2f} m from the closest pick-side "
                f"shoulder; the cross-table walk needs >= {self.min_cross_reach:.2f} m")


@SCENES.register("fruit_delivery")
class FruitDeliveryScene(FruitsOnPlateScene):
    cfg: FruitDeliverySceneCfg

    def __init__(self, cfg: FruitDeliverySceneCfg | None = None) -> None:
        super().__init__(cfg or FruitDeliverySceneCfg())

    def describe(self) -> str:
        fruit = [self.PROSE[n] for n, _k, f, _s, _m in self.cfg.manifest if f]
        other = [self.PROSE[n] for n, _k, f, _s, _m in self.cfg.manifest if not f]
        join = lambda xs: ", ".join(xs[:-1]) + (", and " + xs[-1] if len(xs) > 1 else xs[0])  # noqa: E731
        d = 2 * self.cfg.plate_radius * 100
        return (
            f"A wooden kitchen table holds a mix of items on the side nearest you: fruit — "
            f"{join(fruit)} — scattered among things that are not fruit: {join(other)}. A "
            f"large round clay plate (about {d:.0f} cm across, a shallow dish) stands on the "
            f"FAR side of the same table, well beyond arm's reach from where the items lie.\n"
            "Goal: pick up every FRUIT and place it on the plate, leaving everything else "
            "where it is. You cannot reach the plate from the near side — carry each fruit "
            "around the table and place it from the far side. Note that not all produce is "
            "fruit: the pumpkins and the onion belong on the table, not the plate. A fruit "
            "counts only when it is resting on the plate; the job is done when every fruit "
            "is on the plate and nothing else has been put on it."
        )
