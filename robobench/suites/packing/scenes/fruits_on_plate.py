"""FruitsOnPlateScene — RoboLab's **FruitsOnPlateTask**, ported.

Registered as `fruits_on_plate`; runnable presets are `packing.fruits_on_plate` (NullRobot
oracle, the full 14-item RoboLab set), `packing.fruits_on_plate.franka.{osc,diff_ik,pink_ik,
joint}` and `packing.fruits_on_plate.g1.{joint,pink_ik}`.

A port of NVlabs/RoboLab's `FruitsOnPlateTask`
(github.com/NVlabs/RoboLab/blob/main/robolab/tasks/benchmark/fruits_to_plate.py): a kitchen
work table strewn with FRUIT mixed among things that are not fruit — two pumpkins, a red onion,
a serving bowl, wooden spoons, a spatula, a storage box — and a large round clay plate standing
on the table. **Goal (carried here, no task layer): pick up every fruit and place it on the
plate, leaving everything else where it is.**

The identification is finer-grained than the sibling `clear_organic_objects` scene, and that is
the point of having both. There, the split is FOOD vs NON-FOOD, and a pumpkin and an onion are
targets. Here the split is FRUIT vs NOT-FRUIT, so the pumpkins and the onion are DISTRACTORS
even though they are produce, sitting on the same table as the lemons and limes. An agent that
resolves "organic" cannot pass this task by the same reasoning; it has to resolve "fruit".
RoboLab draws the line in exactly this place — its success term lists the seven fruits and its
contact list carries the pumpkins, the onion and the tableware as scenery.

The GOAL SURFACE is a plate, not a container, and that changes the rubric rather than just the
target pose. RoboLab judges with `object_on_top(fruit, "clay_plates",
require_gripper_detached=True)`, which is the AND of a contact-force cone test (the surface
pushes up on the fruit within 45 deg of vertical) and a footprint test (the fruit's centroid xy
lies inside the surface's bounds). This scene keeps the footprint half verbatim and replaces the
contact-cone half with a settle gate, for the same reason the sibling scene does: contact-force
reads are embodiment- and solver-sensitive, while "resting still, inside the footprint, above
the dish floor" is the thing a rubric can state robot-agnostically.

Judged in the PLATE'S BODY FRAME (the pen-holder / tool-packing lesson: a receptacle judges
identically wherever it sits). A fruit is ON the plate when, measured from the plate's
GEOMETRIC CENTRE (not its prim origin — see `PLATE_MIN`/`PLATE_MAX`, the asset's origin sits
13 mm off-centre in x and 19 mm in y), its radial distance is within `radius_frac` of the
scaled plate radius AND its local z lies between `floor_local_z` and the scaled rim height plus
`stack_local_z`. A fruit is PLACED when it is on the plate AND settled (|v| < `settle_speed`).

`success()` = every PRESENT fruit placed on the plate AND nothing that is not a fruit on it
(the identification is the task: sweeping the whole table onto the plate is NOT success). This
is stricter than RoboLab's own fruits-only check — the deviation is deliberate and documented,
and it is the same deviation the sibling scene makes. RoboLab's `require_gripper_detached`
clause is an EMBODIMENT clause (the arm not still holding a placed item), checked at the robot-
binding / harness layer per the repo convention, not in this robot-agnostic scene — here the
`settle_speed` gate already rejects an item still being carried.

`score()` 0..100: per-present-fruit progress up to 90, minus 10 per non-fruit on the plate, 100
only once every present fruit is placed and the plate holds nothing else.

Per-episode randomization (task-family knobs): item scatter poses (grid slot + xy jitter + free
yaw), optional slot shuffle, and optional fruit-subset sampling (`subset_sample`) so a memorised
fixed pick list fails and a graded floor of episodes stays solvable. Absent fruits park in an
off-camera ground depot (InteractiveScene cannot despawn).

Assets: the plate and the kitchenware are the RoboLab `hot3d` assets, vendored by
`scripts/vendor_fruits_on_plate_assets.py` into `suites/packing/assets/fruits_on_plate/`. The
PRODUCE is the identical RoboLab `fruits_veggies` / `vomp` set already vendored for the sibling
scene, so this one reads it from `suites/packing/assets/clear_organic_objects/` rather than
keeping a second ~190 MB copy of the same meshes in the repo.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FruitsOnPlateSceneCfg(BaseCfg):
    """Config for `FruitsOnPlateScene`. Plain fields (the suite convention); a variant is a copy
    with a few changed. Placement is table-relative xy unless noted."""

    # --- rubric / judging (plate footprint derived from the measured bbox x scale) ----------
    settle_speed: float = 0.06  # max item |lin vel| when judging placed (m/s)
    radius_frac: float = 0.88  # counted radius = scaled plate radius * this. The plate's outer
    # 12% is its upturned rim: a fruit whose CENTRE sits out there is perched on the rim, not
    # resting in the dish, and will roll off. 0.88 -> a 133 mm counted radius on the unscaled
    # 151 mm plate, which still counts a fruit resting against the inside of the rim (its
    # centre is ~rim_radius - fruit_radius).
    floor_local_z: float = 0.015  # min local z above the plate's base to count as "on" (m). The
    # dish floor sits 11.5 mm above the base at the centre and rises to the 47 mm rim, so a
    # fruit resting in the dish clears this comfortably while the radial test does the real
    # work — as in RoboLab, where the footprint term is the discriminating one.
    stack_local_z: float = 0.10  # extra local z above the rim an item may pile to and still
    # count (a fruit resting ON other fruit inside the dish, its centre over the footprint).
    # The full seven-fruit set does not fit the dish in one layer, so piling is expected.

    # --- randomization (the task-family knobs) --------------------------------------------
    reset_pos_jitter: float = 0.025  # uniform +/- xy jitter per item at reset (m)
    reset_yaw_deg: float = 180.0  # uniform +/- yaw per item at reset (items lie at rest)
    reset_yaw_center_deg: float = 0.0  # centre of that yaw range. Non-zero lets a binding keep
    # a WIDE orientation range while steering it away from orientations its embodiment cannot
    # grasp: a hand with a limited approach-azimuth band has a matching band of item yaws it
    # can handle, and centring the range on that band preserves randomization instead of
    # shrinking it. Yaw is applied about z, so the item's local +x axis ends up along this angle.
    shuffle_slots: bool = True  # per-episode random item->scatter-slot permutation
    subset_sample: bool = False  # sample a subset of fruits present (demo/oracle: False)
    min_fruits: int = 3  # per-episode lower bound of sampled fruit count
    max_fruits: int | None = None  # per-episode upper bound; None -> all fruits. Set both
    # bounds equal to present EXACTLY k fruits drawn from a larger pool, which is how a binding
    # randomizes WHICH fruit appears without changing HOW MANY.
    drop_lift: float = 0.03  # spawn clearance above the table before settling (m)

    # --- placement (table-relative xy; the table itself sits at TABLES pos) ---------------
    surface_z: float | None = None  # work-surface height (m); None -> the preset's
    plate_pos: tuple = (0.34, -0.32)  # the plate's GEOMETRIC CENTRE on the surface (RoboLab
    # stands it to one side of the produce; its own layout has it at (0.40, -0.32) in the robot
    # frame with the fruit spread over x 0.30-0.63, y -0.35..0.39). Pushed out far enough that
    # the nearest grid slot's centre is 237 mm from the dish centre — 87 mm of clear table
    # outside the rim, so a settling item cannot end up resting against the rim and then get
    # nudged ONTO the plate, which would read as a non-fruit misplaced by the scene itself.
    plate_scale: float = 1.0  # UNIFORM only — a plate that is not round is not this asset.
    # RoboLab spawns it at 1.0 -> a 301 mm dish, 47 mm to the rim, floor 12 mm above the table.
    plate_yaw_deg: float = 0.0
    plate_mass: float = 1.5  # unused while kinematic; kept for a dynamic-plate variant
    plate_kinematic: bool = True  # a plate standing on the bench is a fixed target here. A
    # DYNAMIC plate is a difficulty variant, not the default: a 300 mm dish struck off-centre by
    # a released fruit both slides and tips, which turns every placement into a stability
    # problem on top of the identification the task exists to measure.
    scatter_center: tuple = (0.0, 0.10)  # centre of the clutter grid on the surface
    scatter_span: tuple = (0.46, 0.42)  # clutter grid extent (x, y) in m
    scatter_cols: int = 4  # grid columns (rows derived from item count)
    #: ((instance, x, y, yaw_deg), ...) explicit table-relative poses that REPLACE the grid slot
    #: for those instances — they also drop out of the slot shuffle and out of the yaw and xy
    #: randomization, because an explicitly placed item is explicitly placed.
    #:
    #: This exists for the long flat utensils. `wooden_spoons` is 309 mm along its own y and
    #: `spatula` is 332 mm along its own x — both longer than any sane grid pitch, so on the grid
    #: they spawn inside their neighbours and the table shoves itself apart while settling; and
    #: under free yaw they sweep a 309 mm circle, which no amount of grid margin survives. Given
    #: their own slots along the far edge at a fixed yaw they lie flat, still, and clear:
    #: the spoons occupy x [-0.415, -0.105] and the spatula x [0.014, 0.346], both in
    #: y [0.455, 0.545], 145 mm beyond the grid's far row and 119 mm apart from each other.
    slot_override: tuple = (
        ("wooden_spoons", -0.26, 0.50, 90.0),  # yaw 90 -> its 309 mm axis lies along table x
        ("spatula", 0.18, 0.50, 0.0),  # authored lying along its own +x already
    )

    # --- work surface (the pen_holder / tool_packing vendored-table pattern) --------------
    table: str = "packing"  # "lab_table" | "packing"
    table_depth_scale: float = 1.5
    workbench_pos: tuple[float, float] | None = None
    workbench_usd: str = ""
    surface_light: float = 2500.0
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0,
                      "pos": (0.40, -0.03), "top_offset": 0.0, "height": 1.05,
                      "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"),
                    "scale": 0.01, "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994,
                    "pos": (0.0, 0.0), "top_offset": 0.994, "height": 0.994,
                    "kinematic": True},
    }

    # --- structure (bboxes measured at vendor time from the MESH POINTS) ------------------
    # `assets/fruits_on_plate/extents.json` for the tableware, `assets/clear_organic_objects/
    # extents.json` for the produce.
    plate_key: ClassVar[str] = "clay_plates"
    PLATE_BBOX: ClassVar[tuple] = (0.30060, 0.30198, 0.04711)  # UNSCALED size (m)
    PLATE_MIN: ClassVar[tuple] = (-0.16307, -0.13234, 0.00017)  # UNSCALED, in the asset frame
    PLATE_MAX: ClassVar[tuple] = (0.13752, 0.16965, 0.04728)
    # The asset's origin is NOT its centre: the dish centre sits at (-0.0128, +0.0187) from the
    # prim origin, which is why `plate_pos` is defined as the geometric centre and the spawn
    # back-solves the origin. Getting that wrong puts the counted footprint 23 mm off the visible
    # dish — the sort of error that shows up as "the fruit is plainly on the plate but does not
    # count".
    # Radial profile, measured over the mesh (r from the dish centre): floor 11.5 mm above the
    # base at r=0, rising to the 47.3 mm rim at r=152 mm; the plate touches the table on a foot
    # ring at r~70 mm. So it is a shallow DISH, ~36 mm deep — a released fruit rolls toward the
    # middle and stays, which is what makes a single-arm release over the centre reliable.

    # manifest: (instance, asset_key, is_fruit, spawn_scale, mass_kg). The 7 fruits are
    # RoboLab's success list verbatim; the 7 non-fruits are its contact-list scenery. Instance
    # names are RoboLab's (`redonion`, not `red_onion` — that is the ASSET key). lime01 and
    # lime01_01 both spawn the one vendored lime asset, as RoboLab does.
    MANIFEST: ClassVar[tuple] = (
        # --- fruit (the targets) ---
        ("lemon_01", "lemon1", True, 1.0, 0.10),
        ("lemon_02", "lemon2", True, 1.0, 0.08),
        ("lime01", "lime", True, 1.0, 0.10),
        ("lime01_01", "lime", True, 1.0, 0.10),
        ("orange_01", "orange1", True, 0.87, 0.12),
        ("orange_02", "orange2", True, 0.88, 0.11),
        ("pomegranate01", "pomegranate", True, 0.56, 0.26),
        # Produce scales are the sibling scene's, unchanged, and they obey ONE constraint:
        # minimum horizontal span <= 64 mm, so a parallel gripper (Panda: 80 mm aperture) keeps
        # ~8 mm of finger travel per side on a curved surface. Within that ceiling the sizes stay
        # realistic and mutually distinct. Spans: pomegranate 64 > orange 63 > pumpkinlarge 62 >
        # lime 61 > onion 59 > pumpkinsmall 55; the lemons are grasped across their 50/40 mm
        # short axis.
        # --- not fruit (the distractors) ---
        # The two gourds and the onion are the interesting ones: they are produce, they sit among
        # the fruit, and putting either on the plate FAILS the task.
        ("pumpkinlarge", "pumpkinlarge", False, 0.70, 0.20),
        ("pumpkinsmall", "pumpkinsmall", False, 0.72, 0.13),
        ("redonion", "red_onion", False, 1.0, 0.12),
        ("serving_bowl", "serving_bowl", False, 1.0, 0.30),
        ("storage_box", "storage_box", False, 1.5, 0.25),  # RoboLab's own 1.5 -> 128x99x185 mm
        ("wooden_spoons", "wooden_spoons", False, 1.0, 0.08),  # 68 x 309 x 18 mm
        ("spatula", "spatula", False, 1.0, 0.09),  # 332 x 90 x 42 mm, authored lying along +x
    )
    #: asset keys that come from the sibling scene's vendored produce directory rather than
    #: this scene's own. Split by ASSET, not by instance — `redonion` the distractor and
    #: `lemon_01` the target are both produce meshes.
    PRODUCE_KEYS: ClassVar[frozenset] = frozenset({
        "lemon1", "lemon2", "lime", "orange1", "orange2", "pomegranate",
        "pumpkinlarge", "pumpkinsmall", "red_onion", "serving_bowl",
    })

    # Instances to leave OUT of this variant, by manifest name. The full 14-object RoboLab set is
    # the default (and what the NullRobot oracle exercises); an ARM binding may drop items whose
    # difficulty is incidental rather than intended.
    exclude: tuple = ()
    contact_offset: float = 0.004  # item speculative contact margin (m)
    item_static_friction: float = 1.1  # produce skin vs rubber gripper pads (see reset())
    item_dynamic_friction: float = 0.95
    asset_dir: str = ""
    produce_dir: str = ""

    # derived (filled in __post_init__)
    manifest: tuple = field(default=None, init=False)  # MANIFEST minus `exclude`
    plate_usd: str = field(default="", init=False)
    item_usds: dict = field(default=None, init=False)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "fruits_on_plate")
        self.produce_dir = self.produce_dir or str(assets / "clear_organic_objects")
        self.plate_usd = str(Path(self.asset_dir) / self.plate_key / f"{self.plate_key}.usd")
        self.manifest = tuple(m for m in self.MANIFEST if m[0] not in self.exclude)
        if not any(m[2] for m in self.manifest):
            raise ValueError(f"exclude={self.exclude} leaves no fruit to place")
        keys = {k for _n, k, _f, _s, _m in self.manifest}
        self.item_usds = {
            k: str(Path(self.produce_dir if k in self.PRODUCE_KEYS else self.asset_dir)
                   / k / f"{k}.usd")
            for k in keys
        }
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets.parents[1] / "assembly" / "assets" / "props" / preset["usd"][0]
            / preset["usd"][1])

    # ----- derived plate geometry (scaled; used by the scene AND by layout math) -----------
    @property
    def plate_radius(self) -> float:
        """Scaled plate radius (m), the mean of the two bbox half-extents (it is round)."""
        return (self.PLATE_BBOX[0] + self.PLATE_BBOX[1]) / 4 * self.plate_scale

    @property
    def plate_center_offset(self) -> tuple[float, float]:
        """Scaled (x, y) from the plate PRIM ORIGIN to its geometric centre, in the plate's own
        frame at yaw 0."""
        return ((self.PLATE_MIN[0] + self.PLATE_MAX[0]) / 2 * self.plate_scale,
                (self.PLATE_MIN[1] + self.PLATE_MAX[1]) / 2 * self.plate_scale)

    @property
    def plate_rim_z(self) -> float:
        """Scaled local z of the plate rim above its prim origin (m)."""
        return self.PLATE_MAX[2] * self.plate_scale


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("fruits_on_plate")
class FruitsOnPlateScene(BaseScene):
    cfg: FruitsOnPlateSceneCfg

    def __init__(self, cfg: FruitsOnPlateSceneCfg | None = None) -> None:
        super().__init__(cfg or FruitsOnPlateSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, the vendored work table, the kinematic plate, and the 14 scattered
        items (7 fruits + 7 non-fruits). Colliders are the assets' AUTHORED ones — the plate
        keeps its concave mesh collider so a fruit settles INSIDE the dish, not on a hull that
        would dome over it."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in [c.plate_usd, *c.item_usds.values()]:
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — vendor the assets first "
                    f"(python scripts/vendor_fruits_on_plate_assets.py, and "
                    f"scripts/vendor_clear_organic_objects_assets.py for the produce)")
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        z0 = c.surface_z
        table_z = z0 - preset["top_offset"]
        ground_z = z0 - preset["height"]
        s = preset["scale"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd,
                                           scale=(s, s * c.table_depth_scale, s))
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        pox, poy = self._plate_origin_xy()

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(c.asset_dir).parents[2] / "assembly" / "assets" / "props"
                    / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.surface_light, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z),
                                                       rot=preset["orient"]),
                spawn=table_spawn,
            ),
            "plate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.plate_usd,
                    scale=(c.plate_scale,) * 3,
                    # keep the AUTHORED mesh collider (concave — the dish); only arm the rigid
                    # body. Kinematic -> a plate that stays where it was set down.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=c.plate_kinematic,
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + pox, wy + poy, z0),
                    rot=self._yaw_quat(c.plate_yaw_deg)),
            ),
        }
        for i, (name, key, _fruit, scale, mass) in enumerate(c.manifest):
            sx, sy = self._slot_xy(i)
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.item_usds[key],
                    scale=(scale,) * 3,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05,
                        angular_damping=0.10,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + sx, wy + sy, z0 + c.drop_lift + 0.03 * (i % 2)),
                    # spawn yaw matters for the long utensils: at their authored yaw the spatula
                    # crosses the whole far edge, so the first frame — before `reset()` runs —
                    # would start it inside its neighbour
                    rot=self._yaw_quat((self._fixed_pose(name) or (0, 0, 0.0))[2])),
            )
        return out

    # ----- geometry helpers -------------------------------------------------------------------
    @staticmethod
    def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
        h = math.radians(deg) / 2
        return (math.cos(h), 0.0, 0.0, math.sin(h))

    def _plate_origin_xy(self) -> tuple[float, float]:
        """Table-relative xy for the plate's PRIM ORIGIN such that its geometric centre lands on
        `plate_pos`. The origin-to-centre offset rotates with `plate_yaw_deg`."""
        c = self.cfg
        ox, oy = c.plate_center_offset
        t = math.radians(c.plate_yaw_deg)
        rx = ox * math.cos(t) - oy * math.sin(t)
        ry = ox * math.sin(t) + oy * math.cos(t)
        return (c.plate_pos[0] - rx, c.plate_pos[1] - ry)

    def _fixed_pose(self, name: str) -> tuple[float, float, float] | None:
        """(x, y, yaw_deg) if `name` has an explicit `slot_override`, else None."""
        for nm, x, y, yaw in self.cfg.slot_override:
            if nm == name:
                return (x, y, yaw)
        return None

    def _grid_names(self) -> list[str]:
        """Manifest instances that take a GRID slot, in manifest order (overrides removed)."""
        fixed = {n for n, _x, _y, _yaw in self.cfg.slot_override}
        return [n for n, _k, _f, _s, _m in self.cfg.manifest if n not in fixed]

    def _slot_xy(self, i: int) -> tuple[float, float]:
        """Table-relative xy of manifest item `i`: its explicit `slot_override` if it has one,
        else its slot on the grid centred at `scatter_center`."""
        c = self.cfg
        name = c.manifest[i][0]
        fixed = self._fixed_pose(name)
        if fixed is not None:
            return (fixed[0], fixed[1])
        grid = self._grid_names()
        j = grid.index(name)
        n = len(grid)
        cols = c.scatter_cols
        rows = math.ceil(n / cols)
        r, col = divmod(j, cols)
        cx, cy = c.scatter_center
        sx, sy = c.scatter_span
        # spread columns along x, rows along y, centred on scatter_center
        fx = (col - (cols - 1) / 2) / max(cols - 1, 1)
        fy = (r - (rows - 1) / 2) / max(rows - 1, 1)
        return (cx + fx * sx, cy + fy * sy)

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

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, resolve fruit/non-fruit index sets, precompute the counted plate
        footprint (plate body frame), and allocate the per-episode present mask."""
        super().bind(env)
        c = self.cfg
        dev = env.device
        self.plate: RigidObject = env.iscene["plate"]
        self.items: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _k, _f, _s, _m in c.manifest}
        self.names = [name for name, _k, _f, _s, _m in c.manifest]
        self.env_origins = env.iscene.env_origins
        self._fruit = torch.tensor([f for _n, _k, f, _s, _m in c.manifest],
                                   dtype=torch.bool, device=dev)
        self._fruit_idx = torch.nonzero(self._fruit, as_tuple=False).flatten()
        self._other_idx = torch.nonzero(~self._fruit, as_tuple=False).flatten()
        # counted footprint in the plate's body frame (origin at the plate base, off-centre)
        self._plate_center = torch.tensor(c.plate_center_offset, device=dev)
        self._plate_r = c.radius_frac * c.plate_radius
        self._plate_floor = c.floor_local_z
        self._plate_ceil = c.plate_rim_z + c.stack_local_z
        # present[e, i]: item i participates (non-fruits always; fruits maybe sampled)
        self.present = torch.ones(env.num_envs, len(c.manifest), dtype=torch.bool, device=dev)
        self._friction_written = False

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plate at its (kinematic) pose; optionally sample the present fruit
        subset; scatter present items on the (optionally shuffled) grid with xy jitter + free
        yaw at a small lift above the table (settled by the following steps); park absent fruits
        in the ground depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        wx, wy = c.workbench_pos
        z0 = c.surface_z

        # --- produce friction (once): PhysX shape materials, CPU tensors are the view's -----
        # contract. `UsdFileCfg` has no `physics_material` field, so the only way to author this
        # for USD-spawned items is the raw view. Without it the items keep PhysX's default mu
        # (~0.5) and a 60-75 mm smooth sphere in a gripper has too little friction to hold (the
        # sibling scene measured exactly this: lemons and squat fruit picked reliably while every
        # large round fruit slipped on the lift or mid-carry). Real fruit skin against rubber
        # pads is mu ~0.8-1.2.
        if not self._friction_written:
            for body in self.items.values():
                view = body.root_physx_view
                mp = view.get_material_properties().clone()  # (N, shapes, 3)
                mp[..., 0] = c.item_static_friction
                mp[..., 1] = c.item_dynamic_friction
                mp[..., 2] = 0.0
                view.set_material_properties(mp, torch.arange(view.count, device="cpu"))
            self._friction_written = True

        # --- present mask: non-fruits always in; fruits optionally subset-sampled ---
        self.present[env_ids] = True
        if c.subset_sample:
            n_fruit = int(self._fruit.sum())
            hi = n_fruit if c.max_fruits is None else min(c.max_fruits, n_fruit)
            lo = min(c.min_fruits, hi)
            k = torch.randint(lo, hi + 1, (m,), device=dev)
            rank = torch.rand(m, n_fruit, device=dev).argsort(dim=1).argsort(dim=1)
            self.present[env_ids.unsqueeze(1), self._fruit_idx.unsqueeze(0)] = rank < k.unsqueeze(1)

        # --- plate: kinematic pose (also covers a dynamic plate shoved last episode) ---
        pox, poy = self._plate_origin_xy()
        q = self._yaw_quat(c.plate_yaw_deg)
        proot = torch.zeros(m, 13, device=dev)
        proot[:, 0] = wx + pox
        proot[:, 1] = wy + poy
        proot[:, 2] = z0
        proot[:, 3] = q[0]
        proot[:, 6] = q[3]
        proot[:, 0:3] += origin
        self.plate.write_root_state_to_sim(proot, env_ids)

        # --- items: grid slot (optionally permuted) + jitter + free yaw; absent -> depot ---
        n = len(c.manifest)
        if c.shuffle_slots:
            # Shuffle WITHIN groups (fruit among the fruit slots, the rest among their own), not
            # across all slots. A free permutation lets a fruit spawn in the serving bowl's slot
            # and roll INSIDE it, and a fruit nested in a bowl cannot be reached by a top-down
            # grasp at all — a degenerate case rather than task difficulty (the sibling scene
            # watched exactly this happen on video). Items with a `slot_override` keep their slot.
            fixed = torch.tensor(
                [self._fixed_pose(name) is not None for name in self.names],
                dtype=torch.bool, device=dev)
            perm = torch.arange(n, device=dev).expand(m, n).clone()
            for group in (self._fruit_idx, self._other_idx):
                movable = group[~fixed[group]]
                if len(movable) < 2:
                    continue
                order = torch.rand(m, len(movable), device=dev).argsort(dim=1)
                perm[:, movable] = movable[order]
        else:
            perm = torch.arange(n, device=dev).expand(m, n)
        slots = torch.tensor([self._slot_xy(i) for i in range(n)], device=dev)  # (n, 2)
        yaw_amp = math.radians(c.reset_yaw_deg)
        for i, (name, _k, _f, _s, _mass) in enumerate(c.manifest):
            st = torch.zeros(m, 13, device=dev)
            fixed_pose = self._fixed_pose(name)
            st[:, 0] = wx + slots[perm[:, i], 0]
            st[:, 1] = wy + slots[perm[:, i], 1]
            st[:, 2] = z0 + c.drop_lift + 0.03 * (i % 2)
            if fixed_pose is None:
                st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
                yaw = (math.radians(c.reset_yaw_center_deg)
                       + (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2)
            else:
                yaw = torch.full((m,), math.radians(fixed_pose[2]), device=dev)
            # HALF the angle — a z-yaw quaternion is (cos(yaw/2), 0, 0, sin(yaw/2)). Writing
            # (cos yaw, 0, 0, sin yaw) silently DOUBLES every commanded orientation, which is
            # invisible for a free +/-180 range (it still covers the circle) and wrong the moment
            # a binding pins the yaw: `reset_yaw_center_deg=90` then lands the item at 180 deg, so
            # the axis a solver was told to expect at azimuth 180 actually presents at 90. Caught
            # here by measuring the settled OBB spans — the lemon's 50 mm narrow axis showed up at
            # az 90 with a 76 mm span at az 180, exactly the doubled pose.
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            # absent fruits -> off-camera ground depot (below the surface, on the floor)
            absent = ~self.present[env_ids, i]
            if absent.any():
                st[absent, 0] = wx + 1.2 + 0.16 * (i % 3)
                st[absent, 1] = wy + 1.2 + 0.16 * (i // 3)
                st[absent, 2] = z0 - c.TABLES[c.table]["height"] + 0.05
            st[:, 0:3] += origin
            self.items[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "plate": self.plate.data.root_state_w[env_ids].clone(),
            "items": {n: b.data.root_state_w[env_ids].clone() for n, b in self.items.items()},
            "present": self.present[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.plate.write_root_state_to_sim(state["plate"], env_ids)
        for n, b in self.items.items():
            b.write_root_state_to_sim(state["items"][n], env_ids)
        self.present[env_ids] = state["present"]

    # ----- description ------------------------------------------------------------------------
    #: manifest instance name -> the words used in `describe()` (the agent reads this text).
    PROSE: ClassVar[dict[str, str]] = {
        "lemon_01": "a lemon", "lemon_02": "a small lemon", "lime01": "a lime",
        "lime01_01": "a second lime", "orange_01": "an orange", "orange_02": "an orange",
        "pomegranate01": "a pomegranate", "pumpkinlarge": "a pumpkin",
        "pumpkinsmall": "a small pumpkin", "redonion": "a red onion",
        "serving_bowl": "a serving bowl", "storage_box": "a storage box",
        "wooden_spoons": "a pair of wooden spoons", "spatula": "a spatula",
    }

    def describe(self) -> str:
        # Built from the LIVE manifest, so a variant that excludes items describes itself
        # honestly instead of promising fruit that is not on the table.
        fruit = [self.PROSE[n] for n, _k, f, _s, _m in self.cfg.manifest if f]
        other = [self.PROSE[n] for n, _k, f, _s, _m in self.cfg.manifest if not f]
        join = lambda xs: ", ".join(xs[:-1]) + (", and " + xs[-1] if len(xs) > 1 else xs[0])  # noqa: E731
        d = 2 * self.cfg.plate_radius * 100
        return (
            f"A kitchen work table holds a mix of items: fruit — {join(fruit)} — scattered "
            f"among things that are not fruit: {join(other)}. A large round clay plate (about "
            f"{d:.0f} cm across, a shallow dish) stands on one side of the table.\n"
            "Goal: pick up every FRUIT and place it on the plate, leaving everything else where "
            "it is. Note that not all produce is fruit — the pumpkins and the onion belong on "
            "the table, not the plate. A fruit counts only when it is resting on the plate; the "
            "job is done when every fruit is on the plate and nothing else has been put on it."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _on_plate(self) -> torch.Tensor:
        """(N, n_items) bool: each item's origin inside the counted plate volume, computed in the
        plate's body frame (plate motion is irrelevant) and measured radially from the plate's
        GEOMETRIC CENTRE."""
        from isaaclab.utils.math import quat_apply_inverse

        pp = self.plate.data.root_pos_w  # (N, 3) — the prim origin, not the dish centre
        pq = self.plate.data.root_quat_w  # (N, 4)
        cols = []
        for name in self.names:
            loc = quat_apply_inverse(pq, self.items[name].data.root_pos_w - pp)
            radial = (loc[:, :2] - self._plate_center).norm(dim=-1)
            inside_xy = radial <= self._plate_r
            inside_z = (loc[:, 2] >= self._plate_floor) & (loc[:, 2] <= self._plate_ceil)
            cols.append(inside_xy & inside_z)
        return torch.stack(cols, dim=1)

    def _speed(self) -> torch.Tensor:
        """(N, n_items) |lin vel| per item."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                            for b in self.items.values()], dim=1)

    def placed(self) -> torch.Tensor:
        """(N, n_items) bool: item on the plate AND settled."""
        return self._on_plate() & (self._speed() < self.cfg.settle_speed)

    def fruits_placed(self) -> torch.Tensor:
        """(N, n_fruits) bool: each fruit placed (on the plate + settled), in fruit order."""
        return self.placed()[:, self._fruit_idx]

    def nonfruits_on_plate(self) -> torch.Tensor:
        """(N, n_others) bool: each non-fruit currently on the plate (misplaced)."""
        return self._on_plate()[:, self._other_idx]

    def fruits_present(self) -> torch.Tensor:
        """(N, n_fruits) bool: fruit sampled present this episode."""
        return self.present[:, self._fruit_idx]

    def score(self) -> torch.Tensor:
        """(N,) int 0..100: per-present-fruit progress up to 90, minus 10 per non-fruit on the
        plate, 100 once every present fruit is placed and the plate holds nothing else."""
        pres = self.fruits_present()
        done = self.fruits_placed() & pres
        n_pres = pres.sum(dim=1).clamp(min=1)
        frac = done.sum(dim=1).float() / n_pres.float()
        base = (90.0 * frac).round().to(torch.long)
        misplaced = self.nonfruits_on_plate().sum(dim=1)
        base = (base - 10 * misplaced).clamp(min=0, max=90)
        complete = (done | ~pres).all(dim=1) & (misplaced == 0)
        return torch.where(complete, torch.full_like(base, 100), base)

    def success(self) -> torch.Tensor:
        """(N,) bool: every present fruit placed on the plate AND nothing that is not a fruit on
        it (scene-level success; the NullRobot oracle's target). The 'gripper detached' clause is
        an embodiment clause checked at the binding/harness layer, not here."""
        pres = self.fruits_present()
        fruits_done = (self.fruits_placed() | ~pres).all(dim=1)
        return fruits_done & ~self.nonfruits_on_plate().any(dim=1)
