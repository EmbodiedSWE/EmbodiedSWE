"""ClearOrganicObjectsScene — RoboLab's **ClearOrganicObjectsTask**, ported.

Registered as `clear_organic_objects`; runnable presets are `packing.clear_organic_objects`
(NullRobot oracle), `packing.clear_organic_objects.franka.{osc,diff_ik,pink_ik,joint}` and
`packing.clear_organic_objects.g1.{joint,pink_ik}` (the G1 humanoid works a resized cell — a
smaller bin and the four smallest organics — because its pinch hand and fixed pelvis are not an
80 mm jaw on a 0.8 m arm; see `_clear_organic_objects_g1_cfg`).

A port of NVlabs/RoboLab's `ClearOrganicObjectsTask`
(github.com/NVlabs/RoboLab/blob/main/robolab/tasks/benchmark/clutter_organic_objects_task.py):
a work table strewn with organic fruits & vegetables mixed among non-food clutter (bottles,
jugs, a serving bowl, a pen holder), and an open bin off to one side. **Goal (carried here,
no task layer): identify every organic item — the fruits and vegetables — and place each into
the bin, leaving the non-food clutter on the table.**

The objects are the RoboLab assets themselves, vendored by
`scripts/vendor_clear_organic_objects_assets.py` into `suites/packing/assets/clear_organic_objects/` (real
scanned USDs — Z-up, metres, authored RigidBody + mesh colliders; textures downsampled to 2K).
The organics are the 11 RoboLab targets — two lemons, two limes, two oranges, a pomegranate, a
large and a small pumpkin, a red onion, an avocado; the clutter is the 5 RoboLab distractors —
a white packer bottle, a crab pen holder, a milk jug, a serving bowl, a utility jug. The bin is
RoboLab's `container_f24`, kept at its scene scale (0.3, 0.3, 0.2) -> a ~0.35 x 0.24 m open
crate, 0.13 m deep; spawned KINEMATIC (a fixed receptacle on the bench).

Judged in the BIN'S BODY FRAME (the pen-holder / tool-packing lesson: a container judges
identically wherever it sits): an item is IN the bin when its origin lies inside the bin's
interior box — |x| <= `wall_frac` * outer_half_x, |y| <= `wall_frac` * outer_half_y, and
`floor_local_z` <= z <= scaled rim height — computed from the measured `bin_bbox` x `bin_scale`.
An item is CLEARED when it is in the bin AND settled (|v| < `settle_speed`).

`success()` = every PRESENT organic cleared into the bin AND no distractor in the bin (the
identification is the task: dumping everything in is NOT success). This is stricter than
RoboLab's own `object_in_container(organics, "all")`, which does not check distractors — the
deviation is deliberate and documented. RoboLab's `require_gripper_detached` clause is an
EMBODIMENT clause (the arm not still holding a placed item), checked at the robot-binding /
harness layer per the repo convention (the pen-holder return-to-origin precedent), not in this
robot-agnostic scene — here the `settle_speed` gate already rejects an item still being carried.

`score()` 0..100: per-present-organic progress up to 90, minus 10 per distractor in the bin,
100 only once every present organic is cleared and no distractor is in the bin.

Per-episode randomization (task-family knobs): item scatter poses (grid slot + xy jitter +
free yaw), optional slot shuffle, and optional organic-subset sampling (`subset_sample`) so a
memorised fixed pick list fails and a graded floor of episodes stays solvable. Absent organics
park in an off-camera ground depot (InteractiveScene cannot despawn).

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
class ClearOrganicObjectsSceneCfg(BaseCfg):
    """Config for `ClearOrganicObjectsScene`. Plain fields (the suite convention); a variant is a copy
    with a few changed. Placement is table-relative xy unless noted."""

    # --- rubric / judging (bin interior derived from the measured bbox x scale) ------------
    settle_speed: float = 0.06  # max item |lin vel| when judging cleared (m/s)
    wall_frac: float = 0.90  # interior half-extent = outer half * this (bin wall inset). The
    # bin wall is thin (~1-2 cm); 0.90 keeps a fruit resting AGAINST the inner wall counted
    # (its centre is ~outer_half - radius). The z-gate (centre below the rim) rejects rim-perch.
    floor_local_z: float = 0.012  # min local z above the bin base to count as "in" (m)
    rim_frac: float = 1.0  # rim local z = scaled bin height * this (top opening)
    bin_rim_stack: float = 0.06  # extra local-z above the rim an item may pile to and still
    # count (an organic resting ON others inside the bin, its centre over the interior footprint)

    # --- randomization (the task-family knobs) --------------------------------------------
    reset_pos_jitter: float = 0.025  # uniform +/- xy jitter per item at reset (m)
    reset_yaw_deg: float = 180.0  # uniform +/- yaw per item at reset (items lie at rest)
    reset_yaw_center_deg: float = 0.0  # centre of that yaw range. Non-zero lets a binding keep
    # a WIDE orientation range while steering it away from orientations its embodiment cannot
    # grasp: a hand with a limited approach-azimuth band has a matching band of item yaws it
    # can handle, and centring the range on that band preserves randomization instead of
    # shrinking it. Yaw is applied about z, so the item's local +x axis ends up along this
    # angle.
    shuffle_slots: bool = True  # per-episode random item->scatter-slot permutation
    subset_sample: bool = False  # sample a subset of organics present (demo/oracle: False)
    min_organics: int = 4  # per-episode lower bound of sampled organic count
    max_organics: int | None = None  # per-episode upper bound; None -> all present organics.
    # Set both bounds equal to present EXACTLY k organics drawn from a larger pool, which is
    # how a binding randomizes WHICH produce appears without changing HOW MANY. A tier whose
    # embodiment can only handle one item at a time still wants type variety, or a solver can
    # memorise the single answer.
    drop_lift: float = 0.03  # spawn clearance above the table before settling (m)

    # --- two-destination sort (the long-horizon variant; all optional) ---------------------
    # With `tray_pos` set, the task becomes BUSSING THE TABLE, SORTED: the `sort_to_bin`
    # organics go into the bin AND the `sort_to_tray` items go onto the tray, each graded and
    # latched per item. Items in neither list are SCENERY: part of the colourful tableau (the
    # RoboLab original's richness is the point), never required to move, but still penalised
    # if dumped into the bin (identification stays live). With `tray_pos` None the scene
    # behaves exactly as before (single-destination clear).
    tray_pos: tuple | None = None  # tray (clay plate) centre on the surface, or None
    tray_scale: float = 1.3
    tray_key: ClassVar[str] = "clay_plates"  # vendored under fruits_on_plate's assets
    # Measured at vendor time (fruits_on_plate/extents.json): bbox and the geometric-centre
    # offset of the plate prim origin.
    tray_bbox: ClassVar[tuple] = (0.3006, 0.30198, 0.04711)
    tray_min: ClassVar[tuple] = (-0.16307, -0.13234, 0.00017)
    tray_max: ClassVar[tuple] = (0.13752, 0.16965, 0.04728)
    tray_margin_frac: float = 0.85  # counted radius = mean half-extent * this
    tray_stack: float = 0.14  # counted z band above the plate top (an item ON the plate)
    sort_to_bin: tuple = ()  # organic instance names REQUIRED in the bin ((), tray unset ->
    # every present organic, the legacy rule)
    sort_to_tray: tuple = ()  # instance names required ON the tray
    # Curated fixed poses, (name, x, y, yaw_deg[, z_lift]) table-relative: the tableau is
    # ARRANGED, not scattered — the original scene reads like a composed still life, and
    # reach margins for the movable items are measured per spot. Listed items skip the grid;
    # jitter still applies (reset_pos_jitter), so solutions cannot memorise millimetre
    # poses. The optional z_lift spawns an item that much higher, which is how one item is
    # cradled INSIDE another (the original nests its avocado in the serving bowl).
    fixed_layout: tuple = ()
    # A standing platform under the robot's feet (x, y, z_top, size_x, size_y), or None. The
    # G1's pelvis is welded at working height and its feet otherwise hang above the floor
    # once the robot stands BESIDE the table instead of inside it.
    riser: tuple | None = None

    # --- placement (table-relative xy; the table itself sits at TABLES pos) ---------------
    surface_z: float | None = None  # work-surface height (m); None -> the preset's
    bin_pos: tuple = (0.28, -0.26)  # bin centre on the surface (RoboLab: to one side)
    # RoboLab scales container_f24 to (0.3, 0.3, 0.2) -> a shallow 28x19 cm, 13 cm-deep crate.
    # That footprint cannot hold all 11 organics with their centres below the rim (they pile
    # above it and fail the in-bin z-test); enlarged here to a ~52x32 cm, 20 cm-deep produce
    # crate — wide enough to hold the full set in ~one layer, walls low enough for a single
    # arm to clear when dropping in. `bin_rim_stack` still allows a modest pile.
    # Sized off the recorded footage (2026-08-26): at (0.45,0.4,0.3) the crate dominated frame
    # and sat ~60% empty with all 11 organics in. (0.36,0.32,0.26) -> ~42 x 25 cm outer,
    # 17 cm deep: still ~2x the produce footprint, but a shorter carry for a single arm.
    bin_scale: tuple = (0.36, 0.32, 0.26)
    bin_yaw_deg: float = 0.0
    bin_mass: float = 10.0  # unused while kinematic; kept for a dynamic-bin variant
    bin_kinematic: bool = True  # fixed receptacle (dynamic-bin is a difficulty variant)
    scatter_center: tuple = (0.02, 0.14)  # centre of the clutter grid on the surface
    scatter_span: tuple = (0.46, 0.40)  # clutter grid extent (x, y) in m
    scatter_cols: int = 4  # grid columns (rows derived from item count)

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
        # The RoboLab clutter scene's own wooden table, extracted from the flattened
        # published scene (NVlabs/RoboLab assets/scenes/clutter_fruit_bottle_bluebin.usda,
        # /world/table subtree) so this port looks like its source. Unscaled it is
        # 0.70 x 1.00 m with the top at prim z ~0.003 and legs to -0.697; the -90 deg z
        # rotation lays its long axis along world x with the open side toward the robot,
        # and `z_scale` 1.114 stretches the legs so the top lands at the 0.78 work height
        # with the legs' feet exactly on the ground plane (z = 0) -- which is also what
        # lets the G1 stand naturally on the floor beside it, no riser needed.
        "robolab": {"usd": ("clear_organic_objects", "robolab_table/robolab_table.usda"),
                    "local": True, "scale": 1.0, "z_scale": 1.114,
                    "orient": (0.70711, 0.0, 0.0, -0.70711), "surface_z": 0.78,
                    # origin_offset centres the tabletop's GEOMETRY at workbench_pos (the
                    # prim origin sits at the table's corner region, not its centre); with it
                    # the top spans x[-0.5,+0.5] (x depth_scale), y[-0.35,+0.35] around the
                    # item frame's origin.
                    "pos": (0.0, 0.0), "origin_offset": (0.0, 0.547),
                    "top_offset": 0.003, "height": 0.78,
                    "kinematic": True,
                    # The extracted top's own collider is a few millimetres thick. Anything
                    # pressed into it -- a hand closing on a pumpkin, a toppling onion, a
                    # bowl taking an impact -- tunnels through and is catapulted by the
                    # depenetration (items were found 20-400 m away). An invisible 8 cm
                    # collision slab with its top at the work surface makes the tabletop
                    # solid. Size is the UNSCALED top footprint (x is multiplied by
                    # table_depth_scale in assets()).
                    "collision_slab": (1.0, 0.70, 0.08)},
    }

    # --- structure (bin bbox measured at vendor time; see assets/clear_organic_objects/extents.json) --
    bin_key: ClassVar[str] = "container_f24"
    bin_bbox: ClassVar[tuple] = (1.15838, 0.7958, 0.66998)  # UNSCALED outer bbox (m)
    # manifest: (instance, asset_key, is_organic, spawn_scale, mass_kg). The 11 organics are the
    # RoboLab targets; the 5 distractors the RoboLab clutter. lime01 + lime01_01 both spawn the
    # one vendored lime asset (RoboLab uses two instances of it).
    MANIFEST: ClassVar[tuple] = (
        ("lemon_01", "lemon1", True, 1.0, 0.10),
        # lemon_02 uses the SAME mesh as lemon_01 (was the lemon2 mesh at 1.2): the lemon1 mesh
        # is the one object the G1 hand grasps AND releases reliably, so every sort target is it.
        ("lemon_02", "lemon2", True, 1.0, 0.08),
        ("lime01", "lime", True, 1.0, 0.10),
        ("lime01_01", "lime", True, 1.0, 0.10),
        # Two more lemon-mesh produce, coloured lime-green (their own asset, below): the
        # sort tier's TRAY targets. The lemon mesh is the ONE object class the G1 hand grasps
        # and releases reliably (76x50 / 72x48 mm, ~700-point faceted collider); every other
        # candidate was measured out -- oranges crushed at the cage, hull avocado/onion/lime
        # seated well and slid out on the lift, faceted limes launched, bottle and pen holder
        # geometrically impossible. A lime is a green lemon-shaped fruit, so the sort still
        # reads naturally: yellow into the crate, green onto the tray.
        # `lemon_g` and `lime_g` are the lemon1 mesh in their OWN usds, the latter tinted
        # lime-green (scripts/make_lime_from_lemon.py). Own files for two measured reasons:
        # extra instances of the SAME usd were launched hundreds of metres whenever a sibling
        # instance was lifted 8-10 cm away (shared cooked collision data), and this tier's
        # grasp needs a coarser lemon collider than `fruits_on_plate` -- which reads produce
        # from this same directory -- was calibrated against.
        ("lemon_a", "lemon_g", True, 1.0, 0.10),
        ("lemon_b", "lemon_g", True, 1.0, 0.10),
        ("lime_a", "lime_g", True, 1.0, 0.10),
        ("lime_b", "lime_g", True, 1.0, 0.10),
        # oranges: the sort tier's TRAY targets, at their shipped 63 mm. Their collider is a
        # single convex hull (scripts/fix_clear_organic_colliders.py): with the source
        # decomposition of the 9.9k-point scan they blew up whenever the G1's fingers pressed a
        # seam (9.5 m, then a numerical explosion), and a 45 mm version slipped out of the hand
        # (lift 24-39 mm) -- the two stacked fingers need something they can both bear on.
        ("orange_01", "orange1", True, 0.87, 0.12),
        ("orange_02", "orange2", True, 0.88, 0.11),
        # Produce scales obey ONE constraint — minimum horizontal span <= 64 mm — so a parallel
        # gripper (Panda: 80 mm aperture) keeps ~8 mm of finger travel per side on a curved
        # surface. Within that ceiling sizes stay REALISTIC and mutually DISTINCT. Two measured
        # lessons behind those numbers (2026-08-27):
        #   - an earlier pass shrank everything into a 61-66 mm band, which made pumpkinlarge
        #     and pumpkinsmall IDENTICAL and left the pomegranate smaller than a lemon — both
        #     plainly wrong on camera (caught by tabulating the baked extents, not by eye);
        #   - the correction overshot to a 70 mm ceiling, leaving only ~5 mm per side, and the
        #     a parallel jaw stopped clearing the produce at all. 64 mm is the
        #     compromise that keeps both realism and a graspable margin.
        # Resulting spans: pomegranate 64 > orange 63 > pumpkinlarge 62 > lime 61 ~ avocado 61
        # > onion 59 > pumpkinsmall 55; the lemons are pinched across their 50/40 mm short axis.
        ("pomegranate01", "pomegranate", True, 0.56, 0.26),
        ("pumpkinlarge", "pumpkinlarge", True, 0.70, 0.20),
        ("pumpkinsmall", "pumpkinsmall", True, 0.72, 0.13),
        # 0.85 / 0.8 and LAID ON THEIR SIDE (fixed_layout roll): 50x51x77 and 49x49x74 mm --
        # the lemon's shape, which is the one shape the G1 hand holds (a ~50 mm cross-section
        # with length along the fingers). Their old explosive behaviour was the sliver-hull
        # decomposition of their scans, replaced by one convex hull (see
        # scripts/fix_clear_organic_colliders.py). These are the sort tier's TRAY targets.
        ("red_onion", "red_onion", True, 0.85, 0.12),
        ("avocado01", "avocado", True, 0.8, 0.17),
        ("whitepackerbottle_a01", "whitepackerbottle_a01", False, 1.0, 0.20),
        ("crabbypenholder", "crabbypenholder", False, 1.0, 0.15),
        ("milkjug_a01", "milkjug_a01", False, 1.0, 0.40),
        ("serving_bowl", "serving_bowl", False, 1.0, 0.30),
        ("utilityjug_a03", "utilityjug_a03", False, 0.4, 0.25),
    )

    # Instances to leave OUT of this variant, by manifest name. The full 16-object RoboLab set
    # is the default (and what the NullRobot oracle exercises); an ARM binding may drop items
    # whose difficulty is incidental rather than intended. The two tall ellipsoids (red_onion
    # 59x59x90, avocado 61x61x92) and the taller orange are near-unpickable by a parallel jaw:
    # a sphere/ellipsoid needs the pads centred to ~1 mm or first contact rolls it away, which
    # tests IK precision, not the identification + long-horizon sequencing this task is for.
    exclude: tuple = ()
    # per-item flat colour override (name -> rgb), bound stronger than the asset's material
    item_colors: dict = field(default_factory=dict)
    contact_offset: float = 0.004  # item speculative contact margin (m)
    item_static_friction: float = 1.1  # produce skin vs rubber gripper pads (see assets())
    item_dynamic_friction: float = 0.95
    # (The G1 sort tier raises these to 2.0 / 1.8 in its own cfg: its palm CAGE holds a fruit by
    # friction against the palm rather than by squeezing it, and at 1.1 a hulled lemon slid out
    # of the cage during the lift-and-roll. That is a property of THAT grasp, not of the produce,
    # so it lives in the binding — the franka tier keeps the pads it was measured with.)
    asset_dir: str = ""

    # derived (filled in __post_init__)
    manifest: tuple = field(default=None, init=False)  # MANIFEST minus `exclude`
    bin_usd: str = field(default="", init=False)
    tray_usd: str = field(default="", init=False)
    item_usds: dict = field(default=None, init=False)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "clear_organic_objects")
        self.bin_usd = str(Path(self.asset_dir) / self.bin_key / f"{self.bin_key}.usd")
        self.tray_usd = str(assets / "fruits_on_plate" / self.tray_key
                            / f"{self.tray_key}.usd")
        self.manifest = tuple(m for m in self.MANIFEST if m[0] not in self.exclude)
        names = {m[0] for m in self.MANIFEST}
        for listed in (*self.sort_to_bin, *self.sort_to_tray,
                       *(entry[0] for entry in self.fixed_layout)):
            if listed not in names:
                raise ValueError(f"unknown manifest instance {listed!r}")
        if self.sort_to_tray and self.tray_pos is None:
            raise ValueError("sort_to_tray set but tray_pos is None")
        if not any(m[2] for m in self.manifest):
            raise ValueError(f"exclude={self.exclude} leaves no organics to clear")
        keys = {k for _n, k, _o, _s, _m in self.manifest}
        self.item_usds = {k: str(Path(self.asset_dir) / k / f"{k}.usd") for k in keys}
        # Per-asset half heights from the vendor-time extents (available to layouts/probes).
        # NOTE these assets are BASE-origin (bbox min z = 0), so reset places the ORIGIN at
        # z0 + drop_lift directly; adding half heights suspends items mid-air and the tall
        # ellipsoids topple on landing.
        import json as _json

        extents_path = Path(self.asset_dir) / "extents.json"
        self.item_half_z = {}
        if extents_path.is_file():
            extents = _json.loads(extents_path.read_text())
            for name, key, _o, scale, _m in self.manifest:
                bbox = extents.get(key, {}).get("bbox_m")
                self.item_half_z[name] = (bbox[2] / 2 * scale) if bbox else 0.03
        else:
            self.item_half_z = {m[0]: 0.03 for m in self.manifest}
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        if preset.get("local"):
            self.workbench_usd = self.workbench_usd or str(
                assets / preset["usd"][0] / preset["usd"][1])
        else:
            self.workbench_usd = self.workbench_usd or str(
                assets.parents[1] / "assembly" / "assets" / "props"
                / preset["usd"][0] / preset["usd"][1])


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("clear_organic_objects")
class ClearOrganicObjectsScene(BaseScene):
    cfg: ClearOrganicObjectsSceneCfg

    def __init__(self, cfg: ClearOrganicObjectsSceneCfg | None = None) -> None:
        super().__init__(cfg or ClearOrganicObjectsSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, the vendored work table, the kinematic bin, and the 16 scattered
        items (11 organics + 5 distractors). Colliders are the assets' AUTHORED ones — the bin
        keeps its concave mesh collider so items fall INSIDE, not onto a hull."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        need = [c.bin_usd, *c.item_usds.values()]
        for usd in need:
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — vendor the clear_organics assets first "
                    f"(python scripts/vendor_clear_organic_objects_assets.py)")
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        z0 = c.surface_z
        table_z = z0 - preset["top_offset"]
        ground_z = z0 - preset["height"]
        s = preset["scale"]
        table_spawn = sim_utils.UsdFileCfg(
            usd_path=c.workbench_usd,
            scale=(s, s * c.table_depth_scale, s * preset.get("z_scale", 1.0)))
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        bin_half = math.radians(c.bin_yaw_deg) / 2

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
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(wx + preset.get("origin_offset", (0.0, 0.0))[0],
                         wy + preset.get("origin_offset", (0.0, 0.0))[1], table_z),
                    rot=preset["orient"]),
                spawn=table_spawn,
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.bin_usd,
                    scale=tuple(c.bin_scale),
                    # keep the AUTHORED mesh collider (concave — the crate interior); only
                    # arm the rigid body. Kinematic -> a fixed receptacle that never drifts.
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=c.bin_kinematic,
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bin_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + c.bin_pos[0], wy + c.bin_pos[1], z0),
                    rot=(math.cos(bin_half), 0.0, 0.0, math.sin(bin_half))),
            ),
        }
        if preset.get("collision_slab"):
            slab_x, slab_y, slab_t = preset["collision_slab"]
            out["table_slab"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TableSlab",
                spawn=sim_utils.CuboidCfg(
                    size=(slab_x * c.table_depth_scale, slab_y, slab_t),
                    visible=False,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, z0 - slab_t / 2.0)),
            )
        if c.tray_pos is not None:
            # The clutter destination: a flat plate, deliberately WALL-LESS — placement on it
            # cannot wedge or penetrate, unlike a second crate. Kinematic like the bin.
            # Spawn offset so tray_pos is the plate's GEOMETRIC centre (the prim origin is
            # off-centre; the fruits_on_plate port established this correction).
            tcx = (c.tray_min[0] + c.tray_max[0]) / 2 * c.tray_scale
            tcy = (c.tray_min[1] + c.tray_max[1]) / 2 * c.tray_scale
            out["tray"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.tray_usd,
                    scale=(c.tray_scale,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + c.tray_pos[0] - tcx, wy + c.tray_pos[1] - tcy, z0)),
            )
        if c.riser is not None:
            rx, ry, rz_top, rsx, rsy = c.riser
            riser_h = rz_top - ground_z
            out["riser"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Riser",
                spawn=sim_utils.CuboidCfg(
                    size=(rsx, rsy, riser_h),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 0.26, 0.28), roughness=0.8),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(rx, ry, ground_z + riser_h / 2)),
            )
        for i, (name, key, _org, scale, mass) in enumerate(c.manifest):
            # BUILD-time spawn poses use their own wide 4 x 4 grid (0.18 m pitch), NOT the
            # task's scatter grid: reset() rewrites every pose anyway, and cramming 16
            # colliders into a small scatter footprint at build interlocked them badly
            # enough that the depenetration ejected the tall items metres off the table
            # even after reset had teleported them home.
            sx = ((i % 4) - 1.5) * 0.18
            sy = ((i // 4) - 1.5) * 0.18
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.item_usds[key],
                    scale=(scale,) * 3,
                    visual_material=(sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(c.item_colors[name]), roughness=0.55)
                        if name in c.item_colors else None),
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
                    pos=(wx + sx, wy + sy, z0 + c.drop_lift + 0.02)),
            )
        return out

    def _slot_xy(self, i: int) -> tuple[float, float]:
        """Table-relative xy of scatter slot `i` on a grid centred at `scatter_center`."""
        c = self.cfg
        n = len(c.manifest)
        cols = c.scatter_cols
        rows = math.ceil(n / cols)
        r, col = divmod(i, cols)
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, resolve organic/distractor index sets, precompute the bin interior box
        (bin body frame), and allocate the per-episode present mask."""
        super().bind(env)
        c = self.cfg
        dev = env.device
        self.bin: RigidObject = env.iscene["bin"]
        self.tray: RigidObject | None = (
            env.iscene["tray"] if c.tray_pos is not None else None)
        self.items: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _k, _o, _s, _m in c.manifest}
        self.names = [name for name, _k, _o, _s, _m in c.manifest]
        self.env_origins = env.iscene.env_origins
        self._organic = torch.tensor([o for _n, _k, o, _s, _m in c.manifest],
                                     dtype=torch.bool, device=dev)
        self._org_idx = torch.nonzero(self._organic, as_tuple=False).flatten()
        self._dis_idx = torch.nonzero(~self._organic, as_tuple=False).flatten()
        # bin interior box in the bin's body frame (origin at the bin base, centre)
        hx = c.bin_bbox[0] / 2 * c.bin_scale[0] * c.wall_frac
        hy = c.bin_bbox[1] / 2 * c.bin_scale[1] * c.wall_frac
        self._bin_inner = torch.tensor([hx, hy], device=dev)
        self._bin_floor = c.floor_local_z
        self._bin_rim = c.bin_bbox[2] * c.bin_scale[2] * c.rim_frac + c.bin_rim_stack
        # present[e, i]: item i participates (distractors always present; organics maybe sampled)
        self.present = torch.ones(env.num_envs, len(c.manifest), dtype=torch.bool, device=dev)
        self._friction_written = False
        # Sort-variant targets and latched stage flags: one stage per required placement,
        # plus all-done (the push_shapes latching pattern — partial credit survives a later
        # mistake while success() stays live).
        if c.sort_to_bin or c.sort_to_tray:
            # Only targets actually IN this variant's manifest (a probe may exclude items).
            self.bin_targets = [n for n in c.sort_to_bin if n in self.names]
            self.tray_targets = [n for n in c.sort_to_tray if n in self.names]
        else:
            self.bin_targets = [n for n, _k, o, _s, _m in c.manifest if o]
            self.tray_targets = []
        self.STAGE_NAMES = ([f"bin:{n}" for n in self.bin_targets]
                            + [f"tray:{n}" for n in self.tray_targets] + ["all_done"])
        self._flags = torch.zeros(env.num_envs, len(self.STAGE_NAMES),
                                  dtype=torch.bool, device=dev)
        if self.tray is not None:
            half = (torch.tensor(c.tray_max[:2], device=dev)
                    - torch.tensor(c.tray_min[:2], device=dev)) / 2 * c.tray_scale
            self._tray_radius = float(half.mean()) * c.tray_margin_frac
            self._tray_center_local = (
                torch.tensor([(c.tray_min[0] + c.tray_max[0]) / 2,
                              (c.tray_min[1] + c.tray_max[1]) / 2,
                              0.0], device=dev) * c.tray_scale)
            self._tray_top = c.tray_max[2] * c.tray_scale

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: bin at its (kinematic) pose; optionally sample the present organic
        subset; scatter present items on the (optionally shuffled) grid with xy jitter + free
        yaw at a small lift above the table (settled by the following steps); park absent
        organics in the ground depot."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        wx, wy = c.workbench_pos
        z0 = c.surface_z

        # --- produce friction (once): PhysX shape materials, CPU tensors are the view's -----
        # contract. `UsdFileCfg` has no `physics_material` field, so the only way to author
        # this for USD-spawned items is the raw view. Without it the items keep PhysX's
        # default mu (~0.5) and a 60-75 mm smooth sphere in an ~80 mm parallel jaw has too
        # little friction to hold: measured, lemons and squat fruit were picked reliably while
        # every large round fruit (orange, pomegranate, pumpkin, onion, lime) slipped on the
        # lift or mid-carry. Real fruit skin against rubber pads is mu ~0.8-1.2.
        if not self._friction_written:
            # NOTE: authoring friction on the BIN as well was tried and REVERTED — it did not
            # stop round produce rolling after landing, and the run measuring it scored worse
            # (45 vs 68). Items only.
            for body in self.items.values():
                view = body.root_physx_view
                mp = view.get_material_properties().clone()  # (N, shapes, 3)
                mp[..., 0] = c.item_static_friction
                mp[..., 1] = c.item_dynamic_friction
                mp[..., 2] = 0.0
                view.set_material_properties(mp, torch.arange(view.count, device="cpu"))
            self._friction_written = True

        # --- present mask: distractors always in; organics optionally subset-sampled ---
        self.present[env_ids] = True
        if c.subset_sample:
            n_org = int(self._organic.sum())
            hi = n_org if c.max_organics is None else min(c.max_organics, n_org)
            lo = min(c.min_organics, hi)
            k = torch.randint(lo, hi + 1, (m,), device=dev)
            rank = torch.rand(m, n_org, device=dev).argsort(dim=1).argsort(dim=1)
            self.present[env_ids.unsqueeze(1), self._org_idx.unsqueeze(0)] = rank < k.unsqueeze(1)

        # --- bin: kinematic pose at bin_pos (also covers a dynamic bin shoved last episode) ---
        bin_half = math.radians(c.bin_yaw_deg) / 2
        broot = torch.zeros(m, 13, device=dev)
        broot[:, 0] = wx + c.bin_pos[0]
        broot[:, 1] = wy + c.bin_pos[1]
        broot[:, 2] = z0
        broot[:, 3] = math.cos(bin_half)
        broot[:, 6] = math.sin(bin_half)
        broot[:, 0:3] += origin
        self.bin.write_root_state_to_sim(broot, env_ids)
        if self.tray is not None:
            troot = torch.zeros(m, 13, device=dev)
            troot[:, 0] = wx + c.tray_pos[0] - float(self._tray_center_local[0])
            troot[:, 1] = wy + c.tray_pos[1] - float(self._tray_center_local[1])
            troot[:, 2] = z0
            troot[:, 3] = 1.0
            troot[:, 0:3] += origin
            self.tray.write_root_state_to_sim(troot, env_ids)
        self._flags[env_ids] = False

        # --- items: grid slot (optionally permuted) + jitter + free yaw; absent -> depot ---
        n = len(c.manifest)
        if c.shuffle_slots:
            # Shuffle WITHIN groups (organics among the organic slots, clutter among the
            # clutter slots), not across all slots. A free permutation let produce spawn in a
            # clutter slot adjacent to the open serving bowl and roll INSIDE it — a fruit
            # nested in the bowl cannot be reached by a top-down pinch at all (watched on
            # video 2026-08-26), which is a degenerate case rather than task difficulty.
            # Grouped shuffling keeps per-episode variety without creating it.
            perm = torch.arange(n, device=dev).expand(m, n).clone()
            for group in (self._org_idx, self._dis_idx):
                order = torch.rand(m, len(group), device=dev).argsort(dim=1)
                perm[:, group] = group[order]
        else:
            perm = torch.arange(n, device=dev).expand(m, n)
        slots = torch.tensor([self._slot_xy(i) for i in range(n)], device=dev)  # (n, 2)
        fixed = {entry[0]: entry[1:] for entry in c.fixed_layout}
        yaw_amp = math.radians(c.reset_yaw_deg)
        for i, (name, _k, _org, _s, _mass) in enumerate(c.manifest):
            st = torch.zeros(m, 13, device=dev)
            z_lift = 0.0
            if name in fixed:
                fx, fy, fyaw = fixed[name][:3]
                z_lift = fixed[name][3] if len(fixed[name]) > 3 else 0.0
                st[:, 0] = wx + fx
                st[:, 1] = wy + fy
            else:
                st[:, 0] = wx + slots[perm[:, i], 0]
                st[:, 1] = wy + slots[perm[:, i], 1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = z0 + c.drop_lift + z_lift
            yaw_center = (math.radians(fixed[name][2]) if name in fixed
                          else math.radians(c.reset_yaw_center_deg))
            yaw = yaw_center + (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            # HALF the angle — a z-yaw quaternion is (cos(yaw/2), 0, 0, sin(yaw/2)). This wrote
            # (cos yaw, 0, 0, sin yaw), which silently DOUBLED every commanded orientation. Free
            # +/-180 hid it (a doubled uniform circle is still a uniform circle) and it only bit
            # once a binding pinned the yaw: `reset_yaw_center_deg=90` presented items at 180 deg,
            # putting a lemon's 50 mm narrow axis at azimuth 90 rather than the 180 the G1 binding
            # was steering for. Found while porting the same reset into the fruits_on_plate scene,
            # by tabulating the settled OBB spans instead of trusting the commanded pose.
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            # Optional 5th field of a fixed_layout entry: ROLL (deg) about the item's own x
            # axis, applied before the yaw, so an item authored upright (avocado, onion) can be
            # laid on its side. q = q_yaw * q_roll.
            if name in fixed and len(fixed[name]) > 4 and fixed[name][4]:
                r2 = math.radians(fixed[name][4]) / 2
                cr, sr = math.cos(r2), math.sin(r2)
                cy, sy = st[:, 3].clone(), st[:, 6].clone()
                st[:, 3] = cy * cr          # w
                st[:, 4] = cy * sr          # x
                st[:, 5] = sy * sr          # y
                st[:, 6] = sy * cr          # z
            # absent organics -> off-camera ground depot (below the surface, on the floor)
            absent = ~self.present[env_ids, i]
            if absent.any():
                st[absent, 0] = wx + 1.2 + 0.16 * (i % 3)
                st[absent, 1] = wy + 1.2 + 0.16 * (i // 3)
                st[absent, 2] = z0 - c.TABLES[c.table]["height"] + 0.05
            st[:, 0:3] += origin
            self.items[name].write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bin": self.bin.data.root_state_w[env_ids].clone(),
            "items": {n: b.data.root_state_w[env_ids].clone() for n, b in self.items.items()},
            "present": self.present[env_ids].clone(),
            "flags": self._flags[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        for n, b in self.items.items():
            b.write_root_state_to_sim(state["items"][n], env_ids)
        self.present[env_ids] = state["present"]
        if "flags" in state:
            self._flags[env_ids] = state["flags"].to(self._flags.device)

    # ----- description ---------------------------------------------------------------------------
    #: manifest instance name -> the words used in `describe()` (the agent reads this text).
    PROSE: ClassVar[dict[str, str]] = {
        "lemon_01": "a lemon", "lemon_02": "a small lemon", "lime01": "a lime",
        "lime_a": "a lime", "lime_b": "another lime",
        "lemon_a": "a lemon", "lemon_b": "another lemon",  # the sort tier's own lemons
        "lime01_01": "a second lime", "orange_01": "an orange", "orange_02": "an orange",
        "pomegranate01": "a pomegranate", "pumpkinlarge": "a pumpkin",
        "pumpkinsmall": "a small pumpkin", "red_onion": "a red onion", "avocado01": "an avocado",
        "whitepackerbottle_a01": "a white plastic bottle", "crabbypenholder": "a crab-shaped pen holder",
        "milkjug_a01": "a milk jug", "serving_bowl": "a serving bowl",
        "utilityjug_a03": "a tall utility jug",
    }

    def _prose(self, name: str) -> str:
        """Human wording for a manifest item; falls back to the item name so describe() can never crash."""
        return self.PROSE.get(name, name.replace("_", " "))

    def describe(self) -> str:
        # Built from the LIVE manifest, so a variant that excludes items describes itself
        # honestly instead of promising produce that is not on the table.
        c = self.cfg
        if c.sort_to_bin or c.sort_to_tray:
            join = lambda xs: (", ".join(xs[:-1]) + (", and " + xs[-1] if len(xs) > 1 else xs[0]))  # noqa: E731
            to_bin = [self._prose(n) for n in c.sort_to_bin]
            to_tray = [self._prose(n) for n in c.sort_to_tray]
            return (
                "A work table holds a colourful spread of fruits, vegetables and household "
                "items, with an open blue plastic crate on one side and a round serving tray "
                "on the other.\n"
                "Goal: bus the table, SORTED. Move every fruit and vegetable that sits in the "
                f"front working row — {join(to_bin)} — into the blue crate, and move the "
                f"non-food items from that row — {join(to_tray)} — onto the round tray. "
                "Items further back on the table are part of the scene and stay where they "
                "are. The job is done when each listed item rests in its correct "
                "destination and nothing that is not food has been put in the crate."
            )
        org = [self._prose(n) for n, _k, o, _s, _m in self.cfg.manifest if o]
        dis = [self._prose(n) for n, _k, o, _s, _m in self.cfg.manifest if not o]
        join = lambda xs: ", ".join(xs[:-1]) + (", and " + xs[-1] if len(xs) > 1 else xs[0])  # noqa: E731
        bx = self.cfg.bin_bbox[0] * self.cfg.bin_scale[0] * 100
        by = self.cfg.bin_bbox[1] * self.cfg.bin_scale[1] * 100
        bz = self.cfg.bin_bbox[2] * self.cfg.bin_scale[2] * 100
        return (
            f"A cluttered work table holds a mix of items: organic fruits and vegetables — "
            f"{join(org)} — scattered among non-food clutter: {join(dis)}. An open blue plastic "
            f"bin (about {bx:.0f} x {by:.0f} cm, {bz:.0f} cm deep) sits to one side of the table.\n"
            "Goal: identify every ORGANIC item — the fruits and vegetables — and place each one "
            "into the bin, leaving all the non-food clutter where it is. An item counts only when "
            "it is resting inside the bin; the job is done when every fruit and vegetable is in "
            "the bin and no non-food item has been put in."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def _in_bin(self) -> torch.Tensor:
        """(N, n_items) bool: each item's origin inside the bin's interior box, computed in the
        bin's body frame (bin motion is irrelevant)."""
        from isaaclab.utils.math import quat_apply_inverse

        bp = self.bin.data.root_pos_w  # (N, 3)
        bq = self.bin.data.root_quat_w  # (N, 4)
        cols = []
        for name in self.names:
            loc = quat_apply_inverse(bq, self.items[name].data.root_pos_w - bp)
            inside_xy = (loc[:, :2].abs() <= self._bin_inner).all(dim=-1)
            inside_z = (loc[:, 2] >= self._bin_floor) & (loc[:, 2] <= self._bin_rim)
            cols.append(inside_xy & inside_z)
        return torch.stack(cols, dim=1)

    def _speed(self) -> torch.Tensor:
        """(N, n_items) |lin vel| per item."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                            for b in self.items.values()], dim=1)

    def cleared(self) -> torch.Tensor:
        """(N, n_items) bool: item in the bin AND settled."""
        return self._in_bin() & (self._speed() < self.cfg.settle_speed)

    def _on_tray(self) -> torch.Tensor:
        """(N, n_items) bool: each item's origin over the tray's counted disc, resting within
        the stack band above the plate top. Judged in the tray's frame like the bin."""
        from isaaclab.utils.math import quat_apply_inverse

        if self.tray is None:
            return torch.zeros_like(self.present)
        tp = self.tray.data.root_pos_w
        tq = self.tray.data.root_quat_w
        cols = []
        for name in self.names:
            loc = quat_apply_inverse(tq, self.items[name].data.root_pos_w - tp)
            loc = loc - self._tray_center_local
            inside_xy = loc[:, :2].norm(dim=-1) <= self._tray_radius
            inside_z = (loc[:, 2] >= 0.0) & (loc[:, 2] <= self._tray_top + self.cfg.tray_stack)
            cols.append(inside_xy & inside_z)
        return torch.stack(cols, dim=1)

    def placed_on_tray(self) -> torch.Tensor:
        """(N, n_items) bool: on the tray AND settled."""
        return self._on_tray() & (self._speed() < self.cfg.settle_speed)

    # ----- sort-variant staging (latched, the push_shapes pattern) -----------------------
    def _stage_live(self) -> torch.Tensor:
        idx = {n: i for i, n in enumerate(self.names)}
        cleared = self.cleared()
        on_tray = self.placed_on_tray()
        cols = [cleared[:, idx[n]] for n in self.bin_targets]
        cols += [on_tray[:, idx[n]] for n in self.tray_targets]
        done = torch.stack(cols, dim=1).all(dim=1) if cols else torch.ones(
            self.env.num_envs, dtype=torch.bool, device=self.env.device)
        cols.append(done & ~self.distractors_in_bin().any(dim=1))
        return torch.stack(cols, dim=1)

    def post_step(self) -> None:
        self._flags |= self._stage_live()

    def stage_flags(self) -> torch.Tensor:
        return self._flags.clone()

    def organics_cleared(self) -> torch.Tensor:
        """(N, n_organics) bool: each organic cleared (bin+settled), in organic order."""
        return self.cleared()[:, self._org_idx]

    def distractors_in_bin(self) -> torch.Tensor:
        """(N, n_distractors) bool: each distractor currently inside the bin (misplaced)."""
        return self._in_bin()[:, self._dis_idx]

    def organics_present(self) -> torch.Tensor:
        """(N, n_organics) bool: organic sampled present this episode."""
        return self.present[:, self._org_idx]

    def score(self) -> torch.Tensor:
        """(N,) int 0..100.

        Sort variant (explicit targets): even partial credit over the LATCHED per-placement
        stages up to 90, minus 10 per distractor currently in the bin; 100 only while every
        target is placed and the bin holds no distractor (live).
        Legacy variant: per-present-organic progress, as shipped.
        """
        misplaced = self.distractors_in_bin().sum(dim=1)
        if self.tray_targets or self.cfg.sort_to_bin:
            n_stage = len(self.STAGE_NAMES) - 1
            frac = self._flags[:, :n_stage].sum(dim=1).float() / max(n_stage, 1)
            base = (90.0 * frac).round().to(torch.long)
            base = (base - 10 * misplaced).clamp(min=0, max=90)
            return torch.where(self.success(), torch.full_like(base, 100), base)
        pres = self.organics_present()
        done = self.organics_cleared() & pres
        n_pres = pres.sum(dim=1).clamp(min=1)
        frac = done.sum(dim=1).float() / n_pres.float()
        base = (90.0 * frac).round().to(torch.long)
        base = (base - 10 * misplaced).clamp(min=0, max=90)
        complete = (done | ~pres).all(dim=1) & (misplaced == 0)
        return torch.where(complete, torch.full_like(base, 100), base)

    def success(self) -> torch.Tensor:
        """(N,) bool, measured LIVE: every required placement holds right now AND no
        distractor is in the bin. (In the sort variant `sort_to_bin`/`sort_to_tray` define
        the required placements; otherwise every present organic must be in the bin.)"""
        if self.tray_targets or self.cfg.sort_to_bin:
            idx = {n: i for i, n in enumerate(self.names)}
            cleared = self.cleared()
            on_tray = self.placed_on_tray()
            checks = [cleared[:, idx[n]] for n in self.bin_targets]
            checks += [on_tray[:, idx[n]] for n in self.tray_targets]
            done = torch.stack(checks, dim=1).all(dim=1)
            return done & ~self.distractors_in_bin().any(dim=1)
        pres = self.organics_present()
        organics_done = (self.organics_cleared() | ~pres).all(dim=1)
        return organics_done & ~self.distractors_in_bin().any(dim=1)
