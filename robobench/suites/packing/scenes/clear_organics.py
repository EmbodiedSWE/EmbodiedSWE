"""ClearOrganicsScene — sort the organic produce out of a cluttered table into the bin.

A port of NVlabs/RoboLab's `ClearOrganicObjectsTask`
(github.com/NVlabs/RoboLab/blob/main/robolab/tasks/benchmark/clutter_organic_objects_task.py):
a work table strewn with organic fruits & vegetables mixed among non-food clutter (bottles,
jugs, a serving bowl, a pen holder), and an open bin off to one side. **Goal (carried here,
no task layer): identify every organic item — the fruits and vegetables — and place each into
the bin, leaving the non-food clutter on the table.**

The objects are the RoboLab assets themselves, vendored by
`scripts/vendor_clear_organics_assets.py` into `suites/packing/assets/clear_organics/` (real
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
class ClearOrganicsSceneCfg(BaseCfg):
    """Config for `ClearOrganicsScene`. Plain fields (the suite convention); a variant is a copy
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
    shuffle_slots: bool = True  # per-episode random item->scatter-slot permutation
    subset_sample: bool = False  # sample a subset of organics present (demo/oracle: False)
    min_organics: int = 4  # per-episode lower bound of sampled organic count
    drop_lift: float = 0.03  # spawn clearance above the table before settling (m)

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
    }

    # --- structure (bin bbox measured at vendor time; see assets/clear_organics/extents.json) --
    bin_key: ClassVar[str] = "container_f24"
    bin_bbox: ClassVar[tuple] = (1.15838, 0.7958, 0.66998)  # UNSCALED outer bbox (m)
    # manifest: (instance, asset_key, is_organic, spawn_scale, mass_kg). The 11 organics are the
    # RoboLab targets; the 5 distractors the RoboLab clutter. lime01 + lime01_01 both spawn the
    # one vendored lime asset (RoboLab uses two instances of it).
    MANIFEST: ClassVar[tuple] = (
        ("lemon_01", "lemon1", True, 1.0, 0.10),
        ("lemon_02", "lemon2", True, 1.0, 0.08),
        ("lime01", "lime", True, 1.0, 0.10),
        ("lime01_01", "lime", True, 1.0, 0.10),
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
        #     franka reference solve fell from 3/5 organics cleared to 0/5. 64 mm is the
        #     compromise that keeps both realism and a graspable margin.
        # Resulting spans: pomegranate 64 > orange 63 > pumpkinlarge 62 > lime 61 ~ avocado 61
        # > onion 59 > pumpkinsmall 55; the lemons are pinched across their 50/40 mm short axis.
        ("pomegranate01", "pomegranate", True, 0.56, 0.26),
        ("pumpkinlarge", "pumpkinlarge", True, 0.70, 0.20),
        ("pumpkinsmall", "pumpkinsmall", True, 0.72, 0.13),
        ("red_onion", "red_onion", True, 1.0, 0.12),
        ("avocado01", "avocado", True, 1.0, 0.17),
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
    contact_offset: float = 0.004  # item speculative contact margin (m)
    item_static_friction: float = 1.1  # produce skin vs rubber gripper pads (see assets())
    item_dynamic_friction: float = 0.95
    asset_dir: str = ""

    # derived (filled in __post_init__)
    manifest: tuple = field(default=None, init=False)  # MANIFEST minus `exclude`
    bin_usd: str = field(default="", init=False)
    item_usds: dict = field(default=None, init=False)

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "clear_organics")
        self.bin_usd = str(Path(self.asset_dir) / self.bin_key / f"{self.bin_key}.usd")
        self.manifest = tuple(m for m in self.MANIFEST if m[0] not in self.exclude)
        if not any(m[2] for m in self.manifest):
            raise ValueError(f"exclude={self.exclude} leaves no organics to clear")
        keys = {k for _n, k, _o, _s, _m in self.manifest}
        self.item_usds = {k: str(Path(self.asset_dir) / k / f"{k}.usd") for k in keys}
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(
            assets.parents[1] / "assembly" / "assets" / "props" / preset["usd"][0] / preset["usd"][1])


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("clear_organics")
class ClearOrganicsScene(BaseScene):
    cfg: ClearOrganicsSceneCfg

    def __init__(self, cfg: ClearOrganicsSceneCfg | None = None) -> None:
        super().__init__(cfg or ClearOrganicsSceneCfg())

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
                    f"(python scripts/vendor_clear_organics_assets.py)")
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
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z), rot=preset["orient"]),
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
        for i, (name, key, _org, scale, mass) in enumerate(c.manifest):
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
                    pos=(wx + sx, wy + sy, z0 + c.drop_lift + 0.03 * (i % 2))),
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
            k = torch.randint(c.min_organics, n_org + 1, (m,), device=dev)
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
        yaw_amp = math.radians(c.reset_yaw_deg)
        for i, (name, _k, _org, _s, _mass) in enumerate(c.manifest):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = wx + slots[perm[:, i], 0]
            st[:, 1] = wy + slots[perm[:, i], 1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 2] = z0 + c.drop_lift + 0.03 * (i % 2)
            h = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(h)
            st[:, 6] = torch.sin(h)
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
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.bin.write_root_state_to_sim(state["bin"], env_ids)
        for n, b in self.items.items():
            b.write_root_state_to_sim(state["items"][n], env_ids)
        self.present[env_ids] = state["present"]

    # ----- description ---------------------------------------------------------------------------
    #: manifest instance name -> the words used in `describe()` (the agent reads this text).
    PROSE: ClassVar[dict[str, str]] = {
        "lemon_01": "a lemon", "lemon_02": "a small lemon", "lime01": "a lime",
        "lime01_01": "a second lime", "orange_01": "an orange", "orange_02": "an orange",
        "pomegranate01": "a pomegranate", "pumpkinlarge": "a pumpkin",
        "pumpkinsmall": "a small pumpkin", "red_onion": "a red onion", "avocado01": "an avocado",
        "whitepackerbottle_a01": "a white plastic bottle", "crabbypenholder": "a crab-shaped pen holder",
        "milkjug_a01": "a milk jug", "serving_bowl": "a serving bowl",
        "utilityjug_a03": "a tall utility jug",
    }

    def describe(self) -> str:
        # Built from the LIVE manifest, so a variant that excludes items describes itself
        # honestly instead of promising produce that is not on the table.
        org = [self.PROSE[n] for n, _k, o, _s, _m in self.cfg.manifest if o]
        dis = [self.PROSE[n] for n, _k, o, _s, _m in self.cfg.manifest if not o]
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
        """(N,) int 0..100: per-present-organic progress up to 90, minus 10 per distractor in
        the bin, 100 once every present organic is cleared and no distractor is in the bin."""
        pres = self.organics_present()
        done = self.organics_cleared() & pres
        n_pres = pres.sum(dim=1).clamp(min=1)
        frac = done.sum(dim=1).float() / n_pres.float()
        base = (90.0 * frac).round().to(torch.long)
        misplaced = self.distractors_in_bin().sum(dim=1)
        base = (base - 10 * misplaced).clamp(min=0, max=90)
        complete = (done | ~pres).all(dim=1) & (misplaced == 0)
        return torch.where(complete, torch.full_like(base, 100), base)

    def success(self) -> torch.Tensor:
        """(N,) bool: every present organic cleared into the bin AND no distractor in the bin
        (scene-level success; the NullRobot oracle's target). The 'gripper detached' clause is an
        embodiment clause checked at the binding/harness layer, not here."""
        pres = self.organics_present()
        organics_done = (self.organics_cleared() | ~pres).all(dim=1)
        return organics_done & ~self.distractors_in_bin().any(dim=1)
