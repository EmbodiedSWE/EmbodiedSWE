"""SpatulaFlipServeScene — flip a bread slice in a fry pan, then serve it onto a plate (port).

The object world for the SimToolReal / DexToolBench "spatula flip & serve" port: a BREAD
slice lies in a FRY PAN, a bamboo PLATE waits beside it, a SPATULA rests on the bench.
**Goal (carried here, no task layer, per `cfg.goal` — the source curriculum):
v0 `serve` = slide the blade under the bread, carry it on the blade and set it down flat
on the plate; v1 `flip` = turn the bread over in place (it lands back in the pan);
v2 `flip_serve` = the full chain, flip first, then serve.**

This is the ported set's handheld-TOOL slot and its manipulation class is NON-PREHENSILE
payload control: the bread is never grasped — it rides a tool the robot holds, so success
is managing an unsecured cargo through wedging, a commit-point flip, and a friction-only
carry. TOOL-ONLY rule: finger/gripper contact with the bread disqualifies the episode —
that is an EMBODIMENT clause, checked at the robot-binding/harness layer, deliberately
not here (the scene is robot-agnostic — the pen-holder/stacking-toy return-to-origin precedent).

Judged by OUTCOME (the one flagged fidelity deviation from the source, which scores 6D
tool-pose trajectory following because it benchmarks policies; the deep survey showed the
source has NO payload — all payload physics here is new work). Staged flags, latched in
`post_step` (the microwave pattern), all geometric checks in the relevant BODY frame
(the pen-holder lesson):
  - `tool_lifted`  — the spatula blade is above the surface by `lift_gate` (the source's
                     own >5 cm lift-gate idea, kept);
  - `blade_under`  — the bread rides the blade (bread pose in the BLADE frame: inside the
                     blade footprint, bottom on the blade top, axes aligned) while still
                     down at pan-floor level — the wedge;
  - `flipped`      — the bread's body up-axis is inverted by >= `flip_min_deg` about a
                     horizontal axis AND it rests flat in the pan, settled (memoryless
                     orientation test);
  - `loaded`       — the bread rides the blade above `lift_gate` (the friction carry);
  - `served`       — the bread rests flat on the plate, settled, AND it ARRIVED ON THE
                     BLADE: the latch only fires within `arrival_window` steps of the last
                     loaded step (the contact-history clause — teleports, shoves and
                     lobbed tosses from across the bench never load at height, so they
                     never serve).
Score tables per goal (transition rubric, monotone prefix over the latched stages):
  serve       [10 lifted, 30 wedged, 60 loaded, 100 served]
  flip        [10 lifted, 30 wedged, 100 flipped]
  flip_serve  [10 lifted, 25 wedged, 50 flipped, 70 loaded-after-flip, 100 served]
Metrics for the brief: `max_carry_tilt_deg` (the blade's worst tilt while loaded — the
finesse number; the smoke's calibration sweep publishes the tilt budget the carry
tolerates) and `spills` (payload-drop events: the bread at rest on the bare bench after
having been loaded).

Assets are SCANNED PRODUCT MODELS (assets/kitchen, authored by
scripts/author_kitchen_rigs.py — the pc_motherboard visual+invisible-collider pattern;
every number below is measured from the scans, see kitchen_parts.json):
  - spatula (a molded red turner): origin at the BLADE-BOTTOM CENTER, +x toward the
    blade tip, +z up — the scan's inclined blade plane is leveled at authoring so the
    rubric and any oracle work directly in blade coordinates. Colliders: three thin
    tapered blade boxes (the molded blade thickens heel-ward; the tip box stays the thin
    leading edge the wedge needs), a neck box and a handle box on the 21 deg incline.
    The blade carries NO climbing feature — the wedge mechanic lives on the payload's
    rounded edge (GPU rounds 1-3 lesson, unchanged).
  - bread (a scan of a real slice, laid flat, ~6.2 cm across, 2.2 cm thick): collider =
    the slice's own CONVEX HULL (cooked from a face-subsampled proxy) — the scan's
    rounded crust edge IS the chamfer that makes wedging well-posed: any tip contact on
    it has an up-forward normal, so a sliding blade converts advance to lift by
    construction; near-symmetric top/bottom, so a FLIPPED slice re-wedges. Three sizes
    spawn (uniform scales of one scan) and ONE is present per episode (the pen-holder
    parking-depot pattern) — the size randomization axis.
  - fry pan (a 27 cm tri-ply pan, its lid dropped at authoring) and bamboo plate
    (30 cm): KINEMATIC bodies with EXACT triangle-mesh colliders — the concave bowl,
    rim and plate recess collide true to the scan. The pan replaces the flat cutting
    board: its RIM (5.3 cm) walls the payload in, so the wedge becomes a pitched
    over-the-rim entry and the wall itself is the anchor a jab pins the bread against
    (a real pan-corner scoop), and every lift-out must clear the rim.

Per-episode randomization (task-family knobs): bread size (one of three), bread pose in
the pan (xy jitter + yaw), plate position (xy jitter), pan yaw, spatula rest pose (xy
jitter + yaw). `reset()` judges the sampled episode.

Embodied bindings hold the tool through the weld-on-closure grasp contract (the
pc_motherboard pattern, ported verbatim; one site = the spatula's handle): close the
fingers across the handle and the tool welds to the hand, open wide to release. No-op
under robot="null" — the NullRobot smoke drives the tool kinematically instead.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import GraspWeldMixin, SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

GOALS = ("serve", "flip", "flip_serve")


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SpatulaFlipServeSceneCfg(BaseCfg):
    """Config for `SpatulaFlipServeScene`. Friction/mass are the brief's feasibility-spike
    knobs; the smoke's calibration sweep publishes the carry tilt budget they produce.
    Structure constants are MEASURED from the scanned assets (kitchen_parts.json)."""

    # --- tunable: the curriculum knob ---------------------------------------------------------
    goal: str = tunable("flip_serve")  # "serve" (v0) | "flip" (v1) | "flip_serve" (v2)

    # --- tunable: rubric thresholds -----------------------------------------------------------
    lift_gate: float = tunable(0.05)  # blade/payload height above the surface = "lifted"
    # (the source's own >5 cm lift gate, kept)
    flip_min_deg: float = tunable(150.0)  # orientation change about a horizontal axis = flipped
    blade_align_max_deg: float = tunable(30.0)  # bread axis vs blade axis while riding it
    flat_tilt_max_deg: float = tunable(15.0)  # "resting flat" gate (pan floor and plate)
    rest_z_tol: float = tunable(0.012)  # bread bottom within this of the resting surface (m)
    served_xy_frac: float = tunable(0.75)  # bread centre within this fraction of the plate radius
    settle_speed: float = tunable(0.05)  # max |v| when judging a resting bread (m/s)
    spill_settle_steps: int = tunable(12)  # sustained bare-surface rest before one spill
    arrival_window: int = tunable(600)  # served must fire within this many steps of the last
    # loaded step (5 s at 120 Hz — the no-toss/no-shove contact-history clause; covers the
    # lower-and-tip end game, where the payload dips under the lift gate). Sized for THIS
    # plate: the bamboo dish is a slope, and a slice tipped off the blade can slide/settle
    # for 3-4 s before it rests flat (measured) — a real toss still never counts, because
    # a thrown slice was never loaded near the plate at all.
    wedge_low_band: float = tunable(0.03)  # blade_under counts only with the bread bottom
    # within this of the pan floor (the wedge happens IN the pan, not in mid-air)

    # --- tunable: physics (the feasibility-spike knobs) ----------------------------------------
    # Friction pair sized from BOTH ends (round-2 GPU lesson): PhysX combines by AVERAGE, so
    # blade-bread ~ (0.33 static / 0.29 dynamic). Low enough that the wedge SLIPS under the
    # payload instead of sticking to it and bulldozing; high enough that the carry has a real
    # tilt budget: atan(0.33) ~ 18 deg. Bread-pan stays grippier (the pan rig bakes the old
    # board's 0.6/0.55) so the pan anchors the payload while the blade slides beneath — and
    # the pan WALL is now a hard anchor no jab can shove the payload past.
    bread_mass: float = tunable(0.06)
    bread_friction: tuple = tunable((0.5, 0.45))  # static, dynamic (moist crumb)
    blade_friction: tuple = tunable((0.15, 0.12))  # molded slick face — the slippery half
    spatula_mass: float = tunable(0.15)
    # Weld-on-closure grasping (the benchmark's auto-weld contract, the pc_motherboard
    # pattern — the machinery at the end of this scene class): close the fingers across
    # the spatula's handle and the tool welds to the hand; open wide to release. The
    # TOOL-ONLY rule is untouched — the contract's one site is the handle, never the
    # payload. Gripper envs only (no-op under robot="null").
    grasp_weld: bool = tunable(True)
    grasp_weld_dist: float = tunable(0.010)  # pinch-point-to-grip-band engage radius (m)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    bread_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the bread in the pan
    plate_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the plate
    spatula_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the spatula rest pose
    spatula_yaw_deg: float = tunable(15.0)  # uniform +/- yaw jitter of the spatula
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- bread yaw (kills any memorizable layout)
    sample_size: bool = tunable(True)  # per-episode bread-size sampling (demo sets False)

    # --- tunable: placement -----------------------------------------------------------------
    pan_pos: tuple = tunable((-0.16, 0.05))  # pan BOWL centre on the surface
    pan_yaw_deg: float = tunable(90.0)  # pan yaw; at 0 the handle points +x, default +y
    # (away from a robot working at -y — out of the wedge corridor)
    plate_pos: tuple = tunable((0.17, 0.06))  # plate centre (before jitter)
    spatula_pos: tuple = tunable((0.02, -0.20))  # spatula rest (blade-bottom centre)

    # --- info: selectable work surface (the pc_motherboard presets, ported verbatim) ----------
    table: str = info("packing")  # which work surface: "lab_table" | "packing"
    # ("packing" = the microwave_meal look, requested 2026-08-05; Franka bindings pin
    # surface_z=0.0 — the table-mounted-arm-at-ground-level pattern from envs.py)
    surface_z: float | None = info(None)  # table-top height (m); None -> the preset's
    workbench_pos: tuple | None = info(None)  # xy the table sits at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.0, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }

    # --- info: structure (measured from the scans; see kitchen_parts.json) ---------------------
    # spatula: blade footprint + thickness (the rubric's blade-frame constants), handle incline
    blade_l: float = info(0.1168)
    blade_w: float = info(0.0893)
    blade_t: float = info(0.0055)  # molded blade max thickness (tip box is thinner)
    handle_angle_deg: float = info(21.1)
    # fry pan: interior floor disc, rim, bowl/handle extents (bowl centre = body origin)
    pan_floor_top: float = info(0.0035)  # interior floor above the pan's base plane
    pan_r_floor: float = info(0.1095)  # flat interior floor radius
    pan_rim_top: float = info(0.0532)  # rim height above the base plane
    pan_r_rim_in: float = info(0.1304)  # rim inner radius
    pan_r_out: float = info(0.1361)  # bowl outer radius (the bare-surface exclusion)
    # bamboo plate: recess floor + rim
    plate_r: float = info(0.1504)
    plate_floor_top: float = info(0.0126)  # recess floor above the plate's base plane
    plate_rim_top: float = info(0.0476)
    # bread: base scan dims (half-extent / thickness) x the per-family uniform scales
    bread_r0: float = info(0.0308)
    bread_h0: float = info(0.0215)
    # (family name, uniform scale): three sizes, ONE present per episode.
    families: tuple = info((("bread_s", 1.15), ("bread_m", 1.30), ("bread_l", 1.45)))
    # Off-camera ground depot for absent breads (the pen-holder depot analysis: extent well
    # under half of env_spacing 3).
    parking_pos: tuple = info((1.0, 1.0))
    # On-blade z band for the bread bottom in the blade frame: a riding slice rests on the
    # tapered blade top (1.5-5.5 mm); the upper margin absorbs offset slop. The lower bound
    # EXCLUDES a slice the blade merely slid toward while it rests on the pan floor
    # (bottom ~ -0/+0.5 mm in the blade frame) — round 1's false-positive wedge check.
    on_blade_z_band: tuple = info((0.0005, 0.015))
    blade_contact_offset: float = info(0.001)  # below the thin tip box thickness
    bread_contact_offset: float = info(0.001)
    pan_contact_offset: float = info(0.002)
    # DYNAMIC pan: real contact response against a KINEMATICALLY driven tool (a
    # kinematic-vs-kinematic pair generates no contacts, so a pose-pinned tool clips
    # straight through a kinematic pan — the NullRobot smoke's penetration bug). Heavy
    # + heavily damped so jab reactions nudge it millimetres, not across the table.
    pan_dynamic: bool = info(False)
    pan_mass: float = info(2.5)
    # Asset USDs; empty -> the authored rigs committed under `assets/kitchen/`.
    asset_dir: str = info("")
    spatula_usd: str = info("")
    bread_usd: str = info("")
    pan_usd: str = info("")
    plate_usd: str = info("")

    # Derived (filled in __post_init__).
    pan_floor_z: float = field(default=None, init=False)  # bread rest height in the pan
    plate_rest_z: float = field(default=None, init=False)  # bread rest height on the plate

    def __post_init__(self) -> None:
        assert self.goal in GOALS, f"goal must be one of {GOALS}, got {self.goal!r}"
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "kitchen")
        self.spatula_usd = self.spatula_usd or str(Path(self.asset_dir) / "spatula.usd")
        self.bread_usd = self.bread_usd or str(Path(self.asset_dir) / "bread.usd")
        self.pan_usd = self.pan_usd or str(Path(self.asset_dir) / "fry_pan.usd")
        self.plate_usd = self.plate_usd or str(Path(self.asset_dir) / "bamboo_plate.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        # the vendored table props are shared from the assembly suite's assets
        props = Path(__file__).resolve().parents[2] / "assembly" / "assets" / "props"
        self.workbench_usd = self.workbench_usd or str(
            props / preset["usd"][0] / preset["usd"][1])
        self.pan_floor_z = round(self.surface_z + self.pan_floor_top, 4)
        self.plate_rest_z = round(self.surface_z + self.plate_floor_top, 4)

    # per-family bread dims (uniform scale on one scan)
    def bread_r(self, i: int) -> float:
        return self.bread_r0 * self.families[i][1]

    def bread_h(self, i: int) -> float:
        return self.bread_h0 * self.families[i][1]


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("spatula")
class SpatulaFlipServeScene(GraspWeldMixin, BaseScene):
    cfg: SpatulaFlipServeSceneCfg

    def __init__(self, cfg: SpatulaFlipServeSceneCfg | None = None) -> None:
        super().__init__(cfg or SpatulaFlipServeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, optional bench, the kinematic pan + plate (exact trimesh
        colliders), the spatula at rest and the three bread sizes (reset() re-places
        everything; the PhysX-side tuning rides the spawn cfg modifiers — the
        pc_motherboard pattern for shipped USDs)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        for usd in (c.spatula_usd, c.bread_usd, c.pan_usd, c.plate_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — run scripts/author_kitchen_rigs.py to build the "
                    f"kitchen assets"
                )

        # the vendored work table (the pc_motherboard preset pattern): ground drops to
        # the table's foot, the kitchen work sits on its top at surface_z
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd,
                                           scale=(preset["scale"],) * 3)
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z),
                                                        rot=preset["orient"]),
                spawn=table_spawn,
            ),
        }

        half = math.radians(c.pan_yaw_deg) / 2
        # Pan: kinematic by default (the wedge needs an unmovable substrate to push
        # against — and now a wall to pin the payload on). `pan_dynamic` swaps it to a
        # heavy, heavily damped free body so a kinematically driven tool gets real
        # contact response instead of clipping through (see the cfg comment).
        if c.pan_dynamic:
            pan_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                linear_damping=5.0, angular_damping=5.0,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16)
            pan_mass = sim_utils.MassPropertiesCfg(mass=c.pan_mass)
        else:
            pan_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            pan_mass = None
        out["pan"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pan",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.pan_usd,
                rigid_props=pan_props,
                mass_props=pan_mass,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.pan_contact_offset, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.pan_pos[0], c.pan_pos[1], z0),
                rot=(math.cos(half), 0.0, 0.0, math.sin(half))),
        )
        # Plate: kinematic; its POSITION is a reset randomization axis (kinematic bodies
        # take pose writes — the turntable/board precedent; nothing ever needs to move it).
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.plate_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.pan_contact_offset, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.plate_pos[0], c.plate_pos[1], z0)),
        )

        # Spatula: the pen-holder anti-pop armor on the tool (a wedge tip driven a hair
        # into the payload in one 120 Hz step must resolve gently, not eject it).
        out["spatula"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Spatula",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.spatula_usd,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.spatula_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.blade_contact_offset, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.spatula_pos[0], c.spatula_pos[1], z0 + 0.003)),
        )
        # Breads: the payload rides a thin plate under SUSTAINED kinematic-drive contact.
        # PhysX clamps position-correction velocity by maxDepenetrationVelocity — at the
        # anti-pop 0.5 the correction cannot keep up with a lift and the payload settles
        # INSIDE the plate, then shears off on the first lateral move (GPU round 5). The
        # higher cap + more position iterations keep it ON the plate; the convex-hull
        # crust edge tolerates the livelier depenetration. Light damping so a 60 g slice
        # crosses the settle gate promptly.
        for i, (name, scale) in enumerate(c.families):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bread_" + name,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.bread_usd,
                    scale=(scale, scale, scale),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bread_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.5,
                        solver_position_iteration_count=12,
                        linear_damping=0.05,
                        angular_damping=0.05,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.bread_contact_offset, rest_offset=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pan_pos[0], c.pan_pos[1],
                         c.pan_floor_z + c.bread_h(i) / 2 + 0.002 + i * 0.03)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, set the tuned frictions (the shipped rigs carry placeholder
        physics materials; the tunables are authoritative), allocate the presence mask
        and the latched stage/metric tensors."""
        super().bind(env)
        c = self.cfg
        self.spatula: RigidObject = env.iscene["spatula"]
        self.pan: RigidObject = env.iscene["pan"]
        self.plate: RigidObject = env.iscene["plate"]
        self.breads: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _s in c.families}
        self.env_origins = env.iscene.env_origins
        self._set_friction(self.spatula, c.blade_friction)
        for b in self.breads.values():
            self._set_friction(b, c.bread_friction)
        self._alloc(env.num_envs, env.device)
        self._grasp_weld_bind()

    def grasp_sites(self) -> list:
        """One grip band: the spatula's molded HANDLE, spanning the mid 9 cm of the
        stick along its 21 deg incline (spatula-local; centre (-0.216, 0, 0.076)). The
        fingers close across its 21 mm width."""
        a = math.radians(self.cfg.handle_angle_deg)
        cx, cz, half = -0.216, 0.076, 0.045
        dx, dz = -math.cos(a) * half, math.sin(a) * half
        return [("spatula", self.spatula, (cx - dx, 0.0, cz - dz),
                 (cx + dx, 0.0, cz + dz), (0.014, 0.028))]

    def _set_friction(self, asset, friction: tuple) -> None:
        """Overwrite static/dynamic friction on every shape of `asset` (all envs) — the
        motherboard scene's pattern, tuple form."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0] = friction[0]
        mats[..., 1] = friction[1]
        asset.root_physx_view.set_material_properties(
            mats, torch.arange(self.env.num_envs, device="cpu"))

    def _alloc(self, n: int, dev: str) -> None:
        """Mechanic-state tensors (separated from bind so the app-free rubric test can
        allocate them against stub handles — the stacking-toy stubbed-quat test pattern)."""
        c = self.cfg
        # present[e, i]: bread i is THE payload of episode e (one-hot; sampled at reset).
        self._present = torch.zeros(n, len(c.families), dtype=torch.bool, device=dev)
        self._present[:, 1] = True
        self._bread_r = torch.tensor([c.bread_r(i) for i in range(len(c.families))],
                                     device=dev)
        self._bread_h = torch.tensor([c.bread_h(i) for i in range(len(c.families))],
                                     device=dev)
        # latched stage flags
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._wedged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._flipped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded_pf = torch.zeros(n, dtype=torch.bool, device=dev)  # loaded AFTER flipped
        self._served = torch.zeros(n, dtype=torch.bool, device=dev)
        # arrival-on-blade clock + metrics
        self._since_loaded = torch.full((n,), 10**6, dtype=torch.long, device=dev)
        self._max_carry_tilt = torch.zeros(n, device=dev)
        self._spills = torch.zeros(n, dtype=torch.long, device=dev)
        self._lost_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_lost = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the bread size (one-hot), place it flat in the pan with
        jitter + yaw, park the absent breads in the ground depot, jitter the plate
        (kinematic pose write), re-pin the pan, jitter the spatula rest pose, zero every
        latch/metric."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- bread-size sampling (the task-family knob) ---
        n_fam = len(c.families)
        pick = (torch.randint(0, n_fam, (m,), device=dev) if c.sample_size
                else torch.full((m,), 1, dtype=torch.long, device=dev))
        self._present[env_ids] = torch.nn.functional.one_hot(pick, n_fam).bool()

        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- breads: the present one flat in the pan, the rest parked ---
        for i, (name, _s) in enumerate(c.families):
            in_pan = torch.zeros(m, 3, device=dev)
            in_pan[:, 0] = c.pan_pos[0]
            in_pan[:, 1] = c.pan_pos[1]
            in_pan[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bread_jitter
            in_pan[:, 2] = c.pan_floor_z + c.bread_h(i) / 2 + 0.002
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + i * 0.16
            park[:, 1] = c.parking_pos[1]
            park[:, 2] = c.bread_h(i) / 2 + 0.003
            pres = (self._present[env_ids, i]).unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, in_pan, park)
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.breads[name].write_root_state_to_sim(st, env_ids)

        # --- pan: re-pin at its configured pose (kinematic write; yaw is a layout knob) ---
        half_p = math.radians(c.pan_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.pan_pos[0]
        st[:, 1] = c.pan_pos[1]
        st[:, 2] = c.surface_z
        st[:, 3] = math.cos(half_p)
        st[:, 6] = math.sin(half_p)
        st[:, 0:3] += origin
        self.pan.write_root_state_to_sim(st, env_ids)

        # --- plate: kinematic pose write with xy jitter ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.plate_pos[0]
        st[:, 1] = c.plate_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.plate_jitter
        st[:, 2] = c.surface_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.plate.write_root_state_to_sim(st, env_ids)

        # --- spatula: at rest on the bench, blade toward +x, jittered ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.spatula_pos[0]
        st[:, 1] = c.spatula_pos[1]
        st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.spatula_jitter
        st[:, 2] = c.surface_z + 0.003
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spatula_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.spatula.write_root_state_to_sim(st, env_ids)

        # --- zero the latches / clocks / metrics; a fresh episode starts empty-handed ---
        for name in ("_lifted", "_wedged", "_flipped", "_loaded", "_loaded_pf", "_served",
                     "_prev_lost"):
            getattr(self, name)[env_ids] = False
        self._since_loaded[env_ids] = 10**6
        self._max_carry_tilt[env_ids] = 0.0
        self._spills[env_ids] = 0
        self._lost_streak[env_ids] = 0
        self._grasp_weld_release_all(env_ids)

    # ----- kinematics helpers -------------------------------------------------------------------
    def _bread_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), quat (N,P,4), |lin_vel| (N,P)) for all breads, family order."""
        pos = torch.stack([b.data.root_pos_w for b in self.breads.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.breads.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.breads.values()], dim=1)
        return pos, quat, vel

    def _bread_up(self) -> torch.Tensor:
        """(N, P, 3): each bread's body up-axis (spawn-top normal) in world frame."""
        from isaaclab.utils.math import quat_apply

        _p, quat, _v = self._bread_tensors()
        n, p = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * p, 3)
        return quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)

    def _blade_up(self) -> torch.Tensor:
        """(N, 3): the blade plane normal (spatula body +z) in world frame."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.spatula.data.root_quat_w, ez)

    def _bread_in_blade_frame(self) -> torch.Tensor:
        """(N, P, 3): bread centres in the SPATULA body frame (origin = blade-bottom centre).
        The on-blade rubric lives in this frame so a tilted, moving, held blade judges its
        cargo identically to a level one (the pen-holder holder-frame lesson)."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, _q, _v = self._bread_tensors()
        n, p = pos.shape[0], pos.shape[1]
        sq = self.spatula.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        sp = self.spatula.data.root_pos_w[:, None, :]
        return quat_apply_inverse(sq, (pos - sp).reshape(n * p, 3)).reshape(n, p, 3)

    # ----- instantaneous predicates ---------------------------------------------------------------
    def tool_lifted_now(self) -> torch.Tensor:
        """(N,) bool: the blade origin is above the work surface by more than `lift_gate`."""
        c = self.cfg
        z = (self.spatula.data.root_pos_w - self.env_origins)[:, 2]
        return z > c.surface_z + c.lift_gate

    def on_blade(self) -> torch.Tensor:
        """(N, P) bool, blade-frame: bread centre inside the blade footprint, its bottom in
        the on-blade z band, its axis within `blade_align_max_deg` of the blade normal
        (either face — a flipped slice rides the blade too)."""
        c = self.cfg
        loc = self._bread_in_blade_frame()
        in_x = loc[:, :, 0].abs() < c.blade_l / 2
        in_y = loc[:, :, 1].abs() < c.blade_w / 2
        bottom = loc[:, :, 2] - self._bread_h.unsqueeze(0) / 2
        in_z = (bottom > c.on_blade_z_band[0]) & (bottom < c.on_blade_z_band[1])
        up = self._bread_up()
        blade_up = self._blade_up().unsqueeze(1)
        aligned = (up * blade_up).sum(-1).abs() >= math.cos(math.radians(c.blade_align_max_deg))
        return in_x & in_y & in_z & aligned

    def on_pan(self) -> torch.Tensor:
        """(N, P) bool: bread resting flat (either face) on the pan's interior floor."""
        c = self.cfg
        pos, _q, _v = self._bread_tensors()
        pp = self.pan.data.root_pos_w.unsqueeze(1)
        in_r = (pos[:, :, :2] - pp[:, :, :2]).norm(dim=-1) < c.pan_r_floor
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        resting = (z_rel - self._bread_h.unsqueeze(0) / 2 - c.pan_floor_z).abs() < c.rest_z_tol
        flat = self._bread_up()[:, :, 2].abs() >= math.cos(math.radians(c.flat_tilt_max_deg))
        return in_r & resting & flat

    def flipped_now(self) -> torch.Tensor:
        """(N, P) bool: the bread's spawn-top normal is inverted by >= `flip_min_deg` from
        world-up — the memoryless orientation half of the flip check."""
        c = self.cfg
        return self._bread_up()[:, :, 2] <= -math.cos(math.radians(180.0 - c.flip_min_deg))

    def on_plate(self) -> torch.Tensor:
        """(N, P) bool: bread resting flat (either face) in the plate's recess, near centre."""
        c = self.cfg
        pos, _q, _v = self._bread_tensors()
        pp = self.plate.data.root_pos_w.unsqueeze(1)
        near = (pos[:, :, :2] - pp[:, :, :2]).norm(dim=-1) < c.served_xy_frac * c.plate_r
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        resting = (z_rel - self._bread_h.unsqueeze(0) / 2 - c.plate_rest_z).abs() < c.rest_z_tol
        flat = self._bread_up()[:, :, 2].abs() >= math.cos(math.radians(c.flat_tilt_max_deg))
        return near & resting & flat

    def settled(self) -> torch.Tensor:
        """(N, P) bool: bread |lin vel| below `settle_speed`."""
        _p, _q, vel = self._bread_tensors()
        return vel < self.cfg.settle_speed

    def on_bare_surface(self) -> torch.Tensor:
        """(N, P) bool: bread sustained by the bare bench/ground — not the pan, the plate
        or the blade.

        Unlike `on_pan`/`on_plate`, this accepts any orientation. The vertical extent
        therefore includes both the slab half-height and the radius projected onto world z.
        """
        c = self.cfg
        pos, _q, _v = self._bread_tensors()
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        up_z = self._bread_up()[:, :, 2].abs().clamp(0.0, 1.0)
        radius = self._bread_r.unsqueeze(0)
        half_extent_z = (up_z * (self._bread_h.unsqueeze(0) / 2)
                         + torch.sqrt((1.0 - up_z.square()).clamp_min(0.0)) * radius)
        at_surface = (z_rel - half_extent_z - c.surface_z).abs() < c.rest_z_tol

        pp = self.pan.data.root_pos_w.unsqueeze(1)
        over_pan = (pos[:, :, :2] - pp[:, :, :2]).norm(dim=-1) < c.pan_r_out
        pl = self.plate.data.root_pos_w.unsqueeze(1)
        over_plate = (pos[:, :, :2] - pl[:, :, :2]).norm(dim=-1) < c.plate_r
        return at_surface & ~over_pan & ~over_plate & ~self.on_blade() & self.settled()

    def loaded_now(self) -> torch.Tensor:
        """(N,) bool: the present bread rides the blade with its bottom above `lift_gate` —
        the friction carry, judged on the sampled payload."""
        c = self.cfg
        pos, _q, _v = self._bread_tensors()
        bottom = (pos - self.env_origins.unsqueeze(1))[:, :, 2] - self._bread_h.unsqueeze(0) / 2
        high = bottom > c.surface_z + c.lift_gate
        return (self.on_blade() & high & self._present).any(dim=1)

    def served_now(self) -> torch.Tensor:
        """(N,) bool: the present bread rests flat on the plate, settled — with the face
        gate per goal: `serve` (v0, no flip) demands spawn-side up (the source's
        right-side-up clause); `flip_serve` accepts either face (tipping off a blade edge
        makes the final face genuinely ambiguous — the flip stage already proved
        orientation control), the flip itself being enforced through the `flipped` latch."""
        c = self.cfg
        ok = self.on_plate() & self.settled() & self._present
        if c.goal == "serve":
            right_side_up = self._bread_up()[:, :, 2] >= math.cos(
                math.radians(c.flat_tilt_max_deg))
            ok = ok & right_side_up
        return ok.any(dim=1)

    def carry_tilt_deg(self) -> torch.Tensor:
        """(N,) blade tilt from level, degrees — the spill budget variable."""
        up_z = self._blade_up()[:, 2].clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(up_z))

    # ----- mechanics: the latches (run every physics substep) --------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._grasp_weld_step()
        c = self.cfg
        pos, _q, _v = self._bread_tensors()
        z_rel = (pos - self.env_origins.unsqueeze(1))[:, :, 2]
        bottom = z_rel - self._bread_h.unsqueeze(0) / 2
        onb = self.on_blade() & self._present

        self._lifted |= self.tool_lifted_now()
        # the wedge: blade under the bread while the bread is still down at pan level
        self._wedged |= (onb & (bottom <= c.pan_floor_z + c.wedge_low_band)).any(dim=1)

        loaded = self.loaded_now()
        self._loaded |= loaded
        self._loaded_pf |= loaded & self._flipped
        self._since_loaded = torch.where(
            loaded, torch.zeros_like(self._since_loaded),
            (self._since_loaded + 1).clamp(max=10**6))

        # the flip: inverted AND at rest in the pan (both halves must hold at once —
        # a slice sailing through 180 deg mid-air has not flipped until it lands flat)
        self._flipped |= (self.flipped_now() & self.on_pan() & self.settled()
                          & self._present).any(dim=1)

        # the serve: resting on the plate within the arrival window of the last carry
        self._served |= self.served_now() & (self._since_loaded <= c.arrival_window)

        # Metrics: worst blade tilt while carrying; spill = sustained rest on the actual
        # bare surface after loading. Furniture transitions are deliberately excluded:
        # `not on_pan()` is not enough because that predicate is false while a slice
        # tumbles in the pan, and likewise `not on_plate()` while it tips onto the plate.
        tilt = self.carry_tilt_deg()
        self._max_carry_tilt = torch.where(
            loaded, torch.maximum(self._max_carry_tilt, tilt), self._max_carry_tilt)
        bare_now = (self.on_bare_surface() & self._present).any(dim=1)
        self._lost_streak = torch.where(
            bare_now, self._lost_streak + 1, torch.zeros_like(self._lost_streak))
        lost = self._lost_streak >= c.spill_settle_steps
        # In v2, `_loaded` can latch incidentally during the flip. A carry spill is only
        # possible after the explicit post-flip load stage; otherwise a recoverable
        # re-wedge on the bare surface is misclassified as dropped cargo.
        spill_armed = self._loaded_pf if c.goal == "flip_serve" else self._loaded
        self._spills += (lost & ~self._prev_lost & spill_armed).long()
        self._prev_lost = lost

    # ----- state (full, restorable) -----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"spatula": self.spatula, "pan": self.pan, "plate": self.plate,
                  **{n: b for n, b in self.breads.items()}}
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone() for n, b in bodies.items()},
            "machine": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_present", "_lifted", "_wedged", "_flipped", "_loaded",
                                  "_loaded_pf", "_served", "_since_loaded",
                                  "_max_carry_tilt", "_spills", "_lost_streak",
                                  "_prev_lost")},
            **self._grasp_weld_state(env_ids),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"spatula": self.spatula, "pan": self.pan, "plate": self.plate,
                  **{n: b for n, b in self.breads.items()}}
        for n, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["machine"].items():
            getattr(self, k)[env_ids] = v
        self._grasp_weld_restore(state, env_ids)

    # ----- description ------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on a sturdy table"
        goal_text = {
            "serve": (
                "Goal: slide the blade under the bread, carry the slice ON THE BLADE — "
                "nothing holds it there but friction, so keep the blade level — lift it "
                "clear of the pan's rim and set it down flat on the plate, same side up."),
            "flip": (
                "Goal: flip the bread over IN PLACE with the spatula — slide the blade "
                "under it, turn it past vertical and let it land flat back in the pan."),
            "flip_serve": (
                "Goal: first FLIP the bread in the pan (the browned underside ends up), "
                "then slide the blade under it again, carry it over the pan's rim and "
                "set it down flat on the plate."),
        }[c.goal]
        d_lo = 2 * c.bread_r(0) * 100
        d_hi = 2 * c.bread_r(len(c.families) - 1) * 100
        return (
            f"A slice of bread ({d_lo:.0f}-{d_hi:.0f} cm across) lies in a stainless fry "
            f"pan ({2 * c.pan_r_rim_in * 100:.0f} cm across, rim "
            f"{c.pan_rim_top * 100:.0f} cm high) {where}; an empty bamboo plate "
            f"({2 * c.plate_r * 100:.0f} cm) waits beside it. A red-handled spatula "
            f"(thin {c.blade_w * 100:.0f} cm blade) rests on the surface.\n{goal_text}\n"
            f"The pan's rim walls the slice in: enter over the rim, and the wall is a "
            f"backstop a slide can pin the slice against.\n"
            f"TOOL ONLY: never touch the bread with fingers or gripper — it disqualifies "
            f"the episode. The bread must ARRIVE on the blade: a slice pushed, shoved or "
            f"thrown onto the plate does not count. A slice spilled onto the bare surface "
            f"is a failure you can recover from — wedge it up and continue."
            + (
                " The spatula holds in a firm pinch: close the fingers across its handle "
                "and the grip locks; open wide to release."
                if self.cfg.grasp_weld
                else ""
            )
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _stage_table(self) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Ordered latched-stage names + transition scores for the configured goal."""
        return {
            "serve": (("_lifted", "_wedged", "_loaded", "_served"), (10, 30, 60, 100)),
            "flip": (("_lifted", "_wedged", "_flipped"), (10, 30, 100)),
            "flip_serve": (("_lifted", "_wedged", "_flipped", "_loaded_pf", "_served"),
                           (10, 25, 50, 70, 100)),
        }[self.cfg.goal]

    def stage_flags(self) -> torch.Tensor:
        """(N, S) bool: the goal's latched stages, in rubric order."""
        names, _vals = self._stage_table()
        return torch.stack([getattr(self, nm) for nm in names], dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int: transition rubric — the score of the LONGEST LATCHED PREFIX of the
        goal's stage chain (a latch reached out of order — e.g. `loaded` during the flip
        lift — earns nothing until its predecessors are in: the chain is the task)."""
        flags = self.stage_flags().int()
        k = flags.cummin(dim=1).values.sum(dim=1)
        _names, vals = self._stage_table()
        table = torch.tensor((0,) + vals, device=flags.device)
        return table[k]

    def success(self) -> torch.Tensor:
        """(N,) bool, goal-dependent and judged on the CURRENT resting state (latches prove
        the journey, the live predicate proves the destination):
          serve       — served latched AND the bread rests on the plate now;
          flip        — flipped latched AND the bread rests flipped in the pan now;
          flip_serve  — flipped AND served latched AND the bread rests on the plate now."""
        g = self.cfg.goal
        if g == "serve":
            return self._served & self.served_now()
        if g == "flip":
            now = (self.flipped_now() & self.on_pan() & self.settled()
                   & self._present).any(dim=1)
            return self._flipped & now
        return self._flipped & self._served & self.served_now()

    # Grasp-weld contract: `GraspWeldMixin` (robobench.core.grasp_weld) — the
    # weld-on-closure machinery shared by the grasping scenes; this scene supplies the
    # part-side `grasp_sites()`.

