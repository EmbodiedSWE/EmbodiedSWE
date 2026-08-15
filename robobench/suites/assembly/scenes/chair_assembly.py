"""ChairAssemblyScene — attach the tufted backrest to a real dining chair, then bolt it on.

**WIP (2026-08-13): scene physics and the oracle path are in; the bimanual-Franka solution is
still under development (no passing robot smoke yet). Task definition may still shift while
the robot solve settles — treat the scene contract as provisional.**

The object world is a REAL scanned product (an Amazon beige tufted dining chair, split into
parts by scripts/split_chair_asset.py; textures and normal maps ship with the parts under
assets/chair/). The chair BASE — seat cushion with all four wooden legs pre-attached — stands
upright on the floor. Its rear frame carries two horizontal hanger-bolt studs: a smooth steel
shank each, tipped with the factory M16 thread (SDF collision, the nut_thread/ikea machinery).
Beside it lie the loose BACKREST (its lower shell carries two matching through-holes) and two
loose M16 nuts. **Goal (carried here, no task layer): slide the backrest onto both studs at
once — its bottom face rides the rear legs' top faces as a natural rail — then thread each nut
onto its exposed stud tip to clamp the backrest. 3 assembly pairs in all.**

Legs are deliberately pre-attached: screwing legs is the ikea_table task's content, and gating
this task's novel stages (large-part two-point insertion + fine threading) behind four
redundant leg screwings would starve them of attempts. `legs_preattached=False` (the full
flat-pack variant; the split leg parts exist under assets/chair/) is reserved and raises until
its mating features are authored.

Success logic is the furniture-bench relative-pose port carried over from the procedural
predecessor: pairs (base,back), (base,nut_0), (base,nut_1), judged in the BASE's body frame
(lifted/tilted assemblies judge identically), with `should_assembled_first` ordering — nuts
never count before the back is on, and a nut parked on a bare stud physically blocks the
back's 18 mm holes (its 27.8 mm hex cannot pass). Every counted pair WELDS (the ikea
pre-authored FixedJoint toggle pattern), so the finished chair is rigid and survives a shake.
UNLIKE the predecessor, the nut stage demands real rotation: an engaged nut advances ONLY
per its measured rotation about the stud axis, at true M16x2 pitch (the pc_motherboard
screw-joint mechanic, nut-side). Direct SDF threading was measured and rejected: an SDF
thread hosted in a rigid COMPOUND body offers zero axial resistance (a nut fell through
the full thread, 24.7 mm on 0 degrees of spin, on a vised vertical probe), while the same
asset pair threads correctly on nut_thread's world-pinned fixed-base articulation — a
MOVABLE chair cannot pin its studs to the world, so the joint is the thread here, and
crest-skipping is impossible by construction. The threading axis is HORIZONTAL (the scan's
own joint geometry: the shell laps the seat's rear face), which the smoke validates.

Assembled back pose in the base frame: origin at (0, +0.219, +0.320) — measured from the scan
(see scripts/author_chair_rigs.py; the shell's collision front kisses the slab's rear face
there with 0.4 mm slack, the fabric visually compressing against the edge exactly as scanned).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class ChairAssemblySceneCfg(BaseCfg):
    """Config for `ChairAssemblyScene`. Geometry constants are measured from the split scan
    parts (assets/chair/chair_parts.json provenance) — see scripts/author_chair_rigs.py."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tau_xy: float = tunable(0.010)  # max lateral error of a mating feature, base frame (m)
    tau_z: float = tunable(0.026)  # max along-axis error at the seated pose (m).
    # Spans the rig's measured bimodal assembled band about the band-centre
    # seat plane (see back_seat_pos): both rest families are complete,
    # nut-threadable assemblies. Lateral (0.010) and ori (0.94) stay strict.
    ori_cos: float = tunable(0.94)  # min axis-alignment cosine (the source's 0.94, verbatim)
    settle_speed: float = tunable(0.05)  # max child |v| at the moment of welding (m/s)
    # A nut is seated once it has threaded far enough down the exposed stud tip.
    nut_seat_depth: float = tunable(0.020)  # min travel from the thread tip (m); full thread 0.025
    # Part friction (static = dynamic), set at bind — the nut_thread scene's proven pairing:
    # the moving threaded part runs slick against a grippier fixed part.
    nut_friction: float = tunable(0.01)
    base_friction: float = tunable(1.2)  # chair base: studs, shell contacts, AND feet on the
    # floor. 1.2 (was 0.75): the insertion press slid the on-side base 35 mm north at 0.75
    # (fp326 plow, measured); the higher grip keeps the base planted under the press.
    back_friction: float = tunable(0.3)  # the shell riding the leg-top rail / shanks

    # Weld-on-closure grasping (the benchmark's auto-weld contract, the pc_motherboard
    # machinery generalized to EVERY hand on the stage — a bi-Franka binding has two):
    # close the fingers across a part's grip band and it welds to that hand; open wide to
    # release. Gripper envs only (no-op under robot="null").
    grasp_weld: bool = tunable(True)
    # Engage radius 25 mm (the mb scene uses 10): the chair's grip bands sit on big parts
    # with no confusable geometry nearby, and a Franka pinching the backrest's 6 cm shell
    # measurably stalls in-window with its pinch centre ~2 cm off the nominal band line.
    grasp_weld_dist: float = tunable(0.025)
    # Measured-rotation screw joints (the pc_motherboard screw mechanic, nut-side): an
    # engaged nut stays DYNAMIC — something must physically hold and rotate it — but its
    # AXIAL advance is written from its measured rotation about the stud axis at the true
    # thread pitch, one-way through an engagement lash. The stud's SDF thread cannot carry
    # this itself: SDF colliders hosted in a rigid COMPOUND body measurably offer no axial
    # resistance (helix ratio 0.00 on the vertical vise probe; nut_thread's fixed-base
    # ARTICULATION bolt threads correctly on the same box — a movable chair cannot pin
    # its studs to the world, so the joint IS the thread here).
    screw_pitch: float = tunable(0.002)  # m per revolution (M16x2), TWO-WAY (nuts unscrew)
    screw_lash_deg: float = tunable(30.0)  # engaged rotation before the helix couples
    screw_engage_lat: float = tunable(0.004)  # max lateral offset to count as on-tip (m)
    screw_engage_window: float = tunable(0.004)  # tip +/- this in y = the engage band (m)

    # --- tunable: randomization (the task-family knobs) ---------------------------------------
    reset_pos_jitter: float = tunable(0.04)  # uniform +/- xy jitter per loose part at reset (m)
    reset_yaw_deg: float = tunable(60.0)  # uniform +/- yaw per loose part at reset
    seat_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the chair base at reset (m)
    seat_yaw_base: float = tunable(0.0)  # fixed base yaw (deg) — bindings orient the chair
    # to their workspace (rubric is base-frame, so yaw is transparent to judging)
    seat_yaw_deg: float = tunable(20.0)  # uniform +/- yaw jitter on top of the base yaw
    shuffle_slots: bool = tunable(True)  # per-episode random part->slot permutation
    legs_preattached: bool = tunable(True)  # False = the flat-pack variant (NOT YET AUTHORED)

    # --- tunable: placement (kept name-compatible with the robot bindings in configs/envs.py) --
    # Default work pose: a LOW assembly platform (the suite's tables would put a 1 m chair's
    # studs at ~1.4 m, outside Franka reach; the floor puts them at 0.40 m, the envelope's
    # low edge). 0.25 m lands the studs at ~0.65 m; robots stand ON the platform.
    surface_z: float = tunable(0.25)  # work-surface height; 0 = the chair stands on the floor
    seat_pos: tuple = tunable((0.0, 0.0))  # chair-base centre on the surface
    spawn_radii: tuple = tunable((0.45, 0.62))  # scatter ring radii for the 3 loose parts
    spawn_arc: tuple = tunable((200.0, 340.0))  # scatter arc (deg) around the base
    # Explicit part placement, overriding the arc when non-empty (the siblings'
    # `nut_init_xy` pattern): one (x, y, yaw_deg) per manifest part [back, nut_0, nut_1].
    # The packing-table top is 2.47 x 0.76 m — radial scatter around the chair walks off
    # its short axis, so robot bindings lay parts along the LONG axis instead.
    spawn_slots: tuple = tunable(())
    # Staging riser (a low parts pallet) under the backrest's slot: lying flat on the
    # bench, the panel's only jaw-sized pinches (the 54 mm bottom lip, the 60-66 mm
    # low side bands) sit within ~3 cm of the tabletop — inside the palm's own
    # height, MEASURED unreachable (the hand body intersects the bench). The riser
    # lifts them into free air. (x, y) centre, scene frame; () = no riser.
    riser_pos: tuple = tunable(())
    riser_size: tuple = info((0.42, 0.30, 0.12))
    back_spawn_dz: float = tunable(0.0)  # extra back spawn height (set = riser height)
    # Leaning rack (a tall staging block): the backrest spawns LEANING against it at
    # `back_lean_deg`, nearly upright. Every welded-wrist rotation beyond ~25 deg
    # stalls against this controller (measured across 8 variants: solo/dual, either
    # arm, gains 30-90, gravity on/off) while grasps, dual translations, and <=25 deg
    # rolls are reliable — so the staging supplies the uprightness instead of the
    # arms. (x, y) centre, scene frame; () = no rack.
    rack_pos: tuple = tunable(())
    rack_size: tuple = info((0.34, 0.20, 0.45))
    back_lean_deg: float = tunable(0.0)  # 0 = lying face-up; >0 = leaning this far up
    # Chair-base root pose OVERRIDE at reset (scene frame, z relative to the surface):
    # (x, y, z, qw, qx, qy, qz). Non-empty -> the base spawns in exactly this pose
    # (e.g. LYING ON ITS SIDE: legs horizontal, stud pair vertical) instead of
    # standing at seat_pos; seat_pos/seat_yaw/seat_jitter are ignored. The pose
    # should be a MEASURED free rest (drop-probe it) so the spawn is a pure settle.
    base_root_pose: tuple = tunable(())
    # Backrest root pose OVERRIDE at reset, same contract (measured rest only —
    # the curved shell WALKS if dropped off-equilibrium, and a footprint that
    # clips a robot base detonates at 12 m/s, both measured). Overrides the
    # back's spawn slot and its jitters.
    back_root_pose: tuple = tunable(())
    # Uniform scale on the three CHAIR PARTS (base+studs, back+holes, nuts) and
    # every derived geometric constant (lengths x s, masses x s^3, aperture
    # windows x s). The full-size shell sits at the parallel jaw's limit (54-72 mm
    # pinches vs the 80 mm jaw) and its 2.5 kg exceeds the wrist's rotation
    # authority; ~0.75 moves both from marginal to comfortable. Bindings must
    # supply MEASURED rest poses for the scaled parts (the rests change).
    part_scale: float = tunable(1.0)

    # --- info: measured structure (assets/chair, authored by author_chair_rigs.py) -------------
    # The work surface is the suite's vendored Heavy-Duty PackingTable (the ikea/mb table),
    # SUNK so its top lands at `surface_z` (the ikea pattern: the buried part clips below
    # the floor, purely cosmetic). A procedural slab was tried and looked wrong.
    workbench_usd: str = info("")  # empty -> the vendored packing table
    # float = isotropic; a (sx, sy, sz) tuple scales axes independently (e.g. widen the
    # top without stretching the length or height). Authored in cm.
    workbench_scale: Any = info(0.01)
    workbench_height: float = info(0.994)  # its intrinsic top height at this scale
    # ---- Dunbar LADDERBACK geometry (split_dunbar_chair.py + author_chair_lb_rigs.py):
    # base = seat + front legs + stretchers/aprons; back = 2 full-height rear posts +
    # 4 ladder slats (real flat-packs bolt the seat unit onto the rear-post unit).
    # Studs root in the side-apron ends, axis +y, through the posts' 124 mm depth.
    base_mass: float = info(6.0)
    back_mass: float = info(2.5)
    nut_mass: float = info(0.03)  # M16 nut (kg), nut_thread verbatim
    stud_x: float = info(0.214)  # stud axes at (+/- stud_x, stud_z), base frame, pointing +y
    stud_z: float = info(0.367)  # side-apron band centre (measured 0.338..0.395)
    shank_y0: float = info(0.150)  # smooth shank from here (40 mm embedded in the apron end)
    thread_y0: float = info(0.3302)  # exposed thread base (usable nut travel starts here)
    thread_y1: float = info(0.3550)  # thread tip
    slab_rear_y: float = info(0.1816)  # the posts' front (mating) plane at the seated pose
    # back origin at the seated pose, base frame. 0.1816 is the source mesh's
    # assembled plane; the PREDICATE plane sits 12 mm (unscaled) deeper = the
    # centre of the rig's measured BIMODAL assembled band. A clearance-fit
    # insert against a friction-held base rests in one of two deterministic
    # families (+4..+13 mm proud / -22..-31 mm deep of the source plane;
    # 15 runs, fp326-335), the branch set by force-history chaos. BOTH are
    # complete assemblies (studs fully through, 25/64 mm tip protrusion, the
    # NUT clamps the final position by design), so seated spans the band via
    # tau_z. Lateral/ori gates stay strict; the nut-thread gate is the real
    # success metric downstream.
    back_seat_pos: tuple = info((0.0, 0.1696, 0.0))
    back_hole_z: float = info(0.367)  # hole height in the BACK's frame (0.0 + 0.367 = 0.367)
    hole_r_in: float = info(0.019)  # 38 mm collision holes over the 15.6 mm shanks
    nut_r_in: float = info(0.008)  # M16 nut bore radius (ordering-violation detector)
    base_height: float = info(0.473)  # floor -> seat top
    back_height: float = info(0.979)  # post bottoms -> post tops
    back_depth: float = info(0.140)  # post front face -> slat rear extreme
    light_intensity: float = info(2500.0)
    # Asset USDs; empty -> the packaged parts under assets/chair + assets/factory.
    asset_dir: str = info("")
    base_usd: str = info("")
    back_usd: str = info("")
    nut_usd: str = info("")

    def __post_init__(self) -> None:
        if not self.legs_preattached:
            raise ValueError(
                "legs_preattached=False (the flat-pack variant) is reserved: the split leg "
                "parts exist under assets/chair/ but their mating features are not authored yet"
            )
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "chair_lb")
        self.base_usd = self.base_usd or str(Path(self.asset_dir) / "chair_lb_base_rig.usd")
        self.back_usd = self.back_usd or str(Path(self.asset_dir) / "chair_lb_back_rig.usd")
        # silver variant (user request: the factory yellow reads ugly on this scene)
        self.nut_usd = self.nut_usd or str(Path(self.asset_dir) / "factory_nut_m16_silver.usd")
        self.workbench_usd = self.workbench_usd or str(
            assets / "props" / "packing_table" / "SM_HeavyDutyPackingTable_C02_01_physics.usd")
        self.manifest = (("back", "back"), ("nut_0", "nut"), ("nut_1", "nut"))


# Assembly-pair order (fixed): the parent of every pair is the chair base.
PAIRS = ("base-back", "base-nut_0", "base-nut_1")


@SCENES.register("chair")
class ChairAssemblyScene(BaseScene):
    cfg: ChairAssemblySceneCfg

    def __init__(self, cfg: ChairAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or ChairAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, dome light, optional bench, the chair base standing upright, the backrest
        lying face-down nearby, and two loose nuts. The threaded parts load with the high
        solver-iteration counts the SDF threads need (nut_thread verbatim)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        if c.part_scale != 1.0 and not getattr(c, "_part_scale_applied", False):
            # fold the part scale into every derived constant ONCE (assets() is the
            # first consumer; the mechanics/predicates read the same cfg later)
            s = c.part_scale
            for name in ("stud_x", "stud_z", "shank_y0", "thread_y0", "thread_y1",
                         "slab_rear_y", "back_hole_z", "hole_r_in", "nut_r_in",
                         "screw_pitch", "base_height"):
                if hasattr(c, name):
                    setattr(c, name, getattr(c, name) * s)
            c.back_seat_pos = tuple(v * s for v in c.back_seat_pos)
            c.base_mass *= s ** 3
            c.back_mass *= s ** 3
            c.nut_mass *= s ** 3
            c._part_scale_applied = True
        z0 = c.surface_z

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9)),
            ),
        }
        if z0 > 0:  # the vendored packing table, SUNK so its top lands at surface_z (the
            # ikea pattern — the buried part clips below the floor, purely cosmetic).
            # Spawned as AssetBaseCfg with kinematic rigid props, exactly like ikea's
            # workbench (a RigidObjectCfg procedural cuboid crashed the 5.1 GPU view).
            out["bench"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.workbench_usd,
                    scale=((c.workbench_scale,) * 3 if isinstance(c.workbench_scale, float)
                           else tuple(c.workbench_scale)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.workbench_height)),
            )
        if c.riser_pos:  # staging riser under the backrest slot (see cfg note) — a
            # referenced USD mesh like the bench: a procedural CuboidCfg here crashed
            # the GPU physics parse at sim.reset(), twice, measured (the bench has the
            # same history). Authored by scripts/author_riser_usd.py, origin at the
            # bottom face.
            out["riser"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Riser",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(Path(c.asset_dir) / "riser.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.riser_pos[0], c.riser_pos[1], z0)),
            )
        if c.rack_pos:  # leaning rack (see cfg note) — same referenced-USD pattern
            out["rack"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(Path(c.asset_dir) / "rack.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.rack_pos[0], c.rack_pos[1], z0)),
            )

        # Chair base: DYNAMIC (welds bind two dynamic bodies), standing on its feet. NO
        # spawner-wide collision offsets: the box/shank colliders carry their own small
        # per-prim offsets (baked by author_chair_rigs — the 1.2 mm/side hole clearance
        # needs them) while the referenced factory threads keep their defaults — a
        # spawner-wide 0.2 mm offset let the press-fed nut TUNNEL through the SDF crests.
        out["base"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ChairBase",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.base_usd,
                scale=(c.part_scale,) * 3,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    # 0.5 (was 5.0): stored press interpenetration converting to
                    # velocity AT this cap is what launched the assembled chair
                    # off the bench (fp320/fp326, both airborne on video). The
                    # cap bounds ejections to a gentle push-out.
                    max_depenetration_velocity=0.5,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.base_mass),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(c.seat_pos[0], c.seat_pos[1], z0 + 0.002)),
        )
        out["back"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Back",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.back_usd,
                scale=(c.part_scale,) * 3,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.5,  # see the base note (launch cap)
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.back_mass),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=self._slot_pos(0, z0 + 0.008)),
        )
        for i in range(2):
            out[f"nut_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Nut_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.nut_usd,
                    scale=(c.part_scale,) * 3,
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=1.0,  # see the base note (launch cap)
                        # Never sleeps: on the HORIZONTAL stud a nut pauses with zero
                        # velocity mid-thread (no gravity feed, unlike nut_thread's
                        # vertical bolt), PhysX puts it to sleep, and the tensor-API
                        # external wrench is ignored on sleeping bodies (measured:
                        # 1400 steps at 2.5 N, zero displacement).
                        sleep_threshold=0.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.nut_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=self._slot_pos(1 + i, z0 + 0.02)),
            )
        return out

    def _slot_angle(self, i: int) -> float:
        """Nominal arc angle (deg) of scatter slot `i` (3 slots evenly on the arc)."""
        a0, a1 = self.cfg.spawn_arc
        n = len(self.cfg.manifest)
        return a0 + (a1 - a0) * (i + 0.5) / n

    def _slot_xy_yaw(self, i: int) -> tuple[float, float, float]:
        """Slot centre + nominal yaw (deg) for part `i`: explicit `spawn_slots` when set,
        else the scatter arc (yaw 0)."""
        c = self.cfg
        if c.spawn_slots:
            x, y, yaw = c.spawn_slots[i]
            return (c.seat_pos[0] + x, c.seat_pos[1] + y, yaw)
        ang = math.radians(self._slot_angle(i))
        r = c.spawn_radii[i % len(c.spawn_radii)]
        return (c.seat_pos[0] + r * math.cos(ang), c.seat_pos[1] + r * math.sin(ang), 0.0)

    def _slot_pos(self, i: int, z: float) -> tuple:
        x, y, _yaw = self._slot_xy_yaw(i)
        return (x, y, z)

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

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles, allocate the weld flags + ordering metric, set the frictions, and
        pre-author the 3 (disabled) weld joints per env (the ikea toggle pattern)."""
        super().bind(env)
        self.base: RigidObject = env.iscene["base"]
        self.back: RigidObject = env.iscene["back"]
        self.nuts: list[RigidObject] = [env.iscene["nut_0"], env.iscene["nut_1"]]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        self.welded = torch.zeros(n, 3, dtype=torch.bool, device=env.device)
        self.order_violations = torch.zeros(n, dtype=torch.long, device=env.device)
        self._viol_prev = torch.zeros(n, 2, dtype=torch.bool, device=env.device)
        self._set_friction(self.base, self.cfg.base_friction)
        self._set_friction(self.back, self.cfg.back_friction)
        for nut in self.nuts:
            self._set_friction(nut, self.cfg.nut_friction)
        self._precreate_weld_joints()
        self._grasp_weld_bind()
        self._screw_bind()

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite static + dynamic friction on every shape of `asset` (all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(
            mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start (all welds released): the base stands at `seat_pos`
        (+ jitter + yaw); the back lies FACE-DOWN and the nuts flat on the scatter-arc slots —
        randomly permuted per episode."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        # --- chair base: standing upright (default) or an explicit pose override ---
        st = torch.zeros(m, 13, device=dev)
        if c.base_root_pose:
            st[:, 0] = c.base_root_pose[0]
            st[:, 1] = c.base_root_pose[1]
            st[:, 2] = c.surface_z + c.base_root_pose[2]
            for j in range(4):
                st[:, 3 + j] = c.base_root_pose[3 + j]
        else:
            st[:, 0] = c.seat_pos[0]
            st[:, 1] = c.seat_pos[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.seat_jitter
            st[:, 2] = c.surface_z + 0.002
            half = (math.radians(c.seat_yaw_base)
                    + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.seat_yaw_deg)) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.base.write_root_state_to_sim(st, env_ids)

        # --- loose parts: slot permutation + jitter + yaw ---
        n_parts = len(c.manifest)
        if c.shuffle_slots and not c.spawn_slots:  # explicit slots are role-assigned
            perm = torch.rand(m, n_parts, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(n_parts, device=dev).expand(m, n_parts)
        slot_xyy = [self._slot_xy_yaw(i) for i in range(n_parts)]
        slot_x = torch.tensor([s[0] for s in slot_xyy], device=dev)
        slot_y = torch.tensor([s[1] for s in slot_xyy], device=dev)
        slot_yaw = torch.tensor([math.radians(s[2]) for s in slot_xyy], device=dev)
        yaw_amp = math.radians(c.reset_yaw_deg)
        bodies = dict(zip([nm for nm, _k in c.manifest], [self.back, *self.nuts]))
        for i, (name, kind) in enumerate(c.manifest):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot_x[perm[:, i]]
            st[:, 1] = slot_y[perm[:, i]]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            half = (slot_yaw[perm[:, i]]
                    + (torch.rand(m, device=dev) * 2 - 1) * yaw_amp) / 2
            cy, sy = torch.cos(half), torch.sin(half)
            if kind == "back" and c.back_root_pose:
                st[:, 0] = c.back_root_pose[0]
                st[:, 1] = c.back_root_pose[1]
                st[:, 2] = c.surface_z + c.back_root_pose[2]
                for j in range(4):
                    st[:, 3 + j] = c.back_root_pose[3 + j]
            elif kind == "back":
                if c.back_lean_deg != 0.0:
                    # LEANING against the rack, nearly upright (see rack_pos note):
                    # slot-local quat = qx(-+(90 - |lean|)) — 0 deg = flat face-up,
                    # 90 = standing; the bottom edge rests on the bench, the upper
                    # face on the rack's edge. The SIGN of back_lean_deg picks the
                    # tip direction about the slot x-axis (positive tips the way
                    # the negative-qx roll leans; negative mirrors it). Bindings
                    # place slot + rack so the geometry closes; the preview render
                    # verifies the settle.
                    tip_sign = -1.0 if c.back_lean_deg > 0.0 else 1.0
                    half_p = tip_sign * math.radians(90.0 - abs(c.back_lean_deg)) / 2
                    rl = (math.cos(half_p), math.sin(half_p), 0.0, 0.0)
                    st[:, 2] = c.surface_z + 0.012
                else:
                    # LADDER UNIT lying flat, slats up: the frame's front face (local
                    # y=0 plane, the posts' mating face) rests on the bench — qx(-90)
                    # maps local +y to world +z. Unlike the old banana shell, this
                    # flat frame has a true stable rest; a 1.5 cm hop settles it.
                    # (A 5 cm hop bounced it into a tumble; the "fall-through" that
                    # motivated it was really a slide off the bench edge from a slot
                    # whose footprint overhung it — fixed in the binding slots.)
                    # qx(+90): local +y -> +z = slats UP (the -90 sign lays them
                    # DOWN into the bench; depenetration then hurls the unit
                    # through the slab — measured)
                    rl = (0.70711, 0.70711, 0.0, 0.0)
                    st[:, 2] = c.surface_z + c.back_spawn_dz + 0.015
                st[:, 3] = cy * rl[0] - sy * rl[3]
                st[:, 4] = cy * rl[1] - sy * rl[2]
                st[:, 5] = cy * rl[2] + sy * rl[1]
                st[:, 6] = cy * rl[3] + sy * rl[0]
            else:  # nut: flat, screw axis up, free yaw (nut_thread's resting pose)
                st[:, 2] = c.surface_z + 0.02
                st[:, 3], st[:, 6] = cy, sy
            st[:, 0:3] += origin
            bodies[name].write_root_state_to_sim(st, env_ids)

        self.order_violations[env_ids] = 0
        self._viol_prev[env_ids] = False
        self._reconcile_welds(env_ids, torch.zeros(m, 3, dtype=torch.bool, device=dev))
        self._grasp_weld_release_all(env_ids)
        self._screw_reset(env_ids)

    def grasp_sites(self) -> list:
        """Grip bands for the weld-on-closure contract: the ladder unit's four SLATS
        and each nut's hex (across the 24.4 mm flats). Band = (name, handle, p0, p1,
        (lo, hi)), part-local. The slats are the whole point of the ladderback asset:
        45-49 mm y-depth x 60 mm height cross-sections, pinchable anywhere along
        their ~0.40 m length, from above or from the side, in any part pose — no
        curved-shell tactics. Band lines run along each slat's centreline (measured
        member boxes, chair_lb_parts.json); windows admit a closure across either
        the depth (45-49 mm) or the height (60 mm), and release (window top + the
        8 mm hysteresis) stays inside the jaw's 80 mm travel."""
        s = self.cfg.part_scale
        raw = [
            # slats bottom-to-top; nut sites stay at indices 3:5 (the nut weld
            # readout `grasp_held[:, :, 3:5]` depends on that ordering)
            ("slat_0", self.back, (-0.19, 0.052, 0.564), (0.19, 0.052, 0.564), (0.038, 0.065)),
            ("slat_1", self.back, (-0.19, 0.075, 0.697), (0.19, 0.075, 0.697), (0.038, 0.065)),
            ("slat_2", self.back, (-0.19, 0.095, 0.823), (0.19, 0.095, 0.823), (0.038, 0.065)),
            ("nut_0", self.nuts[0], (-0.004, 0.0, 0.0165), (0.004, 0.0, 0.0165), (0.020, 0.027)),
            ("nut_1", self.nuts[1], (-0.004, 0.0, 0.0165), (0.004, 0.0, 0.0165), (0.020, 0.027)),
            ("slat_3", self.back, (-0.19, 0.115, 0.944), (0.19, 0.115, 0.944), (0.038, 0.065)),
            # post bands: the rear posts' centrelines (74 x 124 mm cross-section;
            # the 74 mm side presents to a cross-post closure = 55 mm at the task
            # scale). In the ON-SIDE task pose the posts are horizontal beams and
            # their south halves pass 0.35-0.4 m from the arms — the natural
            # slide/carry handles (the slats stand on the unit's far half there).
            # window top 0.094: MEASURED bites are 63.7 mm (side grip) and
            # 65.6+ mm (perpendicular grip on the ~4-deg-tilted resting beam,
            # which presents diagonally); scaled hi 70.5 mm + 8 mm release
            # hysteresis = 78.5 mm, still inside the 80 mm jaw.
            ("post_l", self.back, (-0.214, 0.062, 0.10), (-0.214, 0.062, 0.65), (0.042, 0.094)),
            # post_r split into WEST/EAST half-bands: the weld contract engages
            # one hand per SITE, so the dual-arm firm hold (user design,
            # 2026-08-10 — one arm cannot stabilize the 0.73 m unit, measured
            # drag-rotation) needs each hand its own site on the same beam.
            # This is the ikea dual-carry pattern.
            ("post_r_w", self.back, (0.214, 0.062, 0.10), (0.214, 0.062, 0.35), (0.042, 0.094)),
            ("post_r_e", self.back, (0.214, 0.062, 0.40), (0.214, 0.062, 0.65), (0.042, 0.094)),
        ]
        if s == 1.0:
            return raw
        return [(nm, h, tuple(v * s for v in p0), tuple(v * s for v in p1),
                 (lo * s, hi * s)) for nm, h, p0, p1, (lo, hi) in raw]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile the grasp contract, the screw joints, and every weld against the
        assembly criterion, and accumulate the ordering-violation metric. Runs each step."""
        self._grasp_weld_step()
        self._screw_step()
        ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        self._reconcile_welds(ids, self._weld_targets()[ids])
        viol = self._nut_on_stud() & ~self.welded[:, 0:1]
        edges = viol & ~self._viol_prev
        self.order_violations[ids] += edges[ids].sum(dim=1)
        self._viol_prev[ids] = viol[ids]

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "back": self.back.data.root_state_w[env_ids].clone(),
            "nuts": torch.stack([b.data.root_state_w[env_ids].clone() for b in self.nuts], dim=1),
            "welded": self.welded[env_ids].clone(),
            "order_violations": self.order_violations[env_ids].clone(),
            "viol_prev": self._viol_prev[env_ids].clone(),
            **self._grasp_weld_state(env_ids),
            **self._screw_state(env_ids),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned: write the bodies, then reconcile the welds to
        exactly the recorded flags — from the restored poses, since handle `.data` is stale
        right after a write."""
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.back.write_root_state_to_sim(state["back"], env_ids)
        for i, b in enumerate(self.nuts):
            b.write_root_state_to_sim(state["nuts"][:, i], env_ids)
        self.order_violations[env_ids] = state["order_violations"]
        self._viol_prev[env_ids] = state["viol_prev"]
        self._reconcile_welds(env_ids, state["welded"], saved=state)
        self._grasp_weld_restore(state, env_ids)
        self._screw_restore(state, env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        where = "on the floor" if c.surface_z <= 0 else "on a workbench"
        return (
            f"A real beige tufted dining chair, partly assembled, stands upright {where} on its "
            f"four wooden legs (seat top at {c.base_height:.2f} m). Two horizontal steel studs "
            f"protrude backward from the seat's rear frame at (x = +/-{c.stud_x:.2f} m, "
            f"z = {c.stud_z:.3f} m in the seat's frame): each is a smooth shank ending in an "
            f"exposed threaded tip. Nearby lie the chair's tufted BACKREST, face-down — its "
            f"lower shell carries two through-holes matching the studs — and two loose M16 "
            f"nuts.\n"
            f"Goal: assemble the backrest — lift it upright, slide BOTH holes over BOTH studs "
            f"at once (its bottom face can ride the rear legs' top faces as a rail) until the "
            f"shell seats against the seat's rear edge, then thread each nut onto an exposed "
            f"stud tip (press toward the chair and turn about the stud axis) until it clamps. "
            f"A nut spun onto a bare stud first blocks the backrest's holes and never counts "
            f"before the backrest is on. The chair is assembled once the backrest and both "
            f"nuts are locked (3 pairs in all)."
        )

    # ----- progress / the ported assembly graph ---------------------------------------------------
    def assembled(self) -> torch.Tensor:
        """(N, 3) bool, sticky: which pairs are assembled (welded), ordered as PAIRS."""
        return self.welded.clone()

    def pairs_assembled(self) -> torch.Tensor:
        """(N,) int in 0..3: the cumulative assembly progress (+1 per pair)."""
        return self.welded.sum(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) int: 0..100, a third per assembled pair."""
        return (100 * self.pairs_assembled()) // 3

    def success(self) -> torch.Tensor:
        """(N,) bool: all 3 pairs assembled (scene-level success; the oracle's target)."""
        return self.welded.all(dim=1)

    def pair_seated(self) -> torch.Tensor:
        """(N, 3) bool, live geometry: each pair currently at its mating pose (nut pairs gated
        on the (base,back) pair — `should_assembled_first`, ported as logic)."""
        back = self._back_seated()  # (N,)
        nuts = self._nuts_seated()  # (N, 2)
        gate = (self.welded[:, 0] | back).unsqueeze(1)
        return torch.cat([back.unsqueeze(1), nuts & gate], dim=1)

    # --- per-stage predicates, all in the BASE's body frame ----------------------------------------
    def _base_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base.data.root_pos_w, self.base.data.root_quat_w

    def _axes_w(self, quat: torch.Tensor, axis: tuple) -> torch.Tensor:
        """A local axis of a batch of quats, world frame, shape (..., 3)."""
        from isaaclab.utils.math import quat_apply

        flat = quat.reshape(-1, 4)
        v = torch.tensor(axis, dtype=torch.float32, device=quat.device).expand(flat.shape[0], 3)
        return quat_apply(flat, v).reshape(*quat.shape[:-1], 3)

    def _back_seated(self) -> torch.Tensor:
        """(N,): BOTH of the back's holes around studs at depth in the BASE frame — the
        two-point insertion term, per hole: its centre within `tau_xy` of SOME stud axis in
        the xz-plane and at the seated insertion depth within `tau_z` (order-independent
        across studs, the source's multi-candidate port; a 180-degree insertion is excluded
        by the crown's lean-back colliding with the slab, not by the rubric) — plus the back
        upright: its z-axis within `ori_cos` of the base's."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        bp, bq = self._base_frame()
        kp, kq = self.back.data.root_pos_w, self.back.data.root_quat_w
        studs = torch.tensor([[-c.stud_x, c.stud_z], [c.stud_x, c.stud_z]], device=bp.device)
        seat_y = c.back_seat_pos[1]  # holes lie on the back's y=0 plane
        base_up = self._axes_w(bq, (0.0, 0.0, 1.0))
        back_up = self._axes_w(kq, (0.0, 0.0, 1.0))
        ok = (back_up * base_up).sum(dim=-1) >= c.ori_cos
        for hx in (-c.stud_x, c.stud_x):
            off = torch.tensor([hx, 0.0, c.back_hole_z], device=bp.device).expand(bp.shape[0], 3)
            hole = kp + quat_apply(kq, off)  # hole centre, world
            loc = quat_apply_inverse(bq, hole - bp)  # base frame
            lat = (loc[:, None, [0, 2]] - studs[None]).norm(dim=-1).amin(dim=1)
            ok = ok & (lat <= c.tau_xy) & ((loc[:, 1] - seat_y).abs() <= c.tau_z)
        return ok

    def _nut_rel_base(self) -> torch.Tensor:
        """Each nut's origin in the BASE's body frame, shape (N, 2, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        bp, bq = self._base_frame()
        return torch.stack(
            [quat_apply_inverse(bq, nut.data.root_pos_w - bp) for nut in self.nuts], dim=1)

    def _nut_stud_lat(self) -> torch.Tensor:
        """Each nut's lateral (xz-plane) distance to its NEAREST stud axis, (N, 2)."""
        c = self.cfg
        loc = self._nut_rel_base()  # (N, 2, 3)
        studs = torch.tensor([[-c.stud_x, c.stud_z], [c.stud_x, c.stud_z]], device=loc.device)
        lat = loc[:, :, None, [0, 2]] - studs[None, None]  # (N, 2, 2, 2)
        return lat.norm(dim=-1).amin(dim=-1)

    def _nuts_seated(self) -> torch.Tensor:
        """(N, 2), geometry only (the ordering gate lives in `pair_seated`): each nut within
        `tau_xy` of a stud axis, threaded to >= `nut_seat_depth` from the thread tip, its screw
        axis within `ori_cos` of the stud axis (|cos| — the nut is flip-symmetric)."""
        c = self.cfg
        loc = self._nut_rel_base()
        near = self._nut_stud_lat() <= c.tau_xy
        depth_ok = (c.thread_y1 - loc[..., 1]) >= c.nut_seat_depth
        bp, bq = self._base_frame()
        base_fwd = self._axes_w(bq, (0.0, 1.0, 0.0))
        nut_axis = torch.stack(
            [self._axes_w(nut.data.root_quat_w, (0.0, 0.0, 1.0)) for nut in self.nuts], dim=1)
        ori_ok = (nut_axis * base_fwd[:, None, :]).sum(dim=-1).abs() >= c.ori_cos
        return near & depth_ok & ori_ok

    def _nut_on_stud(self) -> torch.Tensor:
        """(N, 2): nut riding anywhere along a stud's span — the ordering-violation detector,
        deliberately looser than `_nuts_seated` (any depth on the shank or thread counts)."""
        c = self.cfg
        loc = self._nut_rel_base()
        lat = self._nut_stud_lat()
        y_ok = (loc[..., 1] >= c.shank_y0 - 0.004) & (loc[..., 1] <= c.thread_y1 + 0.004)
        return (lat <= c.nut_r_in + 0.006) & y_ok

    # ----- weld machinery (private; auto-weld on seat, the ikea sim-hack) -------------------------
    # Every pair's FixedJoint is authored DISABLED before play and only toggled on/off; a
    # runtime joint binds two DYNAMIC bodies (the base is always dynamic). All pairs weld
    # child->base. Monotonic until reset.
    WELD_CHILD_BODIES: ClassVar[tuple] = ("Back", "Nut_0/factory_nut_loose", "Nut_1/factory_nut_loose")

    def _weld_targets(self) -> torch.Tensor:
        """Which pairs SHOULD be welded now, (N, 3): already-welded stays; a live-seated pair
        welds once its child is also settling (debounce); nut pairs additionally require the
        (base,back) pair to be WELDED already (`should_assembled_first`, hard form)."""
        c = self.cfg
        back, nuts = self._back_seated(), self._nuts_seated()
        child_v = torch.stack(
            [b.data.root_lin_vel_w.norm(dim=-1) for b in (self.back, *self.nuts)], dim=1)
        settling = child_v < c.settle_speed
        gate = self.welded[:, 0].unsqueeze(1)
        live = torch.cat([back.unsqueeze(1), nuts & gate], dim=1)
        return self.welded | (live & settling)

    def _precreate_weld_joints(self) -> None:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        self._weld_paths: list[list[str]] = []
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            paths = []
            for k, child in enumerate(self.WELD_CHILD_BODIES):
                jp = f"{base}/rweld_{k}"
                j = UsdPhysics.FixedJoint.Define(stage, jp)
                j.CreateBody0Rel().SetTargets([f"{base}/ChairBase"])
                j.CreateBody1Rel().SetTargets([f"{base}/{child}"])
                j.CreateJointEnabledAttr(False)
                paths.append(jp)
            self._weld_paths.append(paths)

    def _reconcile_welds(self, env_ids, target, *, saved: dict[str, Any] | None = None) -> None:
        """Bring every (env, pair) joint for `env_ids` into line with `target` (bool, rows
        aligned to `env_ids`). Welds use `saved` poses when given (a set_state restore, where
        handle `.data` is stale right after a write), else the live poses."""
        have = self.welded[env_ids]
        to_weld = target & ~have
        to_unweld = have & ~target
        if to_weld.any():
            if saved is not None:
                ppos, pquat = saved["base"][:, 0:3], saved["base"][:, 3:7]
                child_states = torch.stack(
                    [saved["back"], saved["nuts"][:, 0], saved["nuts"][:, 1]], dim=1)
            else:
                ppos = self.base.data.root_pos_w[env_ids]
                pquat = self.base.data.root_quat_w[env_ids]
                child_states = torch.stack(
                    [b.data.root_state_w[env_ids] for b in (self.back, *self.nuts)], dim=1)
            cpos, cquat = child_states[..., 0:3], child_states[..., 3:7]
            for row, k in to_weld.nonzero(as_tuple=False).tolist():
                self._weld_pair(int(env_ids[row]), k, ppos[row], pquat[row],
                                cpos[row, k], cquat[row, k])
        for row, k in to_unweld.nonzero(as_tuple=False).tolist():
            self._unweld_pair(int(env_ids[row]), k)

    def _weld_pair(self, env_i: int, k: int, pp, pq, cp, cq) -> None:
        """Lock pair k in env_i at the relative pose implied by world poses pp/pq (base) and
        cp/cq (child)."""
        from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
        from pxr import Gf, UsdPhysics

        q1c = quat_conjugate(cq.unsqueeze(0))
        rel_pos = quat_apply(q1c, (pp - cp).unsqueeze(0))[0]
        rel_rot = quat_mul(q1c, pq.unsqueeze(0))[0]
        j = UsdPhysics.FixedJoint.Get(self.env.stage, self._weld_paths[env_i][k])
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in rel_pos.tolist())))
        w, x, y, z = (float(v) for v in rel_rot.tolist())
        j.CreateLocalRot1Attr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        j.GetJointEnabledAttr().Set(True)
        self.welded[env_i, k] = True

    def _unweld_pair(self, env_i: int, k: int) -> None:
        """Release one weld by disabling its joint (re-weldable later)."""
        from pxr import UsdPhysics

        UsdPhysics.FixedJoint.Get(self.env.stage,
                                  self._weld_paths[env_i][k]).GetJointEnabledAttr().Set(False)
        self.welded[env_i, k] = False

    # ----- screw-joint machinery (measured-rotation helix; private — not an agent action) -------
    # An engaged nut stays dynamic; each substep its rotation about its stud's axis is
    # measured and its AXIAL position is written from the accumulated screw-in rotation at
    # true pitch (one-way through the lash, hard stop at the collar). Engagement is earned
    # by real placement: on-tip within `screw_engage_lat`/`screw_engage_window`, axis
    # aligned. A nut yanked off its written pose disengages (turn resets).
    def _screw_bind(self) -> None:
        from pxr import UsdPhysics

        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        self._scr_eng = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._scr_turn = torch.zeros(n, 2, device=dev)
        self._scr_coupled = torch.zeros(n, 2, device=dev)
        self._scr_prev = torch.zeros(n, 2, device=dev)
        self._scr_stud = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self._scr_prev_held = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._scr_turn_max = 2 * math.pi * (c.thread_y1 - c.thread_y0 - 0.0008) / c.screw_pitch
        # The joint IS the thread (mb pattern): the stud threads' collision goes off at
        # bind — before play — so the mechanic alone owns engaged-nut kinematics. The
        # smooth shanks stay live (the back's holes ride them; a nut cannot pass one).
        stage = self.env.stage
        for e in range(n):
            for k in range(2):
                p = stage.GetPrimAtPath(
                    f"/World/envs/env_{e}/ChairBase/stud_{k}/bolt/factory_bolt_loose/collisions")
                if not p.IsValid():
                    raise RuntimeError(f"[screw] stud thread prim missing: stud {k}, env {e}")
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)

    def _screw_reset(self, env_ids: torch.Tensor) -> None:
        self._scr_eng[env_ids] = False
        self._scr_turn[env_ids] = 0.0
        self._scr_coupled[env_ids] = 0.0

    def _screw_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "screw_engaged": self._scr_eng[env_ids].clone(),
            "screw_turn": self._scr_turn[env_ids].clone(),
            "screw_coupled": self._scr_coupled[env_ids].clone(),
            "screw_prev": self._scr_prev[env_ids].clone(),
            "screw_stud": self._scr_stud[env_ids].clone(),
        }

    def _screw_restore(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        if "screw_engaged" not in state:
            return
        self._scr_eng[env_ids] = state["screw_engaged"]
        self._scr_turn[env_ids] = state["screw_turn"]
        self._scr_coupled[env_ids] = state["screw_coupled"]
        self._scr_prev[env_ids] = state["screw_prev"]
        self._scr_stud[env_ids] = state["screw_stud"]

    def _nuts_grasp_held(self) -> torch.Tensor:
        """(N, 2) bool: which nuts are currently hand-welded (any hand). False everywhere
        when the grasp contract is off (NullRobot)."""
        if not getattr(self, "_gw_on", False) or not hasattr(self, "grasp_held"):
            return torch.zeros(self.env.num_envs, 2, dtype=torch.bool, device=self.env.device)
        # grasp sites are ordered (slat_0, slat_1, slat_2, nut_0, nut_1, slat_3)
        return self.grasp_held[:, :, 3:5].any(dim=1)

    def _nut_spin(self, bq: torch.Tensor) -> torch.Tensor:
        """Each nut's rotation angle about the stud axis (+y, base frame), (N, 2). The
        angle of the nut's local x-axis in the base's x/z plane — continuous tracking
        handles the wrap."""
        from isaaclab.utils.math import quat_apply_inverse

        out = []
        for nut in self.nuts:
            ax = self._axes_w(nut.data.root_quat_w, (1.0, 0.0, 0.0))
            ax_b = quat_apply_inverse(bq, ax)  # base frame
            out.append(torch.atan2(ax_b[:, 2], ax_b[:, 0]))
        return torch.stack(out, dim=1)

    def _screw_step(self) -> None:
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        bp, bq = self._base_frame()
        loc = self._nut_rel_base()  # (N, 2, 3)
        studs = torch.tensor([[-c.stud_x, c.stud_z], [c.stud_x, c.stud_z]], device=loc.device)
        lat_all = (loc[:, :, None, [0, 2]] - studs[None, None]).norm(dim=-1)  # (N, 2, 2)
        lat, nearest = lat_all.min(dim=-1)
        # _nut_spin measures -theta_y (atan2 of local-x in the base x/z plane flips the
        # sense), so screw-in torque about -y RAISES the measured angle: accumulate it
        # un-negated. (The first, mb-copied negation left `turn` at zero while the nut
        # spun pinned at the tip until numeric drift tripped the yank release.)
        spin = self._nut_spin(bq)
        dspin = (spin - self._scr_prev + math.pi) % (2 * math.pi) - math.pi  # + = screw-in
        self._scr_prev = spin

        base_fwd = self._axes_w(bq, (0.0, 1.0, 0.0))
        nut_axis = torch.stack(
            [self._axes_w(nut.data.root_quat_w, (0.0, 0.0, 1.0)) for nut in self.nuts], dim=1)
        aligned = (nut_axis * base_fwd[:, None, :]).sum(dim=-1).abs() >= c.ori_cos

        # engage: on-tip, aligned, not already engaged, not welded
        at_tip = (loc[..., 1] - c.thread_y1).abs() <= c.screw_engage_window
        can = ~self._scr_eng & at_tip & (lat <= c.screw_engage_lat) & aligned \
            & ~self.welded[:, 1:3]
        if can.any():
            self._scr_stud = torch.where(can, nearest, self._scr_stud)
            self._scr_coupled = torch.where(can, torch.zeros_like(self._scr_coupled),
                                            self._scr_coupled)
            self._scr_eng |= can

        # a hand-welded nut belongs to the GRASP joint: the mechanic accumulates its
        # measured rotation but must not fight the weld with pose writes (two owners of
        # one body explode); the helix pose is snapped back in on release
        held = self._nuts_grasp_held()

        # disengage: yanked far off the written pose (e.g. teleported away); a held nut
        # gets a wider leash — the hand legitimately wiggles it on the stud
        tgt_y = c.thread_y1 - c.screw_pitch * self._scr_turn / (2 * math.pi)
        leash = torch.where(held, 0.025, 0.010)
        off = (lat > leash) | ((loc[..., 1] - tgt_y).abs() > leash)
        yanked = self._scr_eng & off
        if yanked.any():
            self._scr_eng &= ~yanked
            self._scr_turn = torch.where(yanked, torch.zeros_like(self._scr_turn), self._scr_turn)

        live = self._scr_eng & ~self.welded[:, 1:3]
        if not live.any():
            return
        self._scr_coupled = torch.where(live, self._scr_coupled + dspin, self._scr_coupled)
        lash = math.radians(c.screw_lash_deg)
        # TWO-WAY: turn follows the coupled rotation through the lash in both directions
        # (real nuts unscrew; mb's one-way ratchet would leave a premature nut permanently
        # blocking its stud with no agent-side recovery).
        follow = (self._scr_coupled - lash).clamp(min=0.0, max=self._scr_turn_max)
        self._scr_turn = torch.where(live, follow, self._scr_turn)
        # fully unscrewed and still counter-rotating -> release the joint at the tip (the
        # nut is free to be lifted off)
        backed_out = live & (self._scr_turn <= 0.0) & (self._scr_coupled < -math.radians(60.0))
        if backed_out.any():
            self._scr_eng &= ~backed_out

        # write the engaged nuts: axial position from the turn, lateral pinned to the stud
        # axis, orientation + spin left fully live (the twist is real). Hand-held nuts are
        # skipped (the weld owns them) and snapped onto the helix as the hand lets go.
        release_edge = self._scr_prev_held & ~held & self._scr_eng
        self._scr_prev_held = held
        tgt_y = c.thread_y1 - c.screw_pitch * self._scr_turn / (2 * math.pi)
        for k, nut in enumerate(self.nuts):
            rows = ((live[:, k] & ~held[:, k]) | release_edge[:, k]).nonzero(
                as_tuple=False).flatten()
            if not len(rows):
                continue
            st = nut.data.root_state_w[rows].clone()
            sx = studs[self._scr_stud[rows, k]]  # (m, 2): x, z of the engaged stud
            tgt = torch.stack([sx[:, 0], tgt_y[rows, k], sx[:, 1]], dim=1)
            st[:, 0:3] = bp[rows] + quat_apply(bq[rows], tgt)
            axis = self._axes_w(bq[rows], (0.0, 1.0, 0.0))
            st[:, 7:10] = axis * (st[:, 7:10] * axis).sum(dim=-1, keepdim=True)
            # spin stays live but only ABOUT the stud axis — off-axis angular drift
            # accumulates against the per-substep position pin and ends in a yank release
            st[:, 10:13] = axis * (st[:, 10:13] * axis).sum(dim=-1, keepdim=True)
            nut.write_root_state_to_sim(st, rows)

    # ----- grasp-weld machinery (weld-on-closure; private — not an agent action) ----------------
    # The pc_motherboard contract generalized to EVERY hand on the stage (a bi-Franka env
    # has two `panda_hand`s; hand h and site s get their own joint pool). Pre-authored,
    # normally-disabled FixedJoint pools, toggled and never created mid-sim. PhysX latches
    # a joint's local frames on FIRST enable, so every engage consumes a fresh pool joint.
    # Engage (per hand, debounced): pinch point within `grasp_weld_dist` of a site's LIVE
    # grip band + aperture inside the site's closure window + fingers STALLED. Release:
    # aperture past window-top + margin. One hold per hand; one hand per part.
    GRASP_HAND_BODY: ClassVar[str] = "panda_hand"
    GRASP_FINGER_JOINTS: ClassVar[str] = "panda_finger_joint.*"
    GRASP_PINCH_OFFSET: ClassVar[float] = 0.1034  # hand origin -> finger-pad centre
    GRASP_POOL: ClassVar[int] = 64  # engages per (env, hand, site) per run (nut regrips are many)
    GRASP_STALL: ClassVar[float] = 0.01  # max |finger vel| sum (m/s): fingers stopped ON the part
    GRASP_DEBOUNCE: ClassVar[int] = 8  # consecutive qualifying substeps before the weld engages
    # Release at window-top + this (m). MUST leave window-top + margin strictly below
    # the jaw's 0.080 m max travel or a grip on a window-top-thick part can NEVER
    # release (measured: a hand commanded open stayed welded through a whole phase —
    # 0.072 window + 0.008 margin = exactly 0.080, reachable but never exceedable).
    GRASP_RELEASE_MARGIN: ClassVar[float] = 0.006

    def _grasp_weld_bind(self) -> None:
        """Discover every hand, author the (disabled) joint pools, allocate the hold state.
        Called from `bind()` — authoring must happen BEFORE the sim starts playing."""
        env = self.env
        n = env.num_envs
        self._gw_on = bool(getattr(self.cfg, "grasp_weld", False))
        self._gw_arts: list | None = None  # (articulation, hand_body_i, finger_ids) per hand
        self._gw_sites: list = []
        if not self._gw_on:
            return
        hands0 = self._gw_find_hand_prims()
        if not hands0:  # no gripper in this embodiment (e.g. robot="null") -> no-op contract
            self._gw_on = False
            print(f"[grasp-weld] no '{self.GRASP_HAND_BODY}' on the stage — contract disabled",
                  flush=True)
            return
        self._gw_hands0 = hands0
        self._gw_sites = list(self.grasp_sites())
        h, s = len(hands0), len(self._gw_sites)
        dev = env.device
        self.grasp_held = torch.zeros(n, h, s, dtype=torch.bool, device=dev)
        self._gw_rel_p = torch.zeros(n, h, s, 3, device=dev)
        self._gw_rel_q = torch.zeros(n, h, s, 4, device=dev)
        self._gw_count = torch.zeros(n, h, s, dtype=torch.int32, device=dev)
        self._gw_pool_i = [[[0] * s for _ in range(h)] for _ in range(n)]
        self._gw_pool_warned: set = set()
        self._gw_author_pools(hands0)

    def _gw_find_hand_prims(self) -> list[str]:
        """Every hand body's prim path under env_0 (clones are identical), sorted for a
        stable hand order across runs."""
        from pxr import Usd

        root = self.env.stage.GetPrimAtPath("/World/envs/env_0")
        if not root.IsValid():
            return []
        return sorted(str(p.GetPath()) for p in Usd.PrimRange(root)
                      if p.GetName() == self.GRASP_HAND_BODY)

    def _gw_part_path(self, obj, env_i: int) -> str:
        """The part's RIGID-BODY prim path in env `env_i` (the asset root is not always the
        body — the factory nut nests it one level down)."""
        from pxr import Usd, UsdPhysics

        p = obj.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        root = p.replace("env_.*", f"env_{env_i}")
        prim = self.env.stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise RuntimeError(f"[grasp-weld] part prim missing: {root}")
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(child.GetPath())
        raise RuntimeError(f"[grasp-weld] no RigidBodyAPI prim under {root}")

    def _gw_author_pools(self, hands0: list[str]) -> None:
        from pxr import Gf, UsdPhysics

        stage = self.env.stage
        self._gw_paths: list[list[list[list[str]]]] = []  # [env][hand][site][k]
        for i in range(self.env.num_envs):
            per_hand = []
            for hi, hand0 in enumerate(hands0):
                hand = hand0.replace("env_0", f"env_{i}")
                rows = []
                for name, obj, _p0, _p1, _win in self._gw_sites:
                    part = self._gw_part_path(obj, i)
                    row = []
                    for k in range(self.GRASP_POOL):
                        jp = f"/World/envs/env_{i}/gweld_h{hi}_{name}_{k}"
                        j = UsdPhysics.FixedJoint.Define(stage, jp)
                        j.CreateBody0Rel().SetTargets([hand])
                        j.CreateBody1Rel().SetTargets([part])
                        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                        j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                        j.CreateJointEnabledAttr(False)
                        j.CreateExcludeFromArticulationAttr(True)
                        row.append(jp)
                    rows.append(row)
                per_hand.append(rows)
            self._gw_paths.append(per_hand)

    def _gw_resolve_hands(self) -> bool:
        """Cache (articulation, hand body index, finger joint ids) per hand on first use.
        With a MultiRobot the hands live on different child articulations; each hand prim
        is matched to the articulation whose body list contains it."""
        if self._gw_arts is not None:
            return True
        try:
            robots = getattr(self.env.robot, "robots", None)
            arts = ([r.articulation for r in robots.values()] if robots
                    else [self.env.robot.articulation])
            # subtree matching: the hand prim must live INSIDE the articulation's own
            # subtree (root + '/'). No looser fallback: an earlier dirname-based clause
            # stripped the robot segment itself and bound EVERY hand to the first-listed
            # articulation (both hands read the left arm; the right pinch never counted).
            resolved = []
            for hand0 in self._gw_hands0:
                match = None
                for a in arts:
                    root = str(a.root_physx_view.prim_paths[0])
                    if hand0.startswith(root + "/") or hand0 == root:
                        match = a
                        break
                if match is None:
                    raise RuntimeError(
                        f"[grasp-weld] no articulation owns hand {hand0}; roots="
                        f"{[str(a.root_physx_view.prim_paths[0]) for a in arts]}")
                hand_i = match.body_names.index(self.GRASP_HAND_BODY)
                fingers = match.find_joints([self.GRASP_FINGER_JOINTS])[0]
                assert len(fingers) == 2
                resolved.append((match, hand_i, fingers))
                print(f"[grasp-weld] hand {hand0} -> articulation root "
                      f"{match.root_physx_view.prim_paths[0]}", flush=True)
            self._gw_arts = resolved
        except Exception as e:  # not a gripper we understand -> disable, loudly
            self._gw_on = False
            print(f"[grasp-weld] DISABLED after error: {e!r}", flush=True)
            return False
        return True

    def _grasp_weld_step(self) -> None:
        """Reconcile engages + releases against the closure criterion, per hand."""
        if not getattr(self, "_gw_on", False) or not self._gw_sites or not self._gw_resolve_hands():
            return
        from isaaclab.utils.math import quat_apply

        for hi, (art, hand_i, fingers) in enumerate(self._gw_arts):
            hp = art.data.body_pos_w[:, hand_i]
            hq = art.data.body_quat_w[:, hand_i]
            gap = art.data.joint_pos[:, fingers].sum(dim=-1)
            stalled = art.data.joint_vel[:, fingers].abs().sum(dim=-1) < self.GRASP_STALL
            approach = torch.zeros_like(hp)
            approach[:, 2] = self.GRASP_PINCH_OFFSET
            pinch = hp + quat_apply(hq, approach)

            for row, s in self.grasp_held[:, hi].nonzero(as_tuple=False).tolist():
                if gap[row] > self._gw_sites[s][4][1] + self.GRASP_RELEASE_MARGIN:
                    self._gw_release(row, hi, s)

            hand_free = ~self.grasp_held[:, hi].any(dim=-1)
            part_free = ~self.grasp_held.any(dim=1)  # (n, s): not held by ANY hand
            dists = self._gw_site_dists(pinch)
            cd = getattr(self.cfg, "grasp_weld_dist", 0.010)
            if os.environ.get("GRASP_DEBUG"):
                if not hasattr(self, "_gw_dbg"):
                    self._gw_dbg = 0
                    print(f"[grasp-weld dbg] hand order: {self._gw_hands0}", flush=True)
                if hi == 0:
                    self._gw_dbg += 1
                if self._gw_dbg % 120 == 0:
                    print(f"[grasp-weld dbg] hand {hi}: dists="
                          f"{[f'{v * 1000:.0f}' for v in dists[0].tolist()]}mm "
                          f"gap={float(gap[0]) * 1000:.1f}mm "
                          f"fingervel={float(art.data.joint_vel[0, fingers].abs().sum()) * 1000:.1f}mm/s "
                          f"stalled={bool(stalled[0])} free={bool(hand_free[0])}", flush=True)
            ok = torch.stack(
                [(dists[:, s] < cd) & (gap > win[0]) & (gap < win[1]) & stalled & part_free[:, s]
                 for s, (_n, _o, _p0, _p1, win) in enumerate(self._gw_sites)],
                dim=-1,
            ) & hand_free.unsqueeze(-1)
            self._gw_count[:, hi] = torch.where(
                ok, self._gw_count[:, hi] + 1, torch.zeros_like(self._gw_count[:, hi]))
            ready = (self._gw_count[:, hi] >= self.GRASP_DEBOUNCE).any(dim=-1) & hand_free
            for row in ready.nonzero(as_tuple=False).flatten().tolist():
                masked = torch.where(self._gw_count[row, hi] >= self.GRASP_DEBOUNCE,
                                     dists[row], torch.full_like(dists[row], torch.inf))
                s = int(masked.argmin())
                self._gw_engage(row, hi, s, hp[row], hq[row], gap[row])

    def _gw_site_dists(self, pinch: torch.Tensor) -> torch.Tensor:
        """Pinch-point distance to every site's live grip band, (num_envs, num_sites)."""
        from isaaclab.utils.math import quat_apply

        n = pinch.shape[0]
        out = []
        for _name, obj, p0, p1, _win in self._gw_sites:
            pp, pq = obj.data.root_pos_w, obj.data.root_quat_w
            a = pp + quat_apply(pq, torch.tensor(p0, device=pinch.device).expand(n, 3))
            b = pp + quat_apply(pq, torch.tensor(p1, device=pinch.device).expand(n, 3))
            ab = b - a
            t = ((pinch - a) * ab).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-12)
            closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
            out.append((pinch - closest).norm(dim=-1))
        return torch.stack(out, dim=-1)

    def _gw_engage(self, env_i: int, hi: int, s: int, hp, hq, gap) -> None:
        from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

        name, obj = self._gw_sites[s][0], self._gw_sites[s][1]
        rel_p = quat_apply_inverse(hq.unsqueeze(0), (obj.data.root_pos_w[env_i] - hp).unsqueeze(0))[0]
        rel_q = quat_mul(quat_conjugate(hq.unsqueeze(0)), obj.data.root_quat_w[env_i].unsqueeze(0))[0]
        if not self._gw_set_joint(env_i, hi, s, rel_p, rel_q):
            return
        self._gw_rel_p[env_i, hi, s] = rel_p
        self._gw_rel_q[env_i, hi, s] = rel_q
        self.grasp_held[env_i, hi, s] = True
        self._gw_count[env_i, hi] = 0
        print(f"[grasp-weld] env {env_i} hand {hi}: GRIPPED {name} "
              f"(aperture {float(gap) * 1000:.1f} mm)", flush=True)

    def _gw_set_joint(self, env_i: int, hi: int, s: int, rel_p, rel_q) -> bool:
        from pxr import Gf, UsdPhysics

        k = self._gw_pool_i[env_i][hi][s]
        if k >= self.GRASP_POOL:
            if (env_i, hi, s) not in self._gw_pool_warned:
                self._gw_pool_warned.add((env_i, hi, s))
                print(f"[grasp-weld] env {env_i} hand {hi}: pool dry for "
                      f"{self._gw_sites[s][0]} — no weld", flush=True)
            return False
        j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][hi][s][k])
        p, q = rel_p.tolist(), rel_q.tolist()
        j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
        j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        j.GetJointEnabledAttr().Set(True)
        return True

    def _gw_release(self, env_i: int, hi: int, s: int) -> None:
        from pxr import UsdPhysics

        k = self._gw_pool_i[env_i][hi][s]
        if k < self.GRASP_POOL:
            j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][hi][s][k])
            j.GetJointEnabledAttr().Set(False)
        self._gw_pool_i[env_i][hi][s] = k + 1
        self.grasp_held[env_i, hi, s] = False
        print(f"[grasp-weld] env {env_i} hand {hi}: RELEASED {self._gw_sites[s][0]}", flush=True)

    def _grasp_weld_release_all(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_gw_on", False) or not hasattr(self, "grasp_held"):
            return
        for row, hi, s in self.grasp_held[env_ids].nonzero(as_tuple=False).tolist():
            self._gw_release(int(env_ids[row]), hi, s)
        self._gw_count[env_ids] = 0

    def _grasp_weld_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        if not getattr(self, "_gw_on", False) or not hasattr(self, "grasp_held"):
            return {}
        return {
            "grasp_held": self.grasp_held[env_ids].clone(),
            "grasp_rel_p": self._gw_rel_p[env_ids].clone(),
            "grasp_rel_q": self._gw_rel_q[env_ids].clone(),
        }

    def _grasp_weld_restore(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        if not getattr(self, "_gw_on", False) or "grasp_held" not in state:
            return
        for row in range(len(env_ids)):
            i = int(env_ids[row])
            for hi in range(self.grasp_held.shape[1]):
                for s in range(len(self._gw_sites)):
                    if self.grasp_held[i, hi, s]:
                        self._gw_release(i, hi, s)
                    if bool(state["grasp_held"][row, hi, s]) and self._gw_set_joint(
                            i, hi, s, state["grasp_rel_p"][row, hi, s],
                            state["grasp_rel_q"][row, hi, s]):
                        self._gw_rel_p[i, hi, s] = state["grasp_rel_p"][row, hi, s]
                        self._gw_rel_q[i, hi, s] = state["grasp_rel_q"][row, hi, s]
                        self.grasp_held[i, hi, s] = True
        self._gw_count[env_ids] = 0
