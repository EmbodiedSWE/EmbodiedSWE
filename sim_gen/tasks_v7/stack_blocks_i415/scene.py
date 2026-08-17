"""GradeBeaconScene — build a support pillar to a MEASURED grade and set the beacon
level with the collar (sim_gen task `stack_blocks_i415`).

Derived from rlbench/stack_blocks, but STRATEGICALLY different: the seed is
repetitive color-selection — pick the four RED cubes (all identical 50 mm) out of
distractors and pile them on a marked plane; the goal is the stack itself, any
height, judged by count/containment. Here NO block is a goal object and no fixed
number of blocks is ever "the answer". A kinematic MAST carries an amber COLLAR
band whose height is RANDOMIZED every episode (5 possible grades); the goal is to
raise the red BEACON cube over the build pad until its CENTER is level with the
collar, resting stably on a support the solver must CONSTRUCT from four gray
blocks of four DIFFERENT heights (30/45/60/75 mm). Which blocks to use — and how
many — is episode-dependent height ARITHMETIC (an exact-subset-sum read off a
measured spec), not a fixed pick-and-place loop: a solver needs a different plan
(read the grade, choose a subset, build, crown) and different code structure (a
grade-alignment + support predicate instead of a count-on-plane test). No single
block (upright OR lying: 30/45/55/60/75 mm) lands within the +/-10 mm band of any
grade, so at least a 2-block pillar must be composed; over-building past the
collar also fails.

Geometry (procedural): ground; kinematic mast (r 16 mm, h 420 mm) with a
VISUAL-ONLY amber collar band (NO collider — nothing can perch on it); kinematic
amber build pad (90x90x6 mm) 105 mm from the mast; four gray blocks, cross
section 55x55 mm, heights {30,45,60,75} mm (lighter gray = shorter); red beacon
cube 45 mm. Blocks + beacon scatter on a 340 mm circle around the rig.

success(): beacon center within `col_r` of the pad center (xy), within `band` of
the collar center height, SUPPORTED by an in-column block whose top is at the
beacon's bottom (`support_tol` — a beacon merely held in the air at grade fails),
and the state held still for `settle_steps` SUSTAINED substeps (counter in
post_step).

score(): latched `build_credit` * (pillar-top / grade, counted only while the
pillar does not overshoot grade + band and the column is settled) +
`place_credit` latched at first success; 1.0 iff success() now. Null policy ~0
(nothing spawns in the pad column).

Per-episode randomization (readback-verified in smoke): grade index (5 values),
rig position jitter + free rig yaw (the pad orbits the mast), a 5-slot scatter
permutation for the 4 blocks + beacon with per-body jitter + free yaw. Heavy
imports (isaaclab) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

BLOCK_HS: tuple[float, ...] = (0.030, 0.045, 0.060, 0.075)
BLOCK_GRAYS: tuple[float, ...] = (0.78, 0.62, 0.46, 0.30)  # lighter = shorter
AMBER = (0.95, 0.65, 0.05)


def subset_for(target_h: float, heights: tuple[float, ...] = BLOCK_HS) -> tuple[int, ...]:
    """Indices of the smallest block subset whose UPRIGHT heights sum exactly to
    `target_h` (integer-mm arithmetic — float sums like 0.030+0.075 miss 0.105).
    Raises if no exact subset exists."""
    t_mm = round(target_h * 1000)
    hs_mm = [round(h * 1000) for h in heights]
    for k in range(1, len(heights) + 1):
        for combo in combinations(range(len(heights)), k):
            if sum(hs_mm[i] for i in combo) == t_mm:
                return combo
    raise ValueError(f"no exact block subset sums to {t_mm} mm")


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GradeBeaconSceneCfg(BaseCfg):
    """Config for `GradeBeaconScene`. Rubric honesty is asserted in `__post_init__`:
    every grade is an exact upright-subset sum; no single block (upright or lying)
    reaches any grade's band; neighboring grades sit outside each other's band; a
    centered block always reads in-column; the scatter ring clears the rig."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    band: float = tunable(0.010)  # |beacon center z - collar center z| tolerance (m)
    col_r: float = tunable(0.048)  # pad-column radius for in-column tests (m)
    support_tol: float = tunable(0.008)  # |beacon bottom - pillar top| at support (m)
    settle_lin: float = tunable(0.08)  # max |lin vel| at judging (m/s)
    settle_ang: float = tunable(1.50)  # max |ang vel| at judging (rad/s) — above the
    # GPU resting-contact phantom jitter (~0.5 rad/s on the crowned stack, position
    # dead-stable); real tumbles/kicks read far higher and still reset the counter
    settle_steps: int = tunable(20)  # substeps of SUSTAINED stillness
    build_credit: float = tunable(0.4)  # latched pillar-progress credit weight
    place_credit: float = tunable(0.4)  # latched first-success credit weight

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    grade_min: float = tunable(0.090)  # lowest grade: pillar height above the pad (m)
    grade_step: float = tunable(0.015)  # grade quantum (m)
    n_grades: int = tunable(5)  # grades sampled uniformly from the 5-value ladder
    rig_x: float = tunable(0.25)  # nominal rig (mast) center x
    rig_jitter: float = tunable(0.03)  # uniform +/- rig xy jitter
    yaw_deg: float = tunable(180.0)  # free rig yaw AND free per-body yaw (+/-)
    scatter_r: float = tunable(0.34)  # scatter-slot circle radius around the rig
    scatter_jitter: float = tunable(0.02)  # per-body slot xy jitter

    # --- info: structure ---------------------------------------------------------------------
    mast_r: float = info(0.016)
    mast_h: float = info(0.42)
    mast_gap: float = info(0.115)  # mast center -> pad center distance
    pad_s: float = info(0.090)  # pad side
    pad_t: float = info(0.006)  # pad thickness (pad top = build datum)
    collar_r: float = info(0.019)
    collar_h: float = info(0.024)  # visual band: covers the +/- band at 12 mm half-height
    block_s: float = info(0.055)  # block cross section (both lateral sides)
    block_mass: float = info(0.08)
    beacon_s: float = info(0.045)
    beacon_mass: float = info(0.05)
    contact_offset: float = info(0.002)

    def __post_init__(self) -> None:
        c = self
        grades_mm = [round((c.grade_min + k * c.grade_step) * 1000) for k in range(c.n_grades)]
        hs_mm = [round(h * 1000) for h in BLOCK_HS]
        band_mm = round(c.band * 1000)
        # -- every grade is an exact upright-subset sum (>= 2 blocks) --
        for g in grades_mm:
            combo = subset_for(g / 1000.0)
            assert len(combo) >= 2, "every grade must NEED a composed pillar"
        # -- no single block (upright or lying flat: 55 mm) reaches any band --
        singles = set(hs_mm) | {round(c.block_s * 1000)}
        for g in grades_mm:
            for s in singles:
                assert abs(g - s) > band_mm + 3, \
                    f"single block {s} mm must miss grade {g} mm by > band"
        # -- a neighboring grade's exact pillar is rejected --
        assert round(c.grade_step * 1000) > band_mm + 3, \
            "adjacent grades must sit outside each other's band"
        # -- the collar band visualizes the tolerance --
        assert c.collar_h / 2 >= c.band, "collar must cover the +/- band"
        assert c.mast_h > c.pad_t + grades_mm[-1] / 1000.0 + c.beacon_s / 2 + c.collar_h, \
            "mast must out-reach the highest collar"
        # -- in-column reads are honest --
        assert c.col_r > c.block_s * math.sqrt(2) / 2 + 0.004, \
            "a centered block must read in-column at any yaw"
        assert c.col_r > c.beacon_s * math.sqrt(2) / 2 + 0.004
        # -- support gate is meaningful --
        assert c.support_tol < c.band, "support tolerance tighter than the band"
        # -- rig self-clearance: pad corner clears the mast --
        assert c.mast_gap - c.pad_s * math.sqrt(2) / 2 > c.mast_r + 0.008, \
            "pad must clear the mast"
        assert c.mast_gap - c.col_r > c.mast_r + c.block_s * math.sqrt(2) / 2 + 0.004, \
            "an in-column block must never touch the mast"
        # -- scatter ring clears the whole rig (any rig yaw) --
        assert (c.scatter_r - c.scatter_jitter - c.block_s * math.sqrt(2) / 2
                > c.mast_gap + c.col_r + 0.02), "scatter slots must clear the rig"
        assert 2 * c.scatter_r * math.sin(math.pi / 5) > \
            2 * (c.block_s * math.sqrt(2) / 2 + c.scatter_jitter) + 0.01, \
            "adjacent scatter slots must not overlap"
        # -- embodiment: everything fits an 80 mm jaw --
        assert max(c.block_s, c.beacon_s) <= 0.078
        # -- credit / settle sanity --
        assert c.settle_steps >= 12
        assert 0.0 < c.build_credit < 1.0 and 0.0 < c.place_credit < 1.0
        assert c.build_credit + c.place_credit <= 0.85


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("grade_beacon")
class GradeBeaconScene(BaseScene):
    cfg: GradeBeaconSceneCfg

    def __init__(self, cfg: GradeBeaconSceneCfg | None = None) -> None:
        super().__init__(cfg or GradeBeaconSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        rigid = dict(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5,
            linear_damping=0.10,
            angular_damping=0.50,
        )
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9, dynamic_friction=0.8, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        def kin_box(name: str, size, rgb, z: float, collide: bool = True):
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=coll if collide else None,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    physics_material=mat if collide else None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z)),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # kinematic mast (re-posed per reset)
            "mast": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mast",
                spawn=sim_utils.CylinderCfg(
                    radius=c.mast_r, height=c.mast_h, axis="Z",
                    collision_props=coll,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.35, 0.40)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rig_x, 0.0, c.mast_h / 2)),
            ),
            # VISUAL-ONLY collar band: NO collider — nothing can rest on it
            "collar": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Collar",
                spawn=sim_utils.CylinderCfg(
                    radius=c.collar_r, height=c.collar_h, axis="Z",
                    collision_props=None,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=AMBER),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rig_x, 0.0, 0.12)),
            ),
            # kinematic amber build pad
            "pad": kin_box("Pad", (c.pad_s, c.pad_s, c.pad_t), AMBER, c.pad_t / 2),
        }
        for i, h in enumerate(BLOCK_HS):
            g = BLOCK_GRAYS[i]
            out[f"block_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Block{round(h * 1000)}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.block_s, c.block_s, h),
                    collision_props=coll,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**rigid),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(g, g, g)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.2 - 0.12 * i, 0.0, h / 2 + 0.003)),
            )
        out["beacon"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Beacon",
            spawn=sim_utils.CuboidCfg(
                size=(c.beacon_s, c.beacon_s, c.beacon_s),
                collision_props=coll,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**rigid),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.beacon_mass),
                physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.08, 0.08)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, 0.2, c.beacon_s / 2 + 0.003)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.blocks: list[RigidObject] = [env.iscene[f"block_{i}"] for i in range(len(BLOCK_HS))]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.mast: RigidObject = env.iscene["mast"]
        self.collar: RigidObject = env.iscene["collar"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        # per-episode spec (set at reset)
        self.grade = torch.full((n,), self.cfg.grade_min, device=dev)  # pillar height spec
        self.pad_xy = torch.zeros(n, 2, device=dev)  # pad center, world
        # latches / counters (post_step)
        self.build_latch = torch.zeros(n, device=dev)
        self.place_latch = torch.zeros(n, device=dev)
        self.col_still = torch.zeros(n, device=dev)
        self.hold_count = torch.zeros(n, device=dev)
        self.bad_streak = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the grade (collar height), rig pose (xy jitter +
        free yaw — the pad orbits the mast), and a 5-slot scatter permutation for
        the 4 blocks + beacon (per-body jitter + free yaw); latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.yaw_deg)

        rig = torch.stack([
            c.rig_x + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter,
            (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter,
        ], dim=1)
        rig_yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
        grade_i = torch.randint(0, c.n_grades, (m,), device=dev)
        grade = c.grade_min + grade_i.float() * c.grade_step
        self.grade[env_ids] = grade
        pad_xy = rig + c.mast_gap * torch.stack(
            [torch.cos(rig_yaw), torch.sin(rig_yaw)], dim=1)
        self.pad_xy[env_ids] = pad_xy + origin[:, 0:2]

        def write(body, xy: torch.Tensor, z: torch.Tensor, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros = torch.zeros(m, device=dev)
        write(self.mast, rig, zeros + c.mast_h / 2, zeros)
        write(self.collar, rig, c.pad_t + grade + c.beacon_s / 2, zeros)
        write(self.pad, pad_xy, zeros + c.pad_t / 2, rig_yaw)

        # scatter: 5 slots on a circle around the rig, permuted over 4 blocks + beacon
        phi0 = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        perm = torch.rand(m, 5, device=dev).argsort(dim=1)
        movers = [*self.blocks, self.beacon]
        half_hs = [h / 2 for h in BLOCK_HS] + [c.beacon_s / 2]
        for i, body in enumerate(movers):
            ang = phi0 + perm[:, i].float() * (2 * math.pi / 5)
            xy = rig + c.scatter_r * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
            xy = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp
            write(body, xy, zeros + half_hs[i] + 0.003, yaw)

        self.build_latch[env_ids] = 0.0
        self.place_latch[env_ids] = 0.0
        self.col_still[env_ids] = 0.0
        self.hold_count[env_ids] = 0.0
        self.bad_streak[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = [*self.blocks, self.beacon, self.mast, self.collar, self.pad]
        return {
            "bodies": [b.data.root_state_w[env_ids].clone() for b in bodies],
            "grade": self.grade[env_ids].clone(),
            "pad_xy": self.pad_xy[env_ids].clone(),
            "build_latch": self.build_latch[env_ids].clone(),
            "place_latch": self.place_latch[env_ids].clone(),
            "col_still": self.col_still[env_ids].clone(),
            "hold_count": self.hold_count[env_ids].clone(),
            "bad_streak": self.bad_streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = [*self.blocks, self.beacon, self.mast, self.collar, self.pad]
        for b, st in zip(bodies, state["bodies"]):
            b.write_root_state_to_sim(st, env_ids)
        self.grade[env_ids] = state["grade"]
        self.pad_xy[env_ids] = state["pad_xy"]
        self.build_latch[env_ids] = state["build_latch"]
        self.place_latch[env_ids] = state["place_latch"]
        self.col_still[env_ids] = state["col_still"]
        self.hold_count[env_ids] = state["hold_count"]
        self.bad_streak[env_ids] = state["bad_streak"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        hs = "/".join(f"{round(h * 1000)}" for h in BLOCK_HS)
        return (
            "A gray vertical MAST stands on the floor with a bright AMBER COLLAR "
            "band around it; the collar's height on the mast CHANGES EVERY EPISODE "
            "(5 possible grades) — read it by looking. Next to the mast "
            f"({c.mast_gap * 1000:.0f} mm away; the direction also changes with the "
            "rig's heading every episode) lies a flat AMBER BUILD PAD "
            f"({c.pad_s * 1000:.0f} mm square, {c.pad_t * 1000:.0f} mm thick). "
            "Scattered on the floor around the rig are FOUR GRAY BLOCKS, all "
            f"{c.block_s * 1000:.0f} mm square in cross section but of four "
            f"DIFFERENT heights ({hs} mm — the lighter the gray, the shorter), and "
            f"one RED BEACON cube ({c.beacon_s * 1000:.0f} mm).\n"
            "Goal: raise the beacon over the build pad to exactly the collar's "
            "grade — its CENTER level with the collar band center within "
            f"+/-{c.band * 1000:.0f} mm — by first building a support pillar on the "
            "pad from whichever blocks reach that height, then setting the beacon "
            "on top. The beacon must rest stably ON the built support, inside the "
            "pad column, with everything still. Which blocks (and how many) are "
            "needed depends on the episode's collar height; no single block is "
            "ever tall enough.\n"
            "What does NOT count: the beacon on the floor, on a single block, or "
            "anywhere below/above the band; a pillar built past the collar; the "
            "beacon at the right height but OFF the pad column (e.g. beside the "
            "rig or against the mast); the beacon merely held in the air at the "
            "collar with no built support under it."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Look at the height of the amber collar on the mast, build a pillar on "
            "the amber pad from the gray blocks so it reaches that grade, and set "
            "the red beacon on top so its center is level with the collar band. "
            "The beacon on the floor, on a too-short or too-tall pillar, or off "
            "the pad does not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _top_z(self, body, sx: float, sy: float, sz: float) -> torch.Tensor:
        """(N,): world z of the body's oriented-box TOP face (support function:
        half extent = 0.5 * sum_k |R e_k . z| * s_k)."""
        from isaaclab.utils.math import quat_apply

        q = body.data.root_quat_w
        n = q.shape[0]
        half = torch.zeros(n, device=q.device)
        for s, ax in ((sx, (1.0, 0.0, 0.0)), (sy, (0.0, 1.0, 0.0)), (sz, (0.0, 0.0, 1.0))):
            e = torch.tensor(ax, device=q.device).expand(n, 3)
            half = half + 0.5 * s * quat_apply(q, e)[:, 2].abs()
        return body.data.root_pos_w[:, 2] + half

    def block_tops(self) -> torch.Tensor:
        """(N,B): world top-face z of each block."""
        c = self.cfg
        return torch.stack(
            [self._top_z(b, c.block_s, c.block_s, h)
             for b, h in zip(self.blocks, BLOCK_HS)], dim=1)

    def block_in_col(self) -> torch.Tensor:
        """(N,B) bool: block center xy within `col_r` of the pad center."""
        bx = torch.stack([b.data.root_pos_w[:, 0:2] for b in self.blocks], dim=1)
        return (bx - self.pad_xy[:, None, :]).norm(dim=-1) < self.cfg.col_r

    def pillar_top(self) -> torch.Tensor:
        """(N,): highest in-column block top ABOVE THE PAD TOP; 0 with no block."""
        c = self.cfg
        pad_top = self.env_origins[:, 2] + c.pad_t
        rel = self.block_tops() - pad_top[:, None]
        return torch.where(self.block_in_col(), rel, torch.zeros_like(rel)).max(dim=1).values

    def beacon_dz(self) -> torch.Tensor:
        """(N,): beacon center z MINUS the collar (grade) center z."""
        c = self.cfg
        z_t = self.env_origins[:, 2] + c.pad_t + self.grade + c.beacon_s / 2
        return self.beacon.data.root_pos_w[:, 2] - z_t

    def beacon_in_col(self) -> torch.Tensor:
        """(N,) bool: beacon center xy within `col_r` of the pad center."""
        return (self.beacon.data.root_pos_w[:, 0:2] - self.pad_xy).norm(dim=-1) \
            < self.cfg.col_r

    def supported(self) -> torch.Tensor:
        """(N,) bool: some in-column block's TOP is at the beacon's BOTTOM
        (within `support_tol`) — the beacon rests on a BUILT support, it is not
        merely held in the air at grade."""
        c = self.cfg
        bottom = self.beacon.data.root_pos_w[:, 2] - (
            self._top_z(self.beacon, c.beacon_s, c.beacon_s, c.beacon_s)
            - self.beacon.data.root_pos_w[:, 2])
        gap = bottom[:, None] - self.block_tops()
        near = gap.abs() <= c.support_tol
        return (near & self.block_in_col()).any(dim=1)

    def in_band(self) -> torch.Tensor:
        """(N,) bool: beacon center within `band` of the collar center height."""
        return self.beacon_dz().abs() <= self.cfg.band

    def _still_now(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(col_ok, all_ok): in-column blocks all still now; that AND beacon still."""
        c = self.cfg
        bv = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.blocks], dim=1)
        bw = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.blocks], dim=1)
        blk_still = (bv < c.settle_lin) & (bw < c.settle_ang)
        col_ok = (blk_still | ~self.block_in_col()).all(dim=1)
        bec = (self.beacon.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.beacon.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
        return col_ok, col_ok & bec

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: run the settled-column and correct-hold
        counters; latch build progress (only while the column is settled and NOT
        overshot); latch the place credit at success."""
        c = self.cfg
        col_ok, all_ok = self._still_now()
        self.col_still = torch.where(col_ok, self.col_still + 1.0,
                                     torch.zeros_like(self.col_still))
        top = self.pillar_top()
        ratio = (top / self.grade).clamp(0.0, 1.0)
        can = (self.col_still >= float(c.settle_steps)) & (top <= self.grade + c.band)
        self.build_latch = torch.where(can, torch.maximum(self.build_latch, ratio),
                                       self.build_latch)
        ok_now = self.in_band() & self.beacon_in_col() & self.supported() & all_ok
        # isolated (<3-substep) resting-contact noise spikes stall — not reset — the
        # hold counter; any SUSTAINED disturbance (a kick, a toss, a collapse) resets
        # it within 3 substeps.
        self.bad_streak = torch.where(ok_now, torch.zeros_like(self.bad_streak),
                                      self.bad_streak + 1.0)
        self.hold_count = torch.where(
            ok_now, self.hold_count + 1.0,
            torch.where(self.bad_streak >= 3.0, torch.zeros_like(self.hold_count),
                        self.hold_count))
        self.place_latch = torch.maximum(
            self.place_latch, (self.hold_count >= float(c.settle_steps)).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beacon at grade (center within `band` of the collar center),
        inside the pad column, resting on a built in-column support, held still
        for `settle_steps` SUSTAINED substeps."""
        return self.hold_count >= float(self.cfg.settle_steps)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: `build_credit` * latched pillar progress +
        `place_credit` * latched first-success; exactly 1.0 iff success() now.
        Latched — credit never evaporates, so phase scores are non-decreasing.
        Null policy ~0 (nothing spawns in the pad column)."""
        c = self.cfg
        partial = (c.build_credit * self.build_latch
                   + c.place_credit * self.place_latch).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(partial), partial)


register_env("simgen", lambda: EnvCfg(scene="grade_beacon", robot="null", env_spacing=3.0))
