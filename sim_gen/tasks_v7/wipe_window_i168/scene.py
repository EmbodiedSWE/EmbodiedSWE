"""FrostScrapeScene — knock every frost chip off the tilted pane into the trough.

Derived from the RoboVerse `pick_place/wipe_window` seed but STRATEGICALLY
DIFFERENT (see TASK.md): the seed's plan is to grasp a wiper bar and TRACE a
fixed Z-pattern of floating waypoints across a vertical window — pure held-tool
trajectory tracking, the glass is scenery and nothing in the world changes. Here
NOTHING is a trajectory and the glass carries real cargo: a pane tilted 15 deg
back from vertical holds 2-4 white FROST CHIPS, each resting on its own small
gray ledge (chip count, row height and slot positions all sampled per episode).
The chips lean into the glass (the 75 deg face is far steeper than the friction
angle), so each chip must be pushed SIDEWAYS along the pane until its weight
tips it off the ledge; it then tumbles down the glass and must be CAUGHT by the
green collection trough at the pane's base. Success is a physical outcome — every
present chip settled INSIDE the trough (below the rim, between the walls) — not
any motion pattern. Pressing chips INTO the glass (the seed's wiping contact)
moves nothing; a chip flicked over the trough onto the table does not count.
No execution order is required.

Mechanics (plain rigid bodies, no joints): pane / trough / ledges are kinematic
boxes; chips are light dynamic boxes. `post_step` owns the chips' wrench slots:
it applies the `drive_f` buffer (solve.py's stand-in for a fingertip push on a
chip's side edge) and updates the rubric latches.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - per present chip 0.25 * dislodge_latch: the chip left its rack (its centre
    dropped > `dislodge_drop` below the racked start) — partial credit an agent
    keeps even if the chip misses the trough;
  - per present chip 0.55 * contained_latch: the chip has sat inside the trough
    interior SLOWER than `contained_slow` for `streak_steps` consecutive
    substeps — a fast transit through the trough volume latches nothing;
  - 0.20 * success(): every present chip inside the trough right now, all
    settled. score == 1.0 iff success(); ~0 for the null policy; latched credit
    never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): chip COUNT (2-4), which
of the 4 slots are occupied, the row height on the pane, and per-slot lateral
jitter. Absent chips (and their ledges) park in an off-scene ground depot.

Heavy imports (isaaclab) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

N_SLOTS = 4


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FrostScrapeSceneCfg(BaseCfg):
    """Config for `FrostScrapeScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    dislodge_drop: float = tunable(0.08)  # chip centre this far below its racked start = off
    contained_slow: float = tunable(0.25)  # m/s: chip must be slower than this to latch
    streak_steps: int = tunable(12)  # consecutive contained+slow substeps before latching
    rim_margin: float = tunable(0.02)  # contained needs chip centre below rim by this
    settle_speed: float = tunable(0.05)  # max chip |lin vel| (m/s) when judging success

    # --- tunable: rubric weights (per present chip; + w_success on top) ----------------------
    w_dislodge: float = tunable(0.25)
    w_contained: float = tunable(0.55)
    w_success: float = tunable(0.20)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    min_chips: int = tunable(2)  # chip count sampled in {min_chips .. N_SLOTS}
    slot_jitter: float = tunable(0.012)  # per-slot lateral (y) jitter, +- this
    row_lo: float = tunable(0.22)  # chip row height along the pane face, from the bottom edge
    row_hi: float = tunable(0.44)

    # --- tunable: plant ----------------------------------------------------------------------
    chip_mass: float = tunable(0.03)
    chip_lin_damp: float = tunable(0.08)
    chip_ang_damp: float = tunable(0.08)

    # --- info: structure (env-local coordinates) ---------------------------------------------
    table_center: tuple = info((0.25, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.10, 0.08))  # top at z = 0.40
    theta_deg: float = info(15.0)  # pane lean (about +y, top tips away from the robot)
    pane_base_x: float = info(0.52)  # x of the pane's bottom-edge centre
    pane_lift: float = info(0.004)  # pane bottom edge sits this far above the table top
    pane_size: tuple = info((0.02, 0.68, 0.55))  # (thickness, width, length along the face)
    chip_size: tuple = info((0.012, 0.07, 0.06))  # (thickness, width, height) in pane frame
    ledge_size: tuple = info((0.016, 0.032, 0.010))  # protrusion, width, thickness
    chip_gap: float = info(0.0015)  # racked chip's back-face clearance off the glass
    slot_base_y: tuple = info((-0.225, -0.075, 0.075, 0.225))  # 4 slots, 0.15 pitch
    wall_setback: float = info(0.105)  # trough front wall centre, this far in front of pane_base_x
    wall_size: tuple = info((0.012, 0.72, 0.10))  # the trough front wall (rim at top)
    cap_size: tuple = info((0.13, 0.014, 0.10))  # trough end caps
    cap_y: float = info(0.353)
    depot_pos: tuple = info((-1.3, 1.0))  # off-scene ground depot for absent chips/ledges
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    table_top: float = field(default=None, init=False)
    theta: float = field(default=None, init=False)
    pane_center: tuple = field(default=None, init=False)
    pane_quat: tuple = field(default=None, init=False)  # wxyz, rot about +y by theta
    chip_xl: float = field(default=None, init=False)  # racked chip centre, pane-local x
    chip_zl_off: float = field(default=None, init=False)  # chip centre above slot z (pane-local)
    ledge_xl: float = field(default=None, init=False)
    rim_z: float = field(default=None, init=False)
    interior_x: tuple = field(default=None, init=False)  # trough interior bounds (env-local)
    interior_y: float = field(default=None, init=False)
    interior_z: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.table_top = self.table_center[2] + self.table_size[2] / 2
        self.theta = math.radians(self.theta_deg)
        half_len = self.pane_size[2] / 2
        bot_z = self.table_top + self.pane_lift
        self.pane_center = (self.pane_base_x + half_len * math.sin(self.theta), 0.0,
                            bot_z + half_len * math.cos(self.theta))
        self.pane_quat = (math.cos(self.theta / 2), 0.0, math.sin(self.theta / 2), 0.0)
        self.chip_xl = -(self.pane_size[0] / 2 + self.chip_size[0] / 2 + self.chip_gap)
        self.chip_zl_off = self.ledge_size[2] / 2 + 0.002 + self.chip_size[2] / 2
        self.ledge_xl = -(self.pane_size[0] / 2 + self.ledge_size[0] / 2)
        self.rim_z = self.table_top + self.wall_size[2]
        wall_inner = self.pane_base_x - self.wall_setback + self.wall_size[0] / 2
        self.interior_x = (wall_inner, self.pane_base_x + 0.012)
        self.interior_y = 0.33
        self.interior_z = (self.table_top - 0.02, self.rim_z - self.rim_margin)


# ----- scene -----------------------------------------------------------------------------------
class FrostScrapeScene(BaseScene):
    cfg: FrostScrapeSceneCfg

    def __init__(self, cfg: FrostScrapeSceneCfg | None = None) -> None:
        super().__init__(cfg or FrostScrapeSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + tilted pane + trough (front wall + end caps),
        4 kinematic ledges and 4 dynamic frost chips (reset re-poses everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        glass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.80, 0.92))
        frost = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.97, 0.97, 0.99))
        gray = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38))
        green = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.45, 0.15))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "pane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pane",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pane_size[0], c.pane_size[1], c.pane_size[2]),
                    rigid_props=kin, collision_props=coll, visual_material=glass),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=c.pane_center, rot=c.pane_quat),
            ),
            "wall": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TroughWall",
                spawn=sim_utils.CuboidCfg(
                    size=c.wall_size, rigid_props=kin, collision_props=coll,
                    visual_material=green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pane_base_x - c.wall_setback, 0.0,
                         c.table_top + c.wall_size[2] / 2)),
            ),
        }
        for side, sy in (("cap_l", -c.cap_y), ("cap_r", c.cap_y)):
            out[side] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("CapL" if sy < 0 else "CapR"),
                spawn=sim_utils.CuboidCfg(
                    size=c.cap_size, rigid_props=kin, collision_props=coll,
                    visual_material=green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pane_base_x - c.wall_setback + 0.055, sy,
                         c.table_top + c.cap_size[2] / 2)),
            )
        for i in range(N_SLOTS):
            out[f"ledge{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Ledge{i}",
                spawn=sim_utils.CuboidCfg(
                    size=c.ledge_size, rigid_props=kin, collision_props=coll,
                    visual_material=gray),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.16 * i, c.depot_pos[1] + 0.3, 0.05)),
            )
            out[f"chip{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Chip{i}",
                spawn=sim_utils.CuboidCfg(
                    size=c.chip_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=c.chip_lin_damp,
                        angular_damping=c.chip_ang_damp,
                        # post_step drives chips with external wrenches, which do NOT
                        # wake a sleeping body — keep the plant live.
                        sleep_threshold=0.0,
                        stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.chip_mass),
                    collision_props=coll,
                    visual_material=frost),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0] + 0.16 * i, c.depot_pos[1], 0.04)),
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
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.chips: list[RigidObject] = [env.iscene[f"chip{i}"] for i in range(N_SLOTS)]
        self.ledges: list[RigidObject] = [env.iscene[f"ledge{i}"] for i in range(N_SLOTS)]
        self.env_origins = env.iscene.env_origins
        # Episode state.
        self.present = torch.ones(n, N_SLOTS, dtype=torch.bool, device=dev)
        self.slot_y = torch.zeros(n, N_SLOTS, device=dev)  # sampled slot y (env-local)
        self.row_s = torch.zeros(n, device=dev)  # sampled row height along the pane face
        self.start_z = torch.zeros(n, N_SLOTS, device=dev)  # racked chip centre z (env-local)
        self.dislodge_latch = torch.zeros(n, N_SLOTS, dtype=torch.bool, device=dev)
        self.contained_latch = torch.zeros(n, N_SLOTS, dtype=torch.bool, device=dev)
        self.cont_streak = torch.zeros(n, N_SLOTS, dtype=torch.long, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes + owns
        # the chips' wrench slots — never call set_external_force_and_torque directly).
        self.drive_f = torch.zeros(n, N_SLOTS, 3, device=dev)

    # ----- pane frame helper --------------------------------------------------------------------
    def _pane_to_world(self, xl: torch.Tensor, yl: torch.Tensor,
                       zl: torch.Tensor) -> torch.Tensor:
        """Pane-local (x=normal-out-back, y=across, z=up-the-face) -> env-local, (m, 3)."""
        c = self.cfg
        ct, st = math.cos(c.theta), math.sin(c.theta)
        out = torch.zeros(xl.shape[0], 3, device=xl.device)
        out[:, 0] = c.pane_center[0] + xl * ct + zl * st
        out[:, 1] = c.pane_center[1] + yl
        out[:, 2] = c.pane_center[2] - xl * st + zl * ct
        return out

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample chip count (2..4), which slots are occupied, the row height
        and per-slot lateral jitter; rack present chips on their ledges flush against the
        glass; park absent chips + ledges in the depot; clear latches and drives. Uses
        torch.rand throughout (the first randint after manual_seed is degenerate here)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        k = (c.min_chips
             + torch.floor(torch.rand(m, device=dev)
                           * (N_SLOTS - c.min_chips + 1)).long().clamp(max=N_SLOTS - c.min_chips))
        rank = torch.rand(m, N_SLOTS, device=dev).argsort(dim=1).argsort(dim=1)
        present = rank < k.unsqueeze(1)
        s_row = c.row_lo + torch.rand(m, device=dev) * (c.row_hi - c.row_lo)
        base_y = torch.tensor(c.slot_base_y, device=dev).unsqueeze(0).expand(m, N_SLOTS)
        y = base_y + (torch.rand(m, N_SLOTS, device=dev) * 2 - 1) * c.slot_jitter

        self.present[env_ids] = present
        self.slot_y[env_ids] = y
        self.row_s[env_ids] = s_row
        self.dislodge_latch[env_ids] = False
        self.contained_latch[env_ids] = False
        self.cont_streak[env_ids] = 0
        self.drive_f[env_ids] = 0.0

        zl_row = s_row - c.pane_size[2] / 2  # slot centre in pane-local z
        for i in range(N_SLOTS):
            pres = present[:, i].unsqueeze(1)
            # ledge pose (kinematic; racked or depot)
            lp = self._pane_to_world(torch.full((m,), c.ledge_xl, device=dev),
                                     y[:, i], zl_row)
            lq = torch.tensor(c.pane_quat, device=dev).unsqueeze(0).expand(m, 4)
            park_l = torch.zeros(m, 3, device=dev)
            park_l[:, 0] = c.depot_pos[0] + 0.16 * i
            park_l[:, 1] = c.depot_pos[1] + 0.3
            park_l[:, 2] = 0.05
            idq = torch.zeros(m, 4, device=dev)
            idq[:, 0] = 1.0
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, lp, park_l)
            st[:, 3:7] = torch.where(pres, lq, idq)
            self.ledges[i].write_root_state_to_sim(st, env_ids)
            # chip pose (dynamic; racked flush against the glass, bottom edge on the ledge)
            cp = self._pane_to_world(torch.full((m,), c.chip_xl, device=dev),
                                     y[:, i], zl_row + c.chip_zl_off)
            park_c = torch.zeros(m, 3, device=dev)
            park_c[:, 0] = c.depot_pos[0] + 0.16 * i
            park_c[:, 1] = c.depot_pos[1]
            park_c[:, 2] = 0.04
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, cp, park_c)
            st[:, 3:7] = torch.where(pres, lq, idq)
            self.chips[i].write_root_state_to_sim(st, env_ids)
            self.start_z[env_ids, i] = torch.where(pres.squeeze(1), cp[:, 2], park_c[:, 2])

    # ----- readings -----------------------------------------------------------------------------
    def chip_pos(self) -> torch.Tensor:
        """(N, 4, 3) chip centres, env-local."""
        pos = torch.stack([b.data.root_pos_w for b in self.chips], dim=1)
        return pos - self.env_origins.unsqueeze(1)

    def chip_speed(self) -> torch.Tensor:
        """(N, 4) chip |lin vel|."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.chips], dim=1)

    def contained_now(self) -> torch.Tensor:
        """(N, 4) bool, geometric: chip centre inside the trough interior — between the
        wall's inner face and the pane, between the end caps, and BELOW the rim by
        `rim_margin` (the containment-below-the-aperture rule: a chip resting on the rim
        or leaning outside the wall never counts)."""
        c = self.cfg
        p = self.chip_pos()
        return (p[:, :, 0] > c.interior_x[0]) & (p[:, :, 0] < c.interior_x[1]) \
            & (p[:, :, 1].abs() < c.interior_y) \
            & (p[:, :, 2] > c.interior_z[0]) & (p[:, :, 2] < c.interior_z[1])

    def settled(self) -> torch.Tensor:
        """(N,) bool: every present chip slower than `settle_speed`."""
        slow = self.chip_speed() < self.cfg.settle_speed
        return (slow | ~self.present).all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: every present chip inside the trough RIGHT NOW, all settled
        (current physical state — judged on where the chips actually rest)."""
        return (self.contained_now() | ~self.present).all(dim=1) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: mean over present chips of
        w_dislodge * dislodge_latch + w_contained * contained_latch, plus
        w_success * success(). ~0 for the null policy; exactly 1.0 iff success()
        (all latches are then mature); latched credit never evaporates."""
        c = self.cfg
        pres = self.present.float()
        n_pres = pres.sum(dim=1).clamp(min=1.0)
        per = (c.w_dislodge * self.dislodge_latch.float()
               + c.w_contained * self.contained_latch.float()) * pres
        return per.sum(dim=1) / n_pres + c.w_success * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Chip plant: consume the `drive_f` buffer (owns the chips' wrench slots), then
        update the rubric latches: dislodge on a real drop below the racked start, and
        containment only through the slow-gate streak (a fast transit latches nothing).
        A diverged substep must not latch — NaN comparisons are False, garbage earns 0."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        for i in range(N_SLOTS):
            f = torch.zeros(n, 1, 3, device=dev)
            f[:, 0, :] = self.drive_f[:, i, :]
            self.chips[i].set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=dev))

        pos = self.chip_pos()
        finite = torch.isfinite(pos).all(dim=-1)
        drop = self.start_z - pos[:, :, 2]
        self.dislodge_latch |= (drop > c.dislodge_drop) & finite & self.present
        good = self.contained_now() & (self.chip_speed() < c.contained_slow) & finite
        self.cont_streak = torch.where(good, self.cont_streak + 1,
                                       torch.zeros_like(self.cont_streak))
        self.contained_latch |= (self.cont_streak >= c.streak_steps) & self.present

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {kk: getattr(self, kk)[env_ids].clone()
                     for kk in ("present", "slot_y", "row_s", "start_z", "dislodge_latch",
                                "contained_latch", "cont_streak", "drive_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for kk, v in state["task"].items():
            getattr(self, kk)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        out = {}
        for i in range(N_SLOTS):
            out[f"chip{i}"] = self.chips[i]
            out[f"ledge{i}"] = self.ledges[i]
        return out

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pale-blue glass pane stands on a table in front of you, leaning "
            f"{c.theta_deg:.0f} degrees away from you (its face is a steep slope). Stuck to "
            f"the face, in one horizontal row, sit between 2 and 4 square white FROST CHIPS "
            f"({c.chip_size[1] * 100:.0f} x {c.chip_size[2] * 100:.0f} cm, about 1 cm thick) — "
            f"count what you see. Each chip leans against the glass and rests on its own "
            f"small gray LEDGE, a stub much narrower than the chip. The row's height on the "
            f"pane and the chips' side-to-side positions change every episode. Along the base "
            f"of the pane runs an open GREEN COLLECTION TROUGH: a channel between the glass "
            f"and a low green wall about {c.wall_size[2] * 100:.0f} cm high.\n"
            f"Goal: get EVERY frost chip into the trough. Push a chip SIDEWAYS along the "
            f"glass (either direction, a few centimetres) until it slips off its narrow "
            f"ledge; it then slides down the pane and drops into the trough on its own. "
            f"Pressing a chip straight INTO the glass does nothing — the ledge and the glass "
            f"hold it. A chip only counts when it lies INSIDE the trough, below the green "
            f"rim; a chip knocked over the wall onto the table, or left resting anywhere "
            f"outside the trough, does not count and must be placed into the trough. Any "
            f"chip order works. The task is complete when all chips rest inside the trough "
            f"at the same time, everything still."
        )

    def instruction(self) -> str:
        return (
            "Push each white frost chip sideways off its ledge so it slides down the tilted "
            "pane into the green trough at its base. Finish with every chip resting inside "
            "the trough; a chip left anywhere outside the trough fails the task."
        )


# Guarded registration: the forge may import this module under two names.
if "frost_scrape" not in SCENES.list():
    SCENES.register("frost_scrape", FrostScrapeScene)
if "simgen.frost_scrape" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="frost_scrape", robot="null"))
