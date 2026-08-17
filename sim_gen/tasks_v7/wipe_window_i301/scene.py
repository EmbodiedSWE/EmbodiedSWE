"""GlazeWindowScene — glaze the empty window: uncap the frame, seat the pane, recap.

Derived from the RoboVerse `pick_place/wipe_window` seed but STRATEGICALLY DIFFERENT
(see TASK.md): the seed grasps a wiper bar and TRACES a fixed Z-pattern of six floating
waypoints across an already-installed window — pure held-tool trajectory tracking; the
glass is scenery and no object's final state matters. Here there is NO trajectory and NO
installed glass: the window frame stands EMPTY and the episode is an ordered ASSEMBLY.
A light-blue glass pane waits upright in a slotted stand on the near side of the table;
the frame's top opening is covered by a loose CAP LID resting in a shallow nub tray on
the post tops. The solver must (1) lift the cap off and set it aside, (2) slide the pane
out of its stand, lower it down through the top opening between the posts and SEAT it in
the sill groove (a 22 mm channel between two rails — a real insertion under contact),
and (3) rest the cap back in its tray, roofing the glazed frame. Success is the settled
final CONFIGURATION: pane seated in the groove AND cap seated in its tray.

The order is physically enforced by geometry, not by fiat: the cap covers the only
insertion corridor (the top), and the two fixed MULLION bars across the frame at
mid-height close the front/back "letterbox" — a pane threaded from the side must either
cross a mullion (blocked while leaning < ~10 deg) or swing its top above the cap plane
(blocked while leaning enough to clear the mullion), so with the cap on the groove is
unreachable (smoke constructs the attempt and shows the stall). Wiping/pressing motions
against the pane (the seed's contact) earn nothing.

Mechanics: every frame part, the table and the stand are kinematic boxes re-posed per
episode; the PANE and the CAP are the only dynamic bodies. `post_step` owns their wrench
slots: it applies the `drive_f` buffer (solve.py's stand-in for a fingertip/pinch
contact; world-frame semantics, rotated into the body frame before application) and
updates the rubric latches.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.20 cap_off latch: the cap has been carried clear of the frame top (frame-local
    horizontal distance > `cap_off_dist`); also granted retroactively once the pane is
    seated (a seated pane physically implies the corridor was open);
  - 0.40 pane_seated latch: the pane's bottom edge has rested in the sill groove
    (in-groove x, in-span y, bottom at floor height, near-upright) slower than
    `latch_slow` for `streak_steps` consecutive substeps;
  - 0.25 capped latch: the cap sits in its tray WHILE the pane is seated (recapping an
    empty frame latches nothing), same streak gate;
  - 0.15 success() live: pane seated AND cap seated AND both bodies settled.
  score == 1.0 iff success() (after settling, all latches are then mature); ~0 for the
  null policy; latched credit never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): the frame's table position AND
yaw, and the stand's position AND yaw — every transport target and every insertion axis
moves between episodes.

Heavy imports (isaaclab) are deferred so importing this module — and registering the
scene — stays app-free.
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

# drive_f rows
PANE, CAP = 0, 1


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GlazeWindowSceneCfg(BaseCfg):
    """Config for `GlazeWindowScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_x_tol: float = tunable(0.008)  # pane bottom, frame-local |x| (groove admits +-5 mm)
    seat_y_tol: float = tunable(0.030)  # pane bottom, frame-local |y| (posts admit +-10 mm)
    seat_z_tol: float = tunable(0.008)  # pane bottom height error off the groove floor
    seat_up_min_deg: float = tunable(25.0)  # pane axis within this of world-up (geometry
    # caps the in-groove lean at ~11 deg — the mullions catch a leaning pane first)
    cap_xy_tol: tuple = tunable((0.010, 0.015))  # cap centre, frame-local (tray admits 5-8 mm)
    cap_z_tol: float = tunable(0.008)
    cap_up_min_deg: float = tunable(10.0)  # cap must lie LEVEL in the tray
    cap_off_dist: float = tunable(0.14)  # frame-local horizontal distance = carried clear
    latch_slow: float = tunable(0.10)  # m/s: body must be slower than this to run a streak
    streak_steps: int = tunable(12)  # consecutive good substeps before latching
    settle_speed: float = tunable(0.05)  # max |lin vel| (pane AND cap) when judging success

    # --- tunable: rubric weights -------------------------------------------------------------
    w_cap_off: float = tunable(0.20)
    w_seated: float = tunable(0.40)
    w_capped: float = tunable(0.25)
    w_success: float = tunable(0.15)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    frame_x: tuple = tunable((0.42, 0.52))  # frame centre x range on the table
    frame_y: tuple = tunable((-0.10, 0.10))
    frame_yaw: float = tunable(0.22)  # +- rad
    stand_x: tuple = tunable((0.14, 0.22))  # pane stand centre x range
    stand_y: tuple = tunable((-0.16, 0.16))
    stand_yaw: float = tunable(0.50)  # +- rad

    # --- tunable: plant ----------------------------------------------------------------------
    pane_mass: float = tunable(0.15)
    cap_mass: float = tunable(0.12)
    pane_lin_damp: float = tunable(0.05)
    pane_ang_damp: float = tunable(0.20)  # a 12 mm plate falling guided through a slot rings
    cap_lin_damp: float = tunable(0.05)
    cap_ang_damp: float = tunable(0.10)

    # --- info: structure (env-local; frame parts in FRAME-LOCAL (x, y, z-above-table)) -------
    table_center: tuple = info((0.25, 0.0, 0.36))
    table_size: tuple = info((1.00, 1.10, 0.08))  # top at z = 0.40
    pane_size: tuple = info((0.012, 0.17, 0.20))  # thickness, width, height
    cap_size: tuple = info((0.16, 0.30, 0.015))
    # frame parts: name -> (size, frame-local centre offset). Groove: rails at +-0.018,
    # 14 mm thick -> 22 mm channel for the 12 mm pane; floor = sill top (z0+0.05).
    # Posts inner faces at y +-0.095 (the 0.17 m pane enters with +-10 mm slack), tops at
    # z0+0.28 carry the cap tray (4 x-stop nubs + 2 y-stop nubs, 5-8 mm slack). Mullions
    # (fixed bars front AND back at z0+0.15) close the side letterbox (module docstring).
    frame_parts: tuple = info((
        ("sill", (0.06, 0.34, 0.05), (0.0, 0.0, 0.025)),
        ("rail_f", (0.014, 0.19, 0.04), (-0.018, 0.0, 0.07)),
        ("rail_b", (0.014, 0.19, 0.04), (0.018, 0.0, 0.07)),
        ("post_l", (0.05, 0.06, 0.23), (0.0, -0.125, 0.165)),
        ("post_r", (0.05, 0.06, 0.23), (0.0, 0.125, 0.165)),
        ("mull_f", (0.012, 0.19, 0.02), (-0.030, 0.0, 0.15)),
        ("mull_b", (0.012, 0.19, 0.02), (0.030, 0.0, 0.15)),
        ("nub_fl", (0.012, 0.05, 0.02), (-0.094, -0.125, 0.29)),
        ("nub_fr", (0.012, 0.05, 0.02), (-0.094, 0.125, 0.29)),
        ("nub_bl", (0.012, 0.05, 0.02), (0.094, -0.125, 0.29)),
        ("nub_br", (0.012, 0.05, 0.02), (0.094, 0.125, 0.29)),
        ("nub_yl", (0.05, 0.012, 0.02), (0.0, -0.161, 0.29)),
        ("nub_yr", (0.05, 0.012, 0.02), (0.0, 0.161, 0.29)),
    ))
    # stand parts: name -> (size, stand-local centre offset). 22 mm slot for the pane.
    stand_parts: tuple = info((
        ("stand_base", (0.16, 0.09, 0.012), (0.0, 0.0, 0.006)),
        ("stand_rl", (0.16, 0.014, 0.03), (0.0, -0.018, 0.027)),
        ("stand_rr", (0.16, 0.014, 0.03), (0.0, 0.018, 0.027)),
    ))
    contact_offset: float = info(0.002)
    spawn_lift: float = info(0.002)  # dynamic bodies spawn this far above their rest contact

    # Derived (filled in __post_init__).
    table_top: float = field(default=None, init=False)
    groove_floor: float = field(default=None, init=False)  # above table top
    post_top: float = field(default=None, init=False)
    cap_rest_z: float = field(default=None, init=False)  # cap centre above table when seated
    pane_seat_z: float = field(default=None, init=False)  # pane centre above table when seated
    pane_stand_z: float = field(default=None, init=False)  # pane centre above table in the stand
    cap_top_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.table_top = self.table_center[2] + self.table_size[2] / 2
        self.groove_floor = 0.05  # sill top
        self.post_top = 0.28
        self.cap_rest_z = self.post_top + self.cap_size[2] / 2
        self.pane_seat_z = self.groove_floor + self.pane_size[2] / 2
        self.pane_stand_z = 0.012 + self.pane_size[2] / 2  # on the stand base
        self.cap_top_z = self.cap_rest_z + self.cap_size[2] / 2


# ----- scene -----------------------------------------------------------------------------------
class GlazeWindowScene(BaseScene):
    cfg: GlazeWindowSceneCfg

    def __init__(self, cfg: GlazeWindowSceneCfg | None = None) -> None:
        super().__init__(cfg or GlazeWindowSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + frame parts + stand parts (reset re-poses all of
        them per the sampled frame/stand pose), and the two dynamic bodies: pane and cap."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        frame_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.24))
        glass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.78, 0.95))
        cap_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.20, 0.10))
        stand_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.35))
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
        }
        for name, size, off in c.frame_parts:
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll,
                    visual_material=frame_col),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.47 + off[0], off[1], c.table_top + off[2])),
            )
        for name, size, off in c.stand_parts:
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=size, rigid_props=kin, collision_props=coll,
                    visual_material=stand_col),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.18 + off[0], off[1], c.table_top + off[2])),
            )
        dyn = dict(
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            # post_step drives these with external wrenches, which do NOT wake a
            # sleeping body — keep the plant live.
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        )
        out["pane"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pane",
            spawn=sim_utils.CuboidCfg(
                size=c.pane_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=c.pane_lin_damp, angular_damping=c.pane_ang_damp, **dyn),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.pane_mass),
                collision_props=coll, visual_material=glass),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.18, 0.0, c.table_top + c.pane_stand_z + c.spawn_lift),
                rot=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))),
        )
        out["cap"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cap",
            spawn=sim_utils.CuboidCfg(
                size=c.cap_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=c.cap_lin_damp, angular_damping=c.cap_ang_damp, **dyn),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cap_mass),
                collision_props=coll, visual_material=cap_col),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.47, 0.0, c.table_top + c.cap_rest_z + c.spawn_lift)),
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
        c = self.cfg
        self.pane: RigidObject = env.iscene["pane"]
        self.cap: RigidObject = env.iscene["cap"]
        self.kin_parts = {name: env.iscene[name]
                          for name, _s, _o in (*c.frame_parts, *c.stand_parts)}
        self.env_origins = env.iscene.env_origins
        # Episode state (sampled at reset).
        self.frame_xy = torch.zeros(n, 2, device=dev)
        self.frame_yaw = torch.zeros(n, device=dev)
        self.stand_xy = torch.zeros(n, 2, device=dev)
        self.stand_yaw = torch.zeros(n, device=dev)
        # Rubric latches + streaks.
        self.cap_off_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seated_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.capped_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seat_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self.cap_streak = torch.zeros(n, dtype=torch.long, device=dev)
        # External drive input, WORLD-frame semantics (solve.py and smoke probes write;
        # post_step consumes + owns the wrench slots — never call
        # set_external_force_and_torque directly). Rows: [PANE, CAP].
        self.drive_f = torch.zeros(n, 2, 3, device=dev)

    # ----- pose helpers -------------------------------------------------------------------------
    @staticmethod
    def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
        q[:, 0] = torch.cos(yaw / 2)
        q[:, 3] = torch.sin(yaw / 2)
        return q

    def _place_env(self, xy: torch.Tensor, yaw: torch.Tensor,
                   off: tuple) -> torch.Tensor:
        """Local offset -> env-local position under (xy, yaw) on the table, (m, 3)."""
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        out = torch.zeros(xy.shape[0], 3, device=xy.device)
        out[:, 0] = xy[:, 0] + off[0] * cy - off[1] * sy
        out[:, 1] = xy[:, 1] + off[0] * sy + off[1] * cy
        out[:, 2] = self.cfg.table_top + off[2]
        return out

    def to_frame(self, p_env: torch.Tensor) -> torch.Tensor:
        """Env-local points (N, 3) -> frame-local (x, y, z-above-table), all envs."""
        cy, sy = torch.cos(self.frame_yaw), torch.sin(self.frame_yaw)
        dx = p_env[:, 0] - self.frame_xy[:, 0]
        dy = p_env[:, 1] - self.frame_xy[:, 1]
        out = torch.zeros_like(p_env)
        out[:, 0] = cy * dx + sy * dy
        out[:, 1] = -sy * dx + cy * dy
        out[:, 2] = p_env[:, 2] - self.cfg.table_top
        return out

    def frame_point_env(self, off: tuple) -> torch.Tensor:
        """Frame-local offset -> env-local point, (N, 3)."""
        return self._place_env(self.frame_xy, self.frame_yaw, off)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample frame pose + stand pose, re-pose every kinematic part,
        stand the pane in its stand, rest the cap in its tray, clear latches and drives.
        Uses torch.rand with a burnt first draw (the first post-seed draw is degenerate)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, device=dev)  # burn the degenerate first post-seed draw

        fx = c.frame_x[0] + torch.rand(m, device=dev) * (c.frame_x[1] - c.frame_x[0])
        fy = c.frame_y[0] + torch.rand(m, device=dev) * (c.frame_y[1] - c.frame_y[0])
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * c.frame_yaw
        sx = c.stand_x[0] + torch.rand(m, device=dev) * (c.stand_x[1] - c.stand_x[0])
        sy = c.stand_y[0] + torch.rand(m, device=dev) * (c.stand_y[1] - c.stand_y[0])
        syaw = (torch.rand(m, device=dev) * 2 - 1) * c.stand_yaw

        self.frame_xy[env_ids, 0] = fx
        self.frame_xy[env_ids, 1] = fy
        self.frame_yaw[env_ids] = fyaw
        self.stand_xy[env_ids, 0] = sx
        self.stand_xy[env_ids, 1] = sy
        self.stand_yaw[env_ids] = syaw
        self.cap_off_latch[env_ids] = False
        self.seated_latch[env_ids] = False
        self.capped_latch[env_ids] = False
        self.seat_streak[env_ids] = 0
        self.cap_streak[env_ids] = 0
        self.drive_f[env_ids] = 0.0

        fxy = torch.stack([fx, fy], dim=1)
        sxy = torch.stack([sx, sy], dim=1)
        fq = self._yaw_quat(fyaw)
        parts = {name: (size, off) for name, size, off in c.frame_parts}
        for name, (_size, off) in parts.items():
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + self._place_env(fxy, fyaw, off)
            st[:, 3:7] = fq
            self.kin_parts[name].write_root_state_to_sim(st, env_ids)
        sq = self._yaw_quat(syaw)
        for name, _size, off in c.stand_parts:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + self._place_env(sxy, syaw, off)
            st[:, 3:7] = sq
            self.kin_parts[name].write_root_state_to_sim(st, env_ids)

        # pane: upright in the stand slot, its WIDTH along the stand's long (x) axis
        # -> pane yaw = stand yaw - pi/2 (pane body y = width).
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin + self._place_env(
            sxy, syaw, (0.0, 0.0, c.pane_stand_z + c.spawn_lift))
        st[:, 3:7] = self._yaw_quat(syaw - math.pi / 2)
        self.pane.write_root_state_to_sim(st, env_ids)
        # cap: closed, resting in the tray on the post tops.
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin + self._place_env(
            fxy, fyaw, (0.0, 0.0, c.cap_rest_z + c.spawn_lift))
        st[:, 3:7] = fq
        self.cap.write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def pane_pos(self) -> torch.Tensor:
        """(N, 3) pane centre, env-local."""
        return self.pane.data.root_pos_w - self.env_origins

    def cap_pos(self) -> torch.Tensor:
        return self.cap.data.root_pos_w - self.env_origins

    def pane_bottom_env(self) -> torch.Tensor:
        """(N, 3) centre of the pane's bottom edge, env-local."""
        from isaaclab.utils.math import quat_apply

        half = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        half[:, 2] = self.cfg.pane_size[2] / 2
        return self.pane_pos() - quat_apply(self.pane.data.root_quat_w, half)

    def _up_z(self, body: RigidObject) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    def pane_seated_now(self) -> torch.Tensor:
        """(N,) bool, geometric: the pane's bottom edge rests in the sill groove — frame-
        local |x| within `seat_x_tol`, |y| within `seat_y_tol`, bottom at the groove floor
        within `seat_z_tol`, pane near-upright."""
        c = self.cfg
        b = self.to_frame(self.pane_bottom_env())
        up = self._up_z(self.pane)
        ok = (b[:, 0].abs() < c.seat_x_tol) & (b[:, 1].abs() < c.seat_y_tol) \
            & ((b[:, 2] - c.groove_floor).abs() < c.seat_z_tol) \
            & (up.clamp(-1.0, 1.0) >= math.cos(math.radians(c.seat_up_min_deg)))
        return ok & torch.isfinite(b).all(dim=-1)

    def cap_seated_now(self) -> torch.Tensor:
        """(N,) bool, geometric: the cap lies LEVEL in its nub tray on the post tops."""
        c = self.cfg
        p = self.to_frame(self.cap_pos())
        up = self._up_z(self.cap)
        ok = (p[:, 0].abs() < c.cap_xy_tol[0]) & (p[:, 1].abs() < c.cap_xy_tol[1]) \
            & ((p[:, 2] - c.cap_rest_z).abs() < c.cap_z_tol) \
            & (up.clamp(-1.0, 1.0) >= math.cos(math.radians(c.cap_up_min_deg)))
        return ok & torch.isfinite(p).all(dim=-1)

    def cap_off_now(self) -> torch.Tensor:
        """(N,) bool: cap carried clear of the frame top (frame-local horizontal dist)."""
        p = self.to_frame(self.cap_pos())
        return p[:, :2].norm(dim=-1) > self.cfg.cap_off_dist

    def speeds(self) -> torch.Tensor:
        """(N, 2) |lin vel| of [pane, cap]."""
        return torch.stack([self.pane.data.root_lin_vel_w.norm(dim=-1),
                            self.cap.data.root_lin_vel_w.norm(dim=-1)], dim=1)

    def settled(self) -> torch.Tensor:
        return (self.speeds() < self.cfg.settle_speed).all(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: pane seated in the groove AND cap seated in its tray RIGHT NOW,
        both settled (current physical configuration — judged on where things rest)."""
        return self.pane_seated_now() & self.cap_seated_now() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: w_cap_off * cap_off_latch + w_seated * seated_latch
        + w_capped * capped_latch + w_success * success(). ~0 for the null policy;
        exactly 1.0 iff success() (all latches then mature); latched credit never
        evaporates."""
        c = self.cfg
        return (c.w_cap_off * (self.cap_off_latch | self.seated_latch).float()
                + c.w_seated * self.seated_latch.float()
                + c.w_capped * self.capped_latch.float()
                + c.w_success * self.success().float())

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Apply the drive buffer (world-frame semantics, rotated into each body's frame
        before application — owns the wrench slots), then update the rubric latches.
        A diverged substep must not latch — NaN comparisons are False, garbage earns 0."""
        from isaaclab.utils.math import quat_apply_inverse

        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        zero_t = torch.zeros(n, 1, 3, device=dev)
        for row, body in ((PANE, self.pane), (CAP, self.cap)):
            f_body = quat_apply_inverse(body.data.root_quat_w, self.drive_f[:, row, :])
            body.set_external_force_and_torque(f_body.unsqueeze(1), zero_t)

        spd = self.speeds()
        seated = self.pane_seated_now()
        capped = self.cap_seated_now()
        self.cap_off_latch |= self.cap_off_now()
        good_seat = seated & (spd[:, PANE] < c.latch_slow)
        self.seat_streak = torch.where(good_seat, self.seat_streak + 1,
                                       torch.zeros_like(self.seat_streak))
        self.seated_latch |= self.seat_streak >= c.streak_steps
        # order-aware: recapping an EMPTY frame latches nothing.
        good_cap = capped & seated & (spd[:, CAP] < c.latch_slow)
        self.cap_streak = torch.where(good_cap, self.cap_streak + 1,
                                      torch.zeros_like(self.cap_streak))
        self.capped_latch |= self.cap_streak >= c.streak_steps

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pane": self.pane.data.root_state_w[env_ids].clone(),
            "cap": self.cap.data.root_state_w[env_ids].clone(),
            "kin": {nm: b.data.root_state_w[env_ids].clone()
                    for nm, b in self.kin_parts.items()},
            "task": {kk: getattr(self, kk)[env_ids].clone()
                     for kk in ("frame_xy", "frame_yaw", "stand_xy", "stand_yaw",
                                "cap_off_latch", "seated_latch", "capped_latch",
                                "seat_streak", "cap_streak", "drive_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pane.write_root_state_to_sim(state["pane"], env_ids)
        self.cap.write_root_state_to_sim(state["cap"], env_ids)
        for nm, b in self.kin_parts.items():
            b.write_root_state_to_sim(state["kin"][nm], env_ids)
        for kk, v in state["task"].items():
            getattr(self, kk)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An EMPTY wooden window frame stands on a table: two upright posts on a thick "
            f"sill, with a narrow open groove (a {22:.0f} mm channel between two rails) "
            f"running along the sill top between the posts, and two thin fixed cross-bars "
            f"(mullions) spanning the frame at mid-height, one on each face. The frame's "
            f"position and its yaw on the table change every episode. Resting flat on TOP "
            f"of the two posts, held between small corner nubs, lies a loose DARK-BROWN CAP "
            f"LID (a {c.cap_size[0] * 100:.0f} x {c.cap_size[1] * 100:.0f} cm plate) that "
            f"roofs the frame's top opening. Nearer to you, a LIGHT-BLUE GLASS PANE "
            f"({c.pane_size[1] * 100:.0f} cm wide, {c.pane_size[2] * 100:.0f} cm tall, "
            f"{c.pane_size[0] * 1000:.0f} mm thick) stands upright in a small gray slotted "
            f"stand, also at a changing position and yaw.\n"
            f"Goal — glaze the window, in this order (the geometry enforces it):\n"
            f"1. Lift the cap lid off the post tops and set it down anywhere clear of the "
            f"frame (the lid covers the only insertion opening; while it is on, the groove "
            f"is unreachable — the mullions block sneaking the pane in from the side).\n"
            f"2. Take the pane out of its stand, keep it upright, lower it down through the "
            f"frame's top opening between the posts, and seat its bottom edge fully into "
            f"the sill groove. A pane leaning against the frame, resting on the rails, or "
            f"left anywhere else does not count.\n"
            f"3. Put the cap lid back FLAT on the post tops, inside the corner nubs, "
            f"roofing the glazed frame. Recapping the frame while it is still empty counts "
            f"for nothing.\n"
            f"The task is complete when the pane rests seated in the groove AND the cap "
            f"lies level in its tray at the same time, everything still."
        )

    def instruction(self) -> str:
        return (
            "Lift the cap lid off the window frame's post tops and set it aside, slide the "
            "blue glass pane out of its stand and lower it down between the posts until its "
            "bottom edge seats in the sill groove, then rest the lid back flat on the post "
            "tops. The task fails if the pane is not seated in the groove or the lid is not "
            "back in its tray."
        )


# Guarded registration: the forge may import this module under two names.
if "glaze_window" not in SCENES.list():
    SCENES.register("glaze_window", GlazeWindowScene)
if "simgen.glaze_window" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="glaze_window", robot="null"))
