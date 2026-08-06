"""TableclothPullScene — yank the runner out from under the trophy (sim_gen task
`scene_a_i57`, derived from calvin/scene_A).

The seed is the CALVIN table: every task in that family either actuates a furniture DOF
(drawer / slider / button / switch) or executes grasp-block -> transport -> release, and
all of it is quasi-static — speed never matters, the target object is always the thing
manipulated. This task inverts both properties. A pink trophy block stands on a thin,
slick RUNNER sheet lying on the floor. The graded object (the trophy) must NEVER be
transported: a permanent `disturbed` latch trips the moment its center leaves a 6 cm
sphere around its spawn, capping the score near zero forever. The manipulated object is
the SUBSTRATE underneath, and the required skill is DYNAMIC: pull the runner out fast
enough that friction has no time to drag the trophy along (the tablecloth trick), then
stow the extracted runner flat on a marked pad. Executing the very same pull
quasi-statically fails measurably — the trophy rides the runner past the latch radius
long before the runner clears (there is no other horizontal force on it), and the seed's
own plan (pick the block up, or slide it) trips the same latch by construction: the
shortest path off the runner footprint is >= 75 mm of travel, above the 60 mm latch.

Judged on PHYSICAL outcomes:
  - success() : the `extracted` latch fired (trophy seen standing on the GROUND, clear of
    the runner footprint, undisturbed), the trophy NOW rests upright / settled / within
    `stray_tol` of its spawn, the runner NOW rests flat on the stow pad (centered,
    yaw-aligned mod 180 deg, at pad height, settled), and `disturbed` never fired.
  - score()   : ~0 for doing nothing; live shaping up to 0.30 for partial withdrawal of
    the runner (gated on the trophy still standing untouched); latched milestones
    `extracted` * (0.50 + 0.20 * trophy-preserved-now + 0.30 * runner-stowed-now);
    exactly 1.0 iff success; capped at 0.05 forever once `disturbed` fires — returning
    the trophy to its anchor afterwards does not restore anything.

Per-episode randomization: runner pose (xy + full yaw — the pull direction must be read
from the scene), the trophy's seat on the runner (along / across offsets + free yaw),
and the stow pad's bearing, distance and yaw.

Assets are fully procedural plain primitives (CuboidCfg only): the low-friction runner
(friction_combine_mode="min" makes the runner-trophy interface slick while the trophy
still grips the ground), the trophy block, and a kinematic stow pad re-posed per reset.
Heavy imports (isaaclab) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TableclothPullSceneCfg(BaseCfg):
    """Config for `TableclothPullScene`. Env-local frame: the runner spawns near the
    origin with the trophy standing on it; the stow pad stands ~half a meter away.

    Honesty margins (dry-computed, asserted nowhere but relied on everywhere): the
    trophy's shortest travel OFF the runner footprint is `runner_size[1]/2 +
    trophy half-diagonal - trophy_across` = 70 + 35.4 - 30 = 75 mm laterally (150+ mm
    along the runner axis), both above the 60 mm `stray_tol` latch — so shoving or
    tilt-dumping the trophy off the runner is a failure by construction, and the only
    scene-level winning move is removing the runner from under the standing trophy."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    stray_tol: float = tunable(0.06)        # permanent trophy-displacement latch radius (m, 3D)
    upright_tol_deg: float = tunable(15.0)  # trophy long axis within this of world-up
    grounded_tol: float = tunable(0.02)     # trophy bottom within this of the ground (m)
    clear_bottom: float = tunable(0.003)    # "standing on the GROUND" gate: trophy bottom below
    # this (m) — on the 6 mm runner the bottom reads ~6 mm, on the ground ~0
    clear_margin: float = tunable(0.005)    # extra footprint separation for `clear` (m)
    settle_lin: float = tunable(0.05)       # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.60)       # max |ang vel| when judging settled (rad/s)
    stow_xy_slack: float = tunable(0.010)   # runner-center containment slack on the pad (m)
    stow_z_tol: float = tunable(0.006)      # runner rest height tolerance on the pad (m)
    stow_flat_deg: float = tunable(10.0)    # runner top face within this of world-up
    stow_align_deg: float = tunable(15.0)   # runner axis within this of the pad axis (mod 180)
    withdraw_ref: float = tunable(0.20)     # runner displacement that earns full shaping (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    runner_jitter: float = tunable(0.05)    # uniform +/- xy jitter of the runner spawn (m)
    runner_yaw_deg: float = tunable(180.0)  # uniform +/- runner yaw (the pull direction)
    trophy_along: float = tunable(0.04)     # +/- trophy seat offset along the runner axis (m)
    trophy_across: float = tunable(0.03)    # +/- trophy seat offset across the runner (m)
    trophy_yaw_deg: float = tunable(180.0)  # uniform +/- trophy yaw
    pad_dist_range: tuple = tunable((0.48, 0.62))  # stow pad distance from the runner spawn (m)
    pad_yaw_deg: float = tunable(180.0)     # uniform +/- pad yaw

    # --- info: structure ---------------------------------------------------------------------
    runner_size: tuple = info((0.30, 0.14, 0.006))  # the slick sheet (x = long axis)
    runner_mass: float = info(0.12)
    runner_mu: float = info(0.05)           # runner material friction (combine "min": the
    # runner-trophy AND runner-ground interfaces are slick; trophy-ground stays grippy)
    runner_color: tuple = info((0.20, 0.40, 0.85))
    trophy_size: tuple = info((0.05, 0.05, 0.07))   # squat enough that the yank cannot tip it
    trophy_mass: float = info(0.08)
    trophy_mu: float = info(0.60)
    trophy_color: tuple = info((0.92, 0.32, 0.62))  # CALVIN pink
    pad_size: tuple = info((0.36, 0.20, 0.016))     # kinematic stow pad
    pad_color: tuple = info((0.30, 0.30, 0.34))
    contact_offset: float = info(0.002)     # explicit small offsets: the 3 mm `clear_bottom`
    # gate cannot survive the ~2 cm platform default


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tablecloth_pull")
class TableclothPullScene(BaseScene):
    cfg: TableclothPullSceneCfg

    def __init__(self, cfg: TableclothPullSceneCfg | None = None) -> None:
        super().__init__(cfg or TableclothPullSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the slick runner, the trophy standing on it, and the kinematic
        stow pad (re-posed by reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

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
            "runner": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Runner",
                spawn=sim_utils.CuboidCfg(
                    size=c.runner_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                        linear_damping=0.02, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.runner_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.runner_mu + 0.01, dynamic_friction=c.runner_mu,
                        restitution=0.0, friction_combine_mode="min",
                        restitution_combine_mode="min"),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.runner_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.runner_size[2] / 2 + 0.001)),
            ),
            "trophy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Trophy",
                spawn=sim_utils.CuboidCfg(
                    size=c.trophy_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.trophy_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.trophy_mu + 0.05, dynamic_friction=c.trophy_mu,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.trophy_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.runner_size[2] + c.trophy_size[2] / 2 + 0.002)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/StowPad",
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, c.pad_size[2] / 2)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode anchors and latches the rubric uses."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.runner: RigidObject = env.iscene["runner"]
        self.trophy: RigidObject = env.iscene["trophy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        self.trophy_anchor = torch.zeros(n, 3, device=dev)   # trophy spawn center (env-local)
        self.runner_start_xy = torch.zeros(n, 2, device=dev)  # runner spawn center (env-local)
        self.pad_xy = torch.zeros(n, 2, device=dev)          # pad center (env-local)
        self.disturbed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.extracted = torch.zeros(n, dtype=torch.bool, device=dev)

    def _local(self, body) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample runner pose, seat the trophy on it, place the stow pad at
        a clear bearing, store the anchors, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, device=dev)

        r_xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.runner_jitter
        r_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.runner_yaw_deg)

        # runner: flat on the ground at (r_xy, r_yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = r_xy
        st[:, 2] = c.runner_size[2] / 2 + 0.001
        st[:, 3] = torch.cos(r_yaw / 2)
        st[:, 6] = torch.sin(r_yaw / 2)
        st[:, 0:3] += origin
        self.runner.write_root_state_to_sim(st, env_ids)

        # trophy: standing ON the runner at a sampled seat (runner-frame along/across)
        a = (torch.rand(m, device=dev) * 2 - 1) * c.trophy_along
        b = (torch.rand(m, device=dev) * 2 - 1) * c.trophy_across
        tx = r_xy[:, 0] + a * torch.cos(r_yaw) - b * torch.sin(r_yaw)
        ty = r_xy[:, 1] + a * torch.sin(r_yaw) + b * torch.cos(r_yaw)
        tz = c.runner_size[2] + c.trophy_size[2] / 2 + 0.001
        t_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.trophy_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = tx
        st[:, 1] = ty
        st[:, 2] = tz
        st[:, 3] = torch.cos(t_yaw / 2)
        st[:, 6] = torch.sin(t_yaw / 2)
        st[:, 0:3] += origin
        self.trophy.write_root_state_to_sim(st, env_ids)

        # stow pad: kinematic — POSE write only, sampled bearing/distance/yaw from the
        # runner spawn (min distance 0.48 clears runner + trophy by > 0.10 at any yaw)
        bear = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        dist = u(*c.pad_dist_range)
        p_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pad_yaw_deg)
        px = r_xy[:, 0] + dist * torch.cos(bear)
        py = r_xy[:, 1] + dist * torch.sin(bear)
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = px
        st[:, 1] = py
        st[:, 2] = c.pad_size[2] / 2
        st[:, 3] = torch.cos(p_yaw / 2)
        st[:, 6] = torch.sin(p_yaw / 2)
        st[:, 0:3] += origin
        self.pad.write_root_pose_to_sim(st, env_ids)

        anchor = torch.stack([tx, ty, torch.full_like(tx, tz)], dim=-1)
        self.trophy_anchor[env_ids] = anchor
        self.runner_start_xy[env_ids] = r_xy
        self.pad_xy[env_ids, 0] = px
        self.pad_xy[env_ids, 1] = py
        self.disturbed[env_ids] = False
        self.extracted[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: trip the permanent `disturbed` latch, and latch
        `extracted` the moment the trophy is seen standing on the ground, clear of the
        runner footprint, undisturbed. Latching per substep is the anti-teleport evidence:
        a runner teleported straight to the pad with the trophy still aboard leaves
        `extracted` false forever (the trophy never stood clear on the ground)."""
        disp = (self._local(self.trophy) - self.trophy_anchor).norm(dim=-1)
        self.disturbed |= disp > self.cfg.stray_tol
        self.extracted |= self.clear_of_runner() & self.trophy_upright() & ~self.disturbed

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "runner": self.runner.data.root_state_w[env_ids].clone(),
            "trophy": self.trophy.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "trophy_anchor": self.trophy_anchor[env_ids].clone(),
            "runner_start_xy": self.runner_start_xy[env_ids].clone(),
            "pad_xy": self.pad_xy[env_ids].clone(),
            "disturbed": self.disturbed[env_ids].clone(),
            "extracted": self.extracted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.runner.write_root_state_to_sim(state["runner"], env_ids)
        self.trophy.write_root_state_to_sim(state["trophy"], env_ids)
        self.pad.write_root_pose_to_sim(state["pad"][:, 0:7], env_ids)
        self.trophy_anchor[env_ids] = state["trophy_anchor"]
        self.runner_start_xy[env_ids] = state["runner_start_xy"]
        self.pad_xy[env_ids] = state["pad_xy"]
        self.disturbed[env_ids] = state["disturbed"]
        self.extracted[env_ids] = state["extracted"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pink trophy block ({c.trophy_size[2] * 1000:.0f} mm tall) stands on a thin "
            f"slick blue runner sheet ({c.runner_size[0] * 100:.0f} x "
            f"{c.runner_size[1] * 100:.0f} cm) lying on the floor. A flat gray stow pad "
            f"waits about half a meter away; its direction, and the runner's heading, "
            f"change every episode.\n"
            f"Goal: get the runner out from UNDER the trophy and lay it flat on the stow "
            f"pad — while the trophy keeps standing exactly where it is. The trophy must "
            f"never be picked up, slid or knocked along: if its center ever moves more "
            f"than {c.stray_tol * 100:.0f} cm from where it started, the episode is "
            f"spoiled for good (putting it back does not help). Friction drags the trophy "
            f"with any slow pull — snap the runner out fast, like the tablecloth trick, "
            f"then stow the sheet on the pad."
        )

    # ----- predicates ------------------------------------------------------------------------
    def _up_of(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        return quat_apply(body.data.root_quat_w, ez)

    def trophy_upright(self) -> torch.Tensor:
        """(N,) bool: trophy long axis within `upright_tol_deg` of world-up."""
        up = self._up_of(self.trophy)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_tol_deg))

    def trophy_in_place(self) -> torch.Tensor:
        """(N,) bool: trophy center within `stray_tol` (3D) of its spawn anchor."""
        disp = (self._local(self.trophy) - self.trophy_anchor).norm(dim=-1)
        return disp <= self.cfg.stray_tol

    def trophy_settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.trophy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.trophy.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def runner_settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.runner.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.runner.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def clear_of_runner(self) -> torch.Tensor:
        """(N,) bool, geometric: the trophy stands on the GROUND (bottom below
        `clear_bottom` — on the 6 mm runner it reads ~6 mm) with its center outside the
        runner footprint in the RUNNER's frame, by half-diagonal + `clear_margin`."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = self.trophy.data.root_pos_w - self.runner.data.root_pos_w
        loc = quat_apply_inverse(self.runner.data.root_quat_w, rel)
        half_diag = math.hypot(c.trophy_size[0], c.trophy_size[1]) / 2
        apart = (
            (loc[:, 0].abs() > c.runner_size[0] / 2 + half_diag + c.clear_margin)
            | (loc[:, 1].abs() > c.runner_size[1] / 2 + half_diag + c.clear_margin)
        )
        bottom = self._local(self.trophy)[:, 2] - c.trophy_size[2] / 2
        return apart & (bottom < c.clear_bottom)

    def trophy_preserved(self) -> torch.Tensor:
        """(N,) bool, current state: the trophy rests upright, grounded, settled, within
        the stray sphere — the thing the yank must not break."""
        c = self.cfg
        bottom = self._local(self.trophy)[:, 2] - c.trophy_size[2] / 2
        grounded = bottom.abs() < c.grounded_tol
        return self.trophy_upright() & self.trophy_in_place() & grounded & self.trophy_settled()

    def runner_stowed(self) -> torch.Tensor:
        """(N,) bool, current state: the runner rests flat ON the pad — center contained
        (pad frame, geometric margin + `stow_xy_slack`), at pad-top rest height, top face
        up within `stow_flat_deg`, axis aligned with the pad's within `stow_align_deg`
        (mod 180), settled."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        rel = self.runner.data.root_pos_w - self.pad.data.root_pos_w
        loc = quat_apply_inverse(self.pad.data.root_quat_w, rel)
        tol_x = (c.pad_size[0] - c.runner_size[0]) / 2 + c.stow_xy_slack
        tol_y = (c.pad_size[1] - c.runner_size[1]) / 2 + c.stow_xy_slack
        centered = (loc[:, 0].abs() < tol_x) & (loc[:, 1].abs() < tol_y)
        at_height = (loc[:, 2] - (c.pad_size[2] / 2 + c.runner_size[2] / 2)).abs() < c.stow_z_tol
        flat = self._up_of(self.runner)[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(c.stow_flat_deg))
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        rx = quat_apply(self.runner.data.root_quat_w, ex)[:, :2]
        px = quat_apply(self.pad.data.root_quat_w, ex)[:, :2]
        rx = rx / rx.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        px = px / px.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        aligned = (rx * px).sum(dim=-1).abs() >= math.cos(math.radians(c.stow_align_deg))
        return centered & at_height & flat & aligned & self.runner_settled()

    # ----- progress / rubric ------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: extraction latched (trophy once stood clear on the ground,
        undisturbed), the trophy is preserved NOW, the runner is stowed NOW, and the
        trophy never strayed this episode."""
        return self.extracted & self.trophy_preserved() & self.runner_stowed() & ~self.disturbed

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: live shaping 0.30 * runner-withdrawal (gated on the
        trophy still standing untouched); latched milestones
        extracted * (0.50 + 0.20 * preserved-now + 0.30 * stowed-now); exactly 1.0 iff
        success; capped at 0.05 forever once `disturbed` fires. Doing nothing scores 0
        (the runner has not moved, nothing is latched)."""
        c = self.cfg
        r_disp = (self._local(self.runner)[:, :2] - self.runner_start_xy).norm(dim=-1)
        gate = (self.trophy_upright() & self.trophy_in_place()).float()
        shaping = 0.30 * (r_disp / c.withdraw_ref).clamp(0.0, 1.0) * gate
        milestones = self.extracted.float() * (
            0.50 + 0.20 * self.trophy_preserved().float() + 0.30 * self.runner_stowed().float())
        s = torch.maximum(shaping, milestones)
        s = torch.where(self.success(), torch.ones_like(s), s)
        return torch.where(self.disturbed, s.clamp(max=0.05), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="tablecloth_pull", robot="null", env_spacing=4.0))
