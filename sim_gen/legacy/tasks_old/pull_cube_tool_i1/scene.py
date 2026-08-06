"""TunnelShuffleScene — send a cube AWAY down a lane, under a low tunnel, into a distant
goal box (sim_gen task `pull_cube_tool_i1`, derived from maniskill/pull_cube_tool).

The seed pulls a distant out-of-reach cube INWARD with an L-shaped tool (reach extension,
quasi-static dragging toward the robot base). This task is the strategic inverse: the cube
starts close at hand and must be SENT AWAY down a walled lane, sliding on the surface
UNDER a low tunnel (its opening is barely taller than the cube — nothing holding the cube
fits through), to come to rest fully inside a distant goal box painted on the lane.
The required skill is calibrated-impulse / momentum control (shove speed -> friction
stopping distance), not tool-mediated reach extension; the seed's own plan ("drag the cube
toward your base") slides the cube off the NEAR end of the lane and scores zero.

Judged on PHYSICAL outcomes:
  - success() : the cube rests ON the lane surface, fully inside the goal band in x,
    settled (lin+ang velocity gates), AND it physically transited the tunnel this episode
    (a latch set only by a continuous, on-surface crossing of the gate plane between two
    consecutive physics steps — a kinematic teleport across the gate does NOT latch, so
    "teleport it into the goal" is not a solution).
  - score()   : 0 for doing nothing; 0.45 * forward-progress (current, on-lane, normalized
    spawn -> goal center, clamped so overshoot is not rewarded) + 0.35 for the latched
    tunnel transit; exactly 1.0 iff success. Max non-success score = 0.8.

Per-episode randomization: cube spawn pose (xy + yaw), tunnel gate position along the
lane, and goal-box center distance — so a memorized fixed impulse fails; the solver must
read the scene and calibrate.

Assets are fully procedural (plain CuboidCfg boxes: lane slab, two side walls, tunnel
roof, sunken goal marker, the cube). Heavy imports (isaaclab) are deferred so importing
this module — and registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TunnelShuffleSceneCfg(BaseCfg):
    """Config for `TunnelShuffleScene`. Lane frame: +x = down-lane (away from the solver's
    side), y = across, lane top surface at z = `lane_t`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.04)   # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.50)   # max |ang vel| when judging settled (rad/s)
    on_lane_z_tol: float = tunable(0.012)  # cube center height tolerance for "resting on lane"
    goal_half: float = tunable(0.075)   # goal box half-length in x; success needs the cube
    # FULLY inside: |x - goal_center| <= goal_half - cube_half.

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    spawn_x_range: tuple = tunable((0.02, 0.12))   # cube spawn x band (near end of lane)
    spawn_y_jitter: float = tunable(0.04)          # uniform +/- y jitter around lane center
    spawn_yaw_deg: float = tunable(180.0)          # uniform +/- yaw (cube is symmetric)
    gate_x_range: tuple = tunable((0.42, 0.58))    # tunnel gate center position along lane
    goal_center_range: tuple = tunable((0.85, 1.05))  # goal box center distance

    # --- tunable: difficulty geometry --------------------------------------------------------
    opening_h: float = tunable(0.080)  # tunnel opening height above the lane top; the cube
    # (0.050) passes with 30 mm clearance, a cube held in any gripper does not.
    friction: float = tunable(0.30)    # lane+cube dynamic friction — sets the impulse->distance
    # map (stop distance = v^2 / (2*mu*g)); the solver must calibrate against it.

    # --- info: structure ---------------------------------------------------------------------
    cube_size: float = info(0.050)
    cube_mass: float = info(0.05)
    lane_x0: float = info(-0.15)   # near (solver-side) end of the lane — pull the cube
    # backward and it slides off here onto the ground (the seed-strategy failure).
    lane_x1: float = info(1.35)    # far end — an overshoot slides off and drops to the ground.
    lane_inner_half_w: float = info(0.10)  # inner half-width between the walls
    lane_t: float = info(0.04)     # lane slab thickness; lane top at z = lane_t
    wall_t: float = info(0.02)
    wall_h: float = info(0.12)     # tall enough that the only way down-lane is under the roof
    roof_len_x: float = info(0.10)  # tunnel depth along the lane
    roof_t: float = info(0.03)
    marker_t: float = info(0.006)  # goal paint slab, sunk flush into the lane (top +0.2 mm)
    contact_offset: float = info(0.002)  # small: the default ~2 cm offset would eat the
    # 30 mm roof clearance (phantom roof contact on a sliding cube).

    # Derived (filled in __post_init__).
    lane_len: float = field(default=None, init=False)
    lane_cx: float = field(default=None, init=False)
    lane_top: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.lane_len = round(self.lane_x1 - self.lane_x0, 4)
        self.lane_cx = round((self.lane_x0 + self.lane_x1) / 2, 4)
        self.lane_top = self.lane_t


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tunnel_shuffle")
class TunnelShuffleScene(BaseScene):
    cfg: TunnelShuffleSceneCfg

    def __init__(self, cfg: TunnelShuffleSceneCfg | None = None) -> None:
        super().__init__(cfg or TunnelShuffleSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the static lane (slab + 2 walls), the movable tunnel roof, the
        sunken goal marker, and the cube. Roof / marker / cube are re-placed by reset()."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        lane_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=c.friction, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
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
            "lane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lane",
                spawn=sim_utils.CuboidCfg(
                    size=(c.lane_len, 2 * (c.lane_inner_half_w + c.wall_t), c.lane_t),
                    rigid_props=kin, collision_props=coll, physics_material=lane_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.lane_cx, 0.0, c.lane_t / 2)),
            ),
        }
        for name, side in (("wall_l", 1.0), ("wall_r", -1.0)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=(c.lane_len, c.wall_t, c.wall_h),
                    rigid_props=kin, collision_props=coll, physics_material=lane_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.30, 0.34)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.lane_cx, side * (c.lane_inner_half_w + c.wall_t / 2),
                         c.lane_top + c.wall_h / 2)),
            )
        out["roof"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Roof",
            spawn=sim_utils.CuboidCfg(
                size=(c.roof_len_x, 2 * (c.lane_inner_half_w + c.wall_t) + 0.04, c.roof_t),
                rigid_props=kin, collision_props=coll,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.30, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, c.lane_top + c.opening_h + c.roof_t / 2)),
        )
        # Goal paint: a green slab sunk flush into the lane top (protrudes 0.2 mm — a
        # visual marker the cube slides over, not a wall).
        out["marker"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Marker",
            spawn=sim_utils.CuboidCfg(
                size=(2 * 0.075, 2 * c.lane_inner_half_w, c.marker_t),
                rigid_props=kin, collision_props=coll, physics_material=lane_mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.95, 0.0, c.lane_top + 0.0002 - c.marker_t / 2)),
        )
        out["cube"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(c.cube_size, c.cube_size, c.cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0,
                    linear_damping=0.0, angular_damping=0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                collision_props=coll, physics_material=lane_mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.07, 0.0, c.lane_top + c.cube_size / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode buffers the rubric depends on."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.cube: RigidObject = env.iscene["cube"]
        self.roof: RigidObject = env.iscene["roof"]
        self.marker: RigidObject = env.iscene["marker"]
        self.env_origins = env.iscene.env_origins
        self.spawn_x = torch.zeros(n, device=dev)   # cube spawn x (env-local), reset-time
        self.gate_x = torch.full((n,), 0.5, device=dev)   # tunnel gate plane (env-local)
        self.goal_c = torch.full((n,), 0.95, device=dev)  # goal box center (env-local)
        self.gate_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_x = torch.zeros(n, device=dev)   # last-step cube x, for transit detection

    def _cube_local(self) -> torch.Tensor:
        """Cube center position in env-local coords, (N, 3)."""
        return self.cube.data.root_pos_w - self.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample gate position, goal distance and cube spawn pose; clear the
        transit latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, device=dev)

        gx = u(*c.gate_x_range)
        gc = u(*c.goal_center_range)
        self.gate_x[env_ids] = gx
        self.goal_c[env_ids] = gc

        # Kinematic decorations move by POSE write (no velocity on kinematic bodies).
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = gx
        st[:, 2] = c.lane_top + c.opening_h + c.roof_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.roof.write_root_pose_to_sim(st, env_ids)

        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = gc
        st[:, 2] = c.lane_top + 0.0002 - c.marker_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.marker.write_root_pose_to_sim(st, env_ids)

        sx = u(*c.spawn_x_range)
        sy = (torch.rand(m, device=dev) * 2 - 1) * c.spawn_y_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = sx
        st[:, 1] = sy
        st[:, 2] = c.lane_top + c.cube_size / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.cube.write_root_state_to_sim(st, env_ids)

        self.spawn_x[env_ids] = sx
        self._prev_x[env_ids] = sx
        self.gate_latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Transit detection, every physics substep: latch when the cube crosses the gate
        plane forward between two CONSECUTIVE steps (|dx| < 5 cm — a kinematic teleport
        across the gate moves too far in one step and does NOT latch) while its center is
        below the tunnel opening (an over-the-roof crossing does not latch either)."""
        c = self.cfg
        p = self._cube_local()
        x, z = p[:, 0], p[:, 2]
        dx = x - self._prev_x
        crossed = ((self._prev_x < self.gate_x) & (x >= self.gate_x)
                   & (dx > 0) & (dx < 0.05) & (z < c.lane_top + c.opening_h))
        self.gate_latch |= crossed
        self._prev_x = x.clone()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "roof": self.roof.data.root_state_w[env_ids].clone(),
            "marker": self.marker.data.root_state_w[env_ids].clone(),
            "spawn_x": self.spawn_x[env_ids].clone(),
            "gate_x": self.gate_x[env_ids].clone(),
            "goal_c": self.goal_c[env_ids].clone(),
            "gate_latch": self.gate_latch[env_ids].clone(),
            "prev_x": self._prev_x[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.roof.write_root_pose_to_sim(state["roof"][:, 0:7], env_ids)
        self.marker.write_root_pose_to_sim(state["marker"][:, 0:7], env_ids)
        self.spawn_x[env_ids] = state["spawn_x"]
        self.gate_x[env_ids] = state["gate_x"]
        self.goal_c[env_ids] = state["goal_c"]
        self.gate_latch[env_ids] = state["gate_latch"]
        self._prev_x[env_ids] = state["prev_x"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A long straight lane ({c.lane_len:.2f} m) with tall side walls runs away from "
            f"you; a red cube ({c.cube_size * 1000:.0f} mm) sits near its close end, within "
            f"easy reach. Partway down the lane a low tunnel (an orange roof, opening only "
            f"{c.opening_h * 1000:.0f} mm high) spans the full width; far beyond it a green "
            f"goal box is painted flush on the lane floor. Both the tunnel position and the "
            f"goal distance change every episode.\n"
            f"Goal: send the cube down the lane so it passes UNDER the tunnel sliding on the "
            f"surface and comes to rest fully inside the green box. The tunnel is too low "
            f"for anything but the bare cube, and both lane ends are open: pull the cube "
            f"toward you and it drops off the near edge; shove it too hard and it flies off "
            f"the far edge. Calibrate your push."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def on_lane(self) -> torch.Tensor:
        """(N,) bool: the cube rests flat ON the lane surface, between the walls, between
        the open ends."""
        c = self.cfg
        p = self._cube_local()
        z_ok = (p[:, 2] - (c.lane_top + c.cube_size / 2)).abs() < c.on_lane_z_tol
        y_ok = p[:, 1].abs() < c.lane_inner_half_w
        x_ok = (p[:, 0] > c.lane_x0 + c.cube_size / 2) & (p[:, 0] < c.lane_x1 - c.cube_size / 2)
        return z_ok & y_ok & x_ok

    def settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.cube.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def in_goal(self) -> torch.Tensor:
        """(N,) bool: cube center inside the goal band with full-containment margin."""
        c = self.cfg
        x = self._cube_local()[:, 0]
        return (x - self.goal_c).abs() <= c.goal_half - c.cube_size / 2

    def success(self) -> torch.Tensor:
        """(N,) bool: settled flat on the lane, fully inside the goal box, having physically
        transited the tunnel this episode."""
        return self.on_lane() & self.in_goal() & self.settled() & self.gate_latch

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.45 * current normalized forward progress while on the
        lane (spawn -> goal center, clamped — overshoot is not rewarded and a cube off the
        lane earns nothing) + 0.35 for the latched tunnel transit; exactly 1.0 iff success.
        Doing nothing scores 0 (progress is measured from this episode's own spawn)."""
        x = self._cube_local()[:, 0]
        denom = (self.goal_c - self.spawn_x).clamp(min=1e-6)
        fwd = ((x - self.spawn_x) / denom).clamp(0.0, 1.0) * self.on_lane().float()
        part = 0.45 * fwd + 0.35 * self.gate_latch.float()
        return torch.where(self.success(), torch.ones_like(part), part)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="tunnel_shuffle", robot="null", env_spacing=4.0))
