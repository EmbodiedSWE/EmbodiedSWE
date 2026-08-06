"""RunawayCanScene — catch a coke can already rolling toward a counter edge, stand it
upright, park it on a coaster pad (sim_gen task `coke_task_i20`, derived from
simpler_env/coke_task).

The seed is the SimplerEnv pick-coke-can family: a STATIC can on a table, success the
moment it is grasped and lifted a few centimetres — one stage, no placement, no time
pressure, the world politely waits. This task inverts every one of those properties:
at reset the can is ALREADY ROLLING on its side across a raised counter toward the open
far edge. Below the edge is the floor — a dropped can is ruined (dented / fizzing), so
falling latches an irreversible LOST state that zeroes the episode. The solver must

  1. INTERCEPT — arrest the rolling can on the counter before it goes over the edge
     (a hard reaction deadline of ~1-1.6 s set by the sampled roll speed);
  2. RIGHT     — re-orient the lying cylinder to stand upright (either end up);
  3. PARK      — place it standing, settled, on the coaster pad sampled elsewhere on
     the counter.

Execution order is REQUIRED: nothing can be righted or parked after the can is lost,
and the can is lost unless it is intercepted first. The seed's own plan — grasp the can
and hoist it — is an expressible negative control: a can held aloft is neither resting
on the counter nor on the pad, so the plan earns at most the intercept credit and never
success.

Judged on PHYSICAL outcomes only:
  - success(): the can rests upright (axis within `upright_max_deg` of vertical, either
    end up) ON the coaster pad (xy within `coaster_xy_tol`, base at pad height),
    settled (lin+ang velocity gates), and it NEVER fell below the counter top this
    episode (`lost` latch — set by real fall physics, checked every substep).
  - score(): 0 for doing nothing (the rolling can carries no credit) and 0 forever once
    the can is lost; otherwise 0.2 for the latched intercept (the can came to rest on
    the counter after its rolling start) + 0.3 for the latched righting (upright,
    settled, on the counter); exactly 1.0 iff success. Max non-success score = 0.5.

Per-episode randomization: spawn position, roll speed AND direction yaw, coaster pad
position — so a memorized catch point or a fixed place pose fails.

Assets are fully procedural (kinematic counter slab, flush kinematic coaster pad, one
dynamic cylinder can — no external files). Heavy imports (isaaclab) are deferred so
importing this module — and registering the scene — stays app-free.
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
class RunawayCanSceneCfg(BaseCfg):
    """Config for `RunawayCanScene`. Counter frame: +x = the roll direction (toward the
    open far edge), y = across, counter top surface at z = `counter_top`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.04)      # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.50)      # max |ang vel| when judging settled (rad/s);
    # a can rolling at task speed spins at v/r ~ 9-14 rad/s, far above this gate.
    upright_max_deg: float = tunable(15.0)  # can axis within this of vertical (either end
    # up; the bare cylinder tips over only past atan(r/(len/2)) ~ 30 deg, so 15 is honest).
    coaster_xy_tol: float = tunable(0.045)  # can centre within this of the pad centre; pad
    # half-size 0.08 minus can radius 0.033 = 0.047, so a counted can sits FULLY on the pad.
    on_pad_z_tol: float = tunable(0.012)   # can centre height tolerance for standing on the pad
    lost_below: float = tunable(0.12)      # LOST latches when the can centre falls this far
    # below the counter top (mid-air past the edge; on the floor it sits ~0.2 below).

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    spawn_x_range: tuple = tunable((-0.02, 0.06))    # roll start x band
    spawn_y_jitter: float = tunable(0.10)            # uniform +/- y around the centreline
    roll_speed_range: tuple = tunable((0.30, 0.45))  # initial roll speed (m/s) -> the
    # reaction deadline: 0.44-0.52 m of counter left gives ~1.0-1.7 s before the edge.
    roll_yaw_deg: float = tunable(6.0)               # uniform +/- yaw of the roll direction
    coaster_x_range: tuple = tunable((-0.22, -0.10))  # pad centre x (behind the spawn, so
    # the rolling can never crosses it by itself)
    coaster_y_range: tuple = tunable((-0.16, 0.16))   # pad centre y

    # --- info: structure ---------------------------------------------------------------------
    can_r: float = info(0.033)      # 330 ml can: radius 33 mm
    can_len: float = info(0.115)    # ... length 115 mm
    can_mass: float = info(0.04)
    counter_x0: float = info(-0.30)  # closed near end of the counter
    counter_x1: float = info(0.50)   # OPEN far edge — past here the can drops to the floor
    counter_half_w: float = info(0.30)
    counter_top: float = info(0.25)  # counter top height; the fall below is what "lost" means
    coaster_size: float = info(0.16)  # square pad side; sunk flush (top +0.2 mm — a marker,
    # not a wall: the can rolls straight over it)
    coaster_t: float = info(0.005)
    friction: float = info(0.55)     # counter+can material; the reset writes the matched
    # no-slip angular velocity, so the roll neither skids nor brakes.
    contact_offset: float = info(0.002)  # small: a fat speculative margin would make the
    # rolling can "touch" the flush pad lip and judder.

    # Derived (filled in __post_init__).
    counter_len: float = field(default=None, init=False)
    counter_cx: float = field(default=None, init=False)
    can_half: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.counter_len = round(self.counter_x1 - self.counter_x0, 4)
        self.counter_cx = round((self.counter_x0 + self.counter_x1) / 2, 4)
        self.can_half = round(self.can_len / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("runaway_can")
class RunawayCanScene(BaseScene):
    cfg: RunawayCanSceneCfg

    def __init__(self, cfg: RunawayCanSceneCfg | None = None) -> None:
        super().__init__(cfg or RunawayCanSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the raised counter slab (kinematic), the flush coaster pad
        (kinematic, re-posed by reset) and the can (dynamic cylinder, re-posed AND
        re-launched by reset)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=c.friction, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            "counter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=sim_utils.CuboidCfg(
                    size=(c.counter_len, 2 * c.counter_half_w, c.counter_top),
                    rigid_props=kin, collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.42)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.counter_cx, 0.0, c.counter_top / 2)),
            ),
            "coaster": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Coaster",
                spawn=sim_utils.CuboidCfg(
                    size=(c.coaster_size, c.coaster_size, c.coaster_t),
                    rigid_props=kin, collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.16, 0.0, c.counter_top + 0.0002 - c.coaster_t / 2)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can",
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_len,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                        linear_damping=0.0, angular_damping=0.0),  # do not brake the roll
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.10, 0.12)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.counter_top + c.can_r + 0.002),
                    rot=(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)),  # lying
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
        """Grab handles + allocate the per-episode latch buffers the rubric depends on."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.can: RigidObject = env.iscene["can"]
        self.coaster: RigidObject = env.iscene["coaster"]
        self.env_origins = env.iscene.env_origins
        self.coaster_c = torch.zeros(n, 2, device=dev)   # pad centre (env-local xy)
        self.spawn_v = torch.zeros(n, device=dev)        # sampled roll speed, for readback
        self.lost = torch.zeros(n, dtype=torch.bool, device=dev)         # fell past the edge
        self.intercepted = torch.zeros(n, dtype=torch.bool, device=dev)  # came to rest on top
        self.righted = torch.zeros(n, dtype=torch.bool, device=dev)      # stood upright on top

    def _can_local(self) -> torch.Tensor:
        """Can centre position in env-local coords, (N, 3)."""
        return self.can.data.root_pos_w - self.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the coaster pad pose, then LAUNCH the can — lying on its
        side, already rolling toward the open edge with the matched no-slip angular
        velocity (so the roll is a clean roll, not a skid); clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def u(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(m, device=dev)

        # --- coaster pad: kinematic decoration, moved by POSE write ---
        cx = u(*c.coaster_x_range)
        cy = u(*c.coaster_y_range)
        self.coaster_c[env_ids, 0] = cx
        self.coaster_c[env_ids, 1] = cy
        st = torch.zeros(m, 7, device=dev)
        st[:, 0] = cx
        st[:, 1] = cy
        st[:, 2] = c.counter_top + 0.0002 - c.coaster_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.coaster.write_root_pose_to_sim(st, env_ids)

        # --- can: lying, axis across the roll direction, launched at v0 ---
        v0 = u(*c.roll_speed_range)
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.roll_yaw_deg)
        dx, dy = torch.cos(yaw), torch.sin(yaw)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = u(*c.spawn_x_range)
        st[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.spawn_y_jitter
        st[:, 2] = c.counter_top + c.can_r + 0.002
        # q = qz(yaw) * qx(90 deg): cylinder local +z (the can axis) -> horizontal,
        # perpendicular to the roll direction d = (cos yaw, sin yaw, 0).
        half, c45 = yaw / 2, math.cos(math.pi / 4)
        st[:, 3] = torch.cos(half) * c45
        st[:, 4] = torch.cos(half) * c45
        st[:, 5] = torch.sin(half) * c45
        st[:, 6] = torch.sin(half) * c45
        st[:, 7] = v0 * dx
        st[:, 8] = v0 * dy
        # no-slip rolling: omega = (v0/r) * (z_hat x d) -> contact point velocity is zero
        st[:, 10] = -(v0 / c.can_r) * dy
        st[:, 11] = (v0 / c.can_r) * dx
        st[:, 0:3] += origin
        self.can.write_root_state_to_sim(st, env_ids)

        self.spawn_v[env_ids] = v0
        self.lost[env_ids] = False
        self.intercepted[env_ids] = False
        self.righted[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping, every physics substep: LOST when the can centre drops past
        the edge threshold (real fall physics — the only way down there is over an edge);
        INTERCEPTED when the launched can is at rest ON the counter top; RIGHTED when it
        additionally stands upright. Latches only ever set while not lost."""
        alive = ~self.lost
        p = self._can_local()
        below = p[:, 2] < self.cfg.counter_top - self.cfg.lost_below
        self.lost |= below
        rest = self.on_counter() & self.settled() & alive
        self.intercepted |= rest
        self.righted |= rest & self.upright()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "can": self.can.data.root_state_w[env_ids].clone(),
            "coaster": self.coaster.data.root_state_w[env_ids].clone(),
            "coaster_c": self.coaster_c[env_ids].clone(),
            "spawn_v": self.spawn_v[env_ids].clone(),
            "lost": self.lost[env_ids].clone(),
            "intercepted": self.intercepted[env_ids].clone(),
            "righted": self.righted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self.coaster.write_root_pose_to_sim(state["coaster"][:, 0:7], env_ids)
        self.coaster_c[env_ids] = state["coaster_c"]
        self.spawn_v[env_ids] = state["spawn_v"]
        self.lost[env_ids] = state["lost"]
        self.intercepted[env_ids] = state["intercepted"]
        self.righted[env_ids] = state["righted"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised counter ({c.counter_len:.2f} m long, top {c.counter_top * 100:.0f} cm "
            f"above the floor) with one OPEN far edge. A red drinks can "
            f"({2 * c.can_r * 1000:.0f} mm wide, {c.can_len * 1000:.0f} mm long) is ALREADY "
            f"ROLLING on its side across the counter toward that edge — at its current speed "
            f"it goes over in roughly one to one-and-a-half seconds, and a can that falls to "
            f"the floor is ruined: the episode cannot be recovered. A flat green coaster pad "
            f"({c.coaster_size * 100:.0f} cm square) lies flush on the counter behind the "
            f"can's starting point; its position changes every episode, as do the can's "
            f"start, speed and heading.\n"
            f"Goal, strictly in this order: first STOP the rolling can while it is still on "
            f"the counter; then stand it UPRIGHT (either end up); then place it standing on "
            f"the coaster pad and leave it at rest there. Lifting the can and holding it in "
            f"the air completes nothing — it must end up settled, vertical, on the pad, and "
            f"it must never have left the counter top."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def on_counter(self) -> torch.Tensor:
        """(N,) bool: the can is over the counter footprint, resting at top-surface height
        (lying: centre ~ top + r; upright: centre ~ top + len/2 — both inside the band)."""
        c = self.cfg
        p = self._can_local()
        x_ok = (p[:, 0] > c.counter_x0) & (p[:, 0] < c.counter_x1)
        y_ok = p[:, 1].abs() < c.counter_half_w
        dz = p[:, 2] - c.counter_top
        return x_ok & y_ok & (dz > 0.010) & (dz < 0.090)

    def upright(self) -> torch.Tensor:
        """(N,) bool: can axis within `upright_max_deg` of vertical, either end up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        axis = quat_apply(self.can.data.root_quat_w, ez)
        return axis[:, 2].abs() >= math.cos(math.radians(self.cfg.upright_max_deg))

    def settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.can.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.can.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def on_coaster(self) -> torch.Tensor:
        """(N,) bool: can centre within `coaster_xy_tol` of the pad centre, base standing
        at pad height (the z clause pins it standing ON the pad, not hovering or lying)."""
        c = self.cfg
        p = self._can_local()
        near = (p[:, :2] - self.coaster_c).norm(dim=-1) <= c.coaster_xy_tol
        z_ok = (p[:, 2] - (c.counter_top + c.can_half)).abs() < c.on_pad_z_tol
        return near & z_ok

    def success(self) -> torch.Tensor:
        """(N,) bool: standing upright on the coaster pad, settled, never lost. The latches
        are implied (a can settled upright on the pad latches intercept + righting), so
        success is a current-state physical predicate gated by the fall history."""
        return ~self.lost & self.upright() & self.on_coaster() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 while the can just rolls (doing nothing earns nothing)
        and 0 forever once it is lost; else 0.2 for the latched intercept + 0.3 for the
        latched righting; exactly 1.0 iff success. Max non-success score = 0.5."""
        part = 0.2 * self.intercepted.float() + 0.3 * self.righted.float()
        part = torch.where(self.lost, torch.zeros_like(part), part)
        return torch.where(self.success(), torch.ones_like(part), part)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="runaway_can", robot="null", env_spacing=4.0))
