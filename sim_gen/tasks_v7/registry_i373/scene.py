"""TotemTunnelScene — topple a tall totem, thread it lengthwise through a low wall
tunnel, and stand it back upright on the goal disc outside (sim_gen task
`registry_i373`).

Derived from simpler_env/registry (the SimplerEnv task registry: pick-coke-can,
move-near, open/close-drawer, put-in-drawer, put-on-plate/towel/basket, stack) but
STRATEGICALLY different from every task in it. Every seed task is one grasp-lift-place
(or one drawer-joint push) of an object whose orientation never matters and whose
route to the goal is free space. Here the route itself is the task: a tall TOTEM
(40 x 40 x 110 mm prism) stands inside an open-top walled PEN whose one permitted
exit is a low roofed TUNNEL through the front wall — an aperture 66 mm wide x 56 mm
tall. The totem CANNOT pass standing (110 > 56) and CANNOT pass crosswise
(110 > 66): the only admissible pose is LYING DOWN, LONG AXIS ALONG THE BORE. The
solver must (1) TOPPLE the totem so it lies aligned with the tunnel, (2) SLIDE it
lengthwise through the bore under the lintel, and (3) RE-ERECT it upright on the
green goal disc outside. Reorientation is load-bearing twice (knock down, stand
back up), and the transit is proven by an ORDER-CHAINED trajectory latch — entered
(lying at the tunnel mouth inside the pen) -> mid-bore (body spanning the whole
roofed aperture, a pose unreachable from above) -> through (clear of the wall
outside) — so the seed strategy "lift it over the wall and set it down at the goal"
earns no transit credit and cannot succeed.

Task rule (declared in describe()/instruction(), enforced by the latch chain like a
"place it inside the closed drawer" rule): the totem must LEAVE THE PEN THROUGH THE
TUNNEL. Carrying it over the open top of the walls is not a solution.

Assets are fully procedural: six KINEMATIC wall slabs (back, two sides, two front
segments flanking the aperture, and the lintel above it) posed each episode from a
randomized pen frame (xy + full yaw), one DYNAMIC terracotta totem, one KINEMATIC
green goal disc outside the tunnel on the pen axis.

Per-episode randomization (readback-verifiable): pen xy jitter + full yaw, totem
start position inside the pen (xy + free yaw, standing), goal-disc position in a
pen-local window beyond the tunnel exit.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.15 * lying    — totem ever at rest lying down inside the pen (latched)
  0.30 * progress — running-max transit progress through the tunnel, gated on the
                    entered latch (lying, centred on the bore axis)
  0.10 * through  — the order chain entered -> mid-bore -> through completed
  0.15 * near     — after through, totem centre ever within 10 cm of the goal disc
  1.0 iff success() — through (physically threaded), totem standing upright ON the
                    goal disc (base at disc-top height, centre within the disc),
                    settled and finite. Non-success capped at 0.70.

Heavy imports (isaaclab) are deferred so importing this module — and registering
the scene — stays app-free.
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
class TotemTunnelSceneCfg(BaseCfg):
    """Config for `TotemTunnelScene`. The geometric claims the task rests on are
    honest by construction (asserted in __post_init__): the totem cannot pass the
    aperture standing or crosswise, DOES pass lying lengthwise with real clearance,
    has room to topple inside the pen, and the mid-bore latch band is a pose in
    which the body spans the whole roofed aperture (unreachable from above)."""

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pen_jitter: float = tunable(0.040)      # pen centre xy jitter (+/- m)
    pen_yaw_deg: float = tunable(180.0)     # pen yaw (+/- deg, full)
    totem_x_lo: float = tunable(-0.080)     # totem spawn window, pen-local x
    totem_x_hi: float = tunable(0.040)
    totem_y_amp: float = tunable(0.080)     # totem spawn |pen-local y| bound
    pad_x_lo: float = tunable(0.370)        # goal disc window, pen-local x
    pad_x_hi: float = tunable(0.450)
    pad_y_amp: float = tunable(0.060)       # goal disc |pen-local y| bound

    # --- tunable: rubric thresholds --------------------------------------------------------------
    lying_axis_z: float = tunable(0.35)     # |long-axis . z| below this = lying
    upright_max_deg: float = tunable(12.0)  # long axis within this of vertical = upright
    gate_y: float = tunable(0.045)          # |pen-local y| gate for transit latches
    gate_z: float = tunable(0.040)          # centre height gate for transit latches (m)
    mid_gate_y: float = tunable(0.020)      # tighter y gate for the mid-bore latch
    goal_r: float = tunable(0.045)          # base centre within this of the disc centre (m)
    base_z_tol: float = tunable(0.006)      # |base height - disc top| tolerance (m)
    settle_lin: float = tunable(0.06)       # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(0.60)       # max |ang vel| when judging success (rad/s)
    latch_speed: float = tunable(0.25)      # max |lin vel| for the lying latch to arm
    near_r: float = tunable(0.10)           # near-goal latched credit radius (m)

    # --- info: pen (open-top walled enclosure) ---------------------------------------------------
    pen_pos: tuple = info((0.30, 0.00))     # pen centre (nominal, world)
    interior_half: float = info(0.130)      # interior half-extent (260 x 260 mm inside)
    wall_t: float = info(0.030)             # wall thickness
    wall_h: float = info(0.120)             # wall height
    tun_w: float = info(0.066)              # tunnel aperture width  (pen-local y)
    tun_h: float = info(0.056)              # tunnel aperture height (clear under lintel)
    wall_color: tuple = info((0.45, 0.47, 0.52))
    lintel_color: tuple = info((0.35, 0.37, 0.42))
    # --- info: totem -----------------------------------------------------------------------------
    totem_a: float = info(0.040)            # square cross-section side
    totem_h: float = info(0.110)            # length (long axis = body z)
    totem_mass: float = info(0.150)
    totem_color: tuple = info((0.72, 0.36, 0.22))  # terracotta
    # --- info: goal disc -------------------------------------------------------------------------
    pad_r: float = info(0.060)
    pad_t: float = info(0.010)
    pad_color: tuple = info((0.10, 0.55, 0.18))
    # --- info: transit latch bands (pen-local x of the totem centre) -----------------------------
    x_start: float = info(0.075)            # progress origin / entered band lo
    enter_hi: float = info(0.105)           # entered band hi (inside the pen, at the mouth)
    mid_lo: float = info(0.140)             # mid-bore band (body spans the whole aperture)
    mid_hi: float = info(0.150)
    through_x: float = info(0.220)          # centre past this (lying, on axis) = through
    # --- info: rubric weights (sum = 0.70 = the non-success cap) ---------------------------------
    w_lying: float = info(0.15)
    w_prog: float = info(0.30)
    w_through: float = info(0.10)
    w_near: float = info(0.15)

    def __post_init__(self) -> None:
        # Honesty-by-construction asserts (the claims describe() makes).
        assert self.totem_h > self.tun_h + 0.03, \
            "the totem must NOT fit through the tunnel standing up"
        assert self.totem_h > self.tun_w + 0.03, \
            "the totem must NOT fit through the tunnel lying crosswise"
        assert self.totem_a < self.tun_h - 0.012 and self.totem_a < self.tun_w - 0.020, \
            "lying lengthwise the totem must pass with real clearance"
        assert 2 * self.interior_half > self.totem_h + self.totem_a + 0.025, \
            "the pen interior must leave room to topple the totem flat"
        # the mid-bore band is a pose in which the body spans the whole roofed bore
        bore_in = self.interior_half                    # inner face of the front wall
        bore_out = self.interior_half + self.wall_t     # outer face
        assert self.mid_lo - self.totem_h / 2 < bore_in - 0.004, \
            "at mid-bore the totem tail must still be inside the pen"
        assert self.mid_hi + self.totem_h / 2 > bore_out + 0.004, \
            "at mid-bore the totem nose must already be outside"
        assert self.x_start < self.enter_hi < self.mid_lo < self.mid_hi < self.through_x
        assert self.enter_hi < bore_in - self.totem_a / 2, \
            "the entered band must lie fully inside the pen"
        assert self.through_x - self.totem_h / 2 > bore_out, \
            "at through_x the totem must be clear of the wall"
        # the goal disc sits beyond the push run-out, never inside the slide path
        assert self.pad_x_lo - self.pad_r > self.through_x + self.totem_h / 2, \
            "the goal disc must be clear of the tunnel exit run-out"
        # upright cone is comfortably inside the critical tip angle atan(a/h)
        crit = math.degrees(math.atan(self.totem_a / self.totem_h))
        assert self.upright_max_deg < crit - 5.0, \
            "the upright cone must be well inside the totem's critical tip angle"
        assert abs(self.w_lying + self.w_prog + self.w_through + self.w_near - 0.70) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("totem_tunnel")
class TotemTunnelScene(BaseScene):
    cfg: TotemTunnelSceneCfg

    def __init__(self, cfg: TotemTunnelSceneCfg | None = None) -> None:
        super().__init__(cfg or TotemTunnelSceneCfg())

    # ----- wall layout (pen-local): name -> (centre xyz, size xyz) ------------------------------
    def _wall_specs(self) -> dict[str, tuple[tuple, tuple]]:
        c = self.cfg
        ih, wt, wh = c.interior_half, c.wall_t, c.wall_h
        fx = ih + wt / 2                                  # front/back wall centre |x|
        seg_len = (ih + wt) - c.tun_w / 2                 # front segment y-length
        seg_cy = c.tun_w / 2 + seg_len / 2
        lin_h = wh - c.tun_h                              # lintel z-size
        lin_cz = c.tun_h + lin_h / 2
        return {
            "wall_back": ((-fx, 0.0, wh / 2), (wt, 2 * (ih + wt), wh)),
            "wall_left": ((0.0, ih + wt / 2, wh / 2), (2 * ih, wt, wh)),
            "wall_right": ((0.0, -(ih + wt / 2), wh / 2), (2 * ih, wt, wh)),
            "wall_front_l": ((fx, seg_cy, wh / 2), (wt, seg_len, wh)),
            "wall_front_r": ((fx, -seg_cy, wh / 2), (wt, seg_len, wh)),
            "lintel": ((fx, 0.0, lin_cz), (wt, c.tun_w + 0.004, lin_h)),
        }

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        kin_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
        )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "totem": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Totem",
                spawn=sim_utils.CuboidCfg(
                    size=(c.totem_a, c.totem_a, c.totem_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.1,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=1),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.totem_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.totem_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, c.totem_h / 2 + 0.002)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GoalDisc",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_t,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, c.pad_t / 2)),
            ),
        }
        for name, (ctr, size) in self._wall_specs().items():
            color = c.lintel_color if name == "lintel" else c.wall_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pen_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **kin_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pen_pos[0] + ctr[0], c.pen_pos[1] + ctr[1], ctr[2])),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.totem: RigidObject = env.iscene["totem"]
        self.pad: RigidObject = env.iscene["pad"]
        self.walls: dict[str, RigidObject] = {
            n: env.iscene[n] for n in self._wall_specs()}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # pen frame (world, incl. env origin) + pad centre, refreshed each reset
        self.pen_pos_w = torch.zeros(n, 3, device=dev)
        self.pen_yaw = torch.zeros(n, device=dev)
        self.pad_pos_w = torch.zeros(n, 3, device=dev)
        # latches (partial credit survives transients; success is judged live except
        # for the transit chain, which is inherently a trajectory property)
        self._lying = torch.zeros(n, dtype=torch.bool, device=dev)
        self._entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self._mid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._through = torch.zeros(n, dtype=torch.bool, device=dev)
        self._near = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prog = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the pen (xy jitter + full yaw) and its six wall slabs,
        stand the totem inside (pen-local window, free yaw), place the goal disc
        outside beyond the tunnel, clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(8, device=dev)  # burn: first post-seed draws are degenerate

        pen_xy = torch.tensor(c.pen_pos, device=dev).expand(m, 2).clone()
        pen_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.pen_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pen_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        self.pen_pos_w[env_ids, 0] = pen_xy[:, 0] + origin[:, 0]
        self.pen_pos_w[env_ids, 1] = pen_xy[:, 1] + origin[:, 1]
        self.pen_pos_w[env_ids, 2] = origin[:, 2]
        self.pen_yaw[env_ids] = yaw
        qpen = _qz(yaw)

        def to_world(lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
            wx = pen_xy[:, 0] + cy * lx - sy * ly + origin[:, 0]
            wy = pen_xy[:, 1] + sy * lx + cy * ly + origin[:, 1]
            return torch.stack([wx, wy], dim=-1)

        # --- walls: kinematic slabs posed from the pen frame ---
        for name, (ctr, _size) in self._wall_specs().items():
            st = torch.zeros(m, 13, device=dev)
            lx = torch.full((m,), ctr[0], device=dev)
            ly = torch.full((m,), ctr[1], device=dev)
            st[:, 0:2] = to_world(lx, ly)
            st[:, 2] = ctr[2] + origin[:, 2]
            st[:, 3:7] = qpen
            self.walls[name].write_root_state_to_sim(st, env_ids)

        # --- totem: standing inside the pen, free yaw ---
        tx = c.totem_x_lo + torch.rand(m, device=dev) * (c.totem_x_hi - c.totem_x_lo)
        ty = (torch.rand(m, device=dev) * 2 - 1) * c.totem_y_amp
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = to_world(tx, ty)
        st[:, 2] = c.totem_h / 2 + 0.002 + origin[:, 2]
        st[:, 3:7] = _qmul(qpen, _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))
        self.totem.write_root_state_to_sim(st, env_ids)

        # --- goal disc: pen-local window beyond the tunnel exit ---
        px = c.pad_x_lo + torch.rand(m, device=dev) * (c.pad_x_hi - c.pad_x_lo)
        py = (torch.rand(m, device=dev) * 2 - 1) * c.pad_y_amp
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = to_world(px, py)
        st[:, 2] = c.pad_t / 2 + origin[:, 2]
        st[:, 3:7] = qpen
        self.pad.write_root_state_to_sim(st, env_ids)
        self.pad_pos_w[env_ids] = st[:, 0:3]

        # --- clear latches ---
        for t in (self._lying, self._entered, self._mid, self._through, self._near):
            t[env_ids] = False
        self._prog[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "totem": self.totem.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "walls": {n: w.data.root_state_w[env_ids].clone()
                      for n, w in self.walls.items()},
            "pen_pos_w": self.pen_pos_w[env_ids].clone(),
            "pen_yaw": self.pen_yaw[env_ids].clone(),
            "pad_pos_w": self.pad_pos_w[env_ids].clone(),
            "lying": self._lying[env_ids].clone(),
            "entered": self._entered[env_ids].clone(),
            "mid": self._mid[env_ids].clone(),
            "through": self._through[env_ids].clone(),
            "near": self._near[env_ids].clone(),
            "prog": self._prog[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.totem.write_root_state_to_sim(state["totem"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        for n, w in self.walls.items():
            w.write_root_state_to_sim(state["walls"][n], env_ids)
        self.pen_pos_w[env_ids] = state["pen_pos_w"]
        self.pen_yaw[env_ids] = state["pen_yaw"]
        self.pad_pos_w[env_ids] = state["pad_pos_w"]
        self._lying[env_ids] = state["lying"]
        self._entered[env_ids] = state["entered"]
        self._mid[env_ids] = state["mid"]
        self._through[env_ids] = state["through"]
        self._near[env_ids] = state["near"]
        self._prog[env_ids] = state["prog"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An open-top square PEN of four gray walls stands on the ground "
            f"({2 * c.interior_half * 100:.0f} cm x {2 * c.interior_half * 100:.0f} cm "
            f"inside, walls {c.wall_h * 100:.0f} cm tall and {c.wall_t * 1000:.0f} mm "
            f"thick); its position and heading vary per episode. The FRONT wall has a "
            f"low roofed TUNNEL through it at floor level: an aperture "
            f"{c.tun_w * 1000:.0f} mm wide and {c.tun_h * 1000:.0f} mm tall under a "
            f"darker lintel. Inside the pen stands a terracotta TOTEM — a square "
            f"prism {c.totem_a * 1000:.0f} x {c.totem_a * 1000:.0f} x "
            f"{c.totem_h * 1000:.0f} mm, upright on its base, position and heading "
            f"randomized. Outside, beyond the tunnel on the pen's axis, a flat green "
            f"GOAL DISC (radius {c.pad_r * 100:.0f} cm, {c.pad_t * 1000:.0f} mm "
            f"thick) lies on the ground.\n"
            f"Task rule: the totem must leave the pen THROUGH THE TUNNEL — carrying "
            f"it over the open top of the walls is not a solution and earns nothing. "
            f"The totem cannot pass the tunnel standing up "
            f"({c.totem_h * 1000:.0f} > {c.tun_h * 1000:.0f} mm) and cannot pass "
            f"lying crosswise ({c.totem_h * 1000:.0f} > {c.tun_w * 1000:.0f} mm); it "
            f"passes ONLY lying down with its long axis along the tunnel. So: "
            f"(1) TOPPLE the totem inside the pen so it lies flat, long axis aimed "
            f"through the tunnel; (2) SLIDE it lengthwise through the aperture under "
            f"the lintel until it is fully outside the wall; (3) STAND it back "
            f"upright with its base ON the green goal disc (base centred within "
            f"{c.goal_r * 1000:.0f} mm of the disc centre, tilt under "
            f"{c.upright_max_deg:.0f} degrees). Success: the totem physically "
            f"threaded the tunnel lying down and now stands upright, at rest, on the "
            f"goal disc."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Tip the terracotta totem over inside the pen, slide it lying lengthwise "
            "out through the low tunnel in the front wall, then stand it upright on "
            "the green goal disc outside. It must exit through the tunnel, not over "
            "the walls."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _pen_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> pen-local (x, y, z_rel) as (N,3)."""
        rel = pos_w - self.pen_pos_w
        cy, sy = torch.cos(self.pen_yaw), torch.sin(self.pen_yaw)
        xl = cy * rel[:, 0] + sy * rel[:, 1]
        yl = -sy * rel[:, 0] + cy * rel[:, 1]
        return torch.stack([xl, yl, rel[:, 2]], dim=-1)

    def _axis_w(self) -> torch.Tensor:
        """(N,3) world direction of the totem's long (body-z) axis."""
        from isaaclab.utils.math import quat_apply

        q = self.totem.data.root_quat_w
        ez = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, ez)

    def lying(self) -> torch.Tensor:
        return self._axis_w()[:, 2].abs() < self.cfg.lying_axis_z

    def upright(self) -> torch.Tensor:
        return self._axis_w()[:, 2] > math.cos(math.radians(self.cfg.upright_max_deg))

    def on_pad_upright(self) -> torch.Tensor:
        """(N,) bool, live: totem upright, base at disc-top height, base centre
        within `goal_r` of the disc centre."""
        c = self.cfg
        p = self.totem.data.root_pos_w
        near = (p[:, :2] - self.pad_pos_w[:, :2]).norm(dim=-1) < c.goal_r
        base_z = p[:, 2] - c.totem_h / 2 - (self.pad_pos_w[:, 2] + c.pad_t / 2)
        return self.upright() & near & (base_z.abs() < c.base_z_tol)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.totem.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.totem.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _update_latches(self) -> None:
        c = self.cfg
        p = self.totem.data.root_pos_w
        finite = torch.isfinite(p).all(dim=-1)
        loc = self._pen_local(p)
        xl, yl, zl = loc[:, 0], loc[:, 1], loc[:, 2]
        lying = self.lying() & finite
        calm = self.totem.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        inside = (xl.abs() < c.interior_half) & (yl.abs() < c.interior_half)
        self._lying |= lying & calm & inside
        # order-chained transit latches (lying, centred on the bore axis, low)
        gate = lying & (yl.abs() < c.gate_y) & (zl < c.gate_z)
        self._entered |= gate & (xl > c.x_start) & (xl < c.enter_hi)
        self._mid |= self._entered & gate & (yl.abs() < c.mid_gate_y) \
            & (xl > c.mid_lo) & (xl < c.mid_hi)
        self._through |= self._mid & gate & (xl > c.through_x)
        # transit progress: running max, gated on the entered latch
        prog = ((xl - c.x_start) / (c.through_x - c.x_start)).clamp(0.0, 1.0)
        prog = torch.where(self._entered & gate, prog, torch.zeros_like(prog))
        self._prog = torch.maximum(self._prog, prog)
        # near-goal credit only after a real threading
        near = (p[:, :2] - self.pad_pos_w[:, :2]).norm(dim=-1) < c.near_r
        self._near |= self._through & near & finite

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the totem physically threaded the tunnel (order-chained
        trajectory latch entered -> mid-bore -> through) and now stands upright,
        settled, on the goal disc. The standing clauses are live; the transit clause
        is a trajectory property and is necessarily latched."""
        self._update_latches()
        finite = torch.isfinite(self.totem.data.root_pos_w).all(dim=-1)
        return self._through & self.on_pad_upright() & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15*lying + 0.30*progress + 0.10*through +
        0.15*near (latched; ~0 for doing nothing since the totem spawns standing),
        capped at 0.70 — and exactly 1.0 iff success() holds."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lying * self._lying.float()
                + c.w_prog * self._prog
                + c.w_through * self._through.float()
                + c.w_near * self._near.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="totem_tunnel", robot="null"))
