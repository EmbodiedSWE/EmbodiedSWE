"""BayonetSwitchScene — arm a machine by seating a spring-loaded captive knob through a
bayonet (L-slot) interlock (seed: rlbench/press_switch, strategically inverted).

The seed task is a momentary fingertip flick: reach the wall switch, poke its lever
through a small arc, done. Here the "switch" is a child-safety / bayonet interlock that a
poke CANNOT actuate: an orange captive knob rides in an L-shaped guide channel milled
into a console block, preloaded against its OFF stop by a return spring that acts along
the channel's press axis. A straight press — the seed's entire plan — travels the press
leg and springs straight back the instant the finger leaves. To switch the machine on,
the solver must execute a COMPOUND, ORDERED stroke that the walls themselves enforce:
(1) press the knob the full depth of the press leg, (2) sweep it sideways along the cross
leg (whose direction — left or right of the press axis — is sampled per episode), then
release so the spring seats the knob into a locking notch cut back toward the OFF end.
The notch shelf then blocks the return path: the goal state is held MECHANICALLY, by
geometry against the live spring, not by a code latch. Success is the settled, seated
knob — a physical outcome that persists.

Mechanism (all procedural primitives, no joints): the channel is 8 kinematic wall boxes
+ a kinematic base slab, each re-posed per reset from the sampled parameters (press-leg
length, mirror handedness, console position + yaw) — the jointless-guide-wall pattern
proven in sibling suites. The knob is a free dynamic cylinder (rotation-proof footprint:
it can never wedge in the channel the way a yawed box could); the return spring is a
post_step external force f_x = k*(home - x) - c*v expressed in the console's canonical
frame, plus a small lateral damper. Low-friction materials keep partial strokes honest:
anywhere on the press leg, release -> the knob glides back to the OFF stop.

Anti-cheat: the knob is nominally captive (a real bayonet knob has a flange under the
slot lip); the channel top is left open only so the task reads on camera. A permanent
`escaped` latch trips if the knob ever rises above the wall tops — lifting it out and
dropping it into the notch from above voids the episode (success requires ~escaped and
score is capped). All progress latches are gated on the knob moving IN the channel
plane.

score() in [0, 1]: 0..0.25 latched with the deepest press-leg excursion, 0.25..0.55
latched with the farthest cross-leg sweep (only reachable at full depth — the geometry
orders the stages), 0.70 latched once the knob has entered the notch, 1.0 iff success
(knob settled in the notch, held by the spring, sustained). Doing nothing scores 0.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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
class BayonetSwitchSceneCfg(BaseCfg):
    """Config for `BayonetSwitchScene`. All geometry lives in the console's CANONICAL frame:
    origin at the knob's OFF home, press leg along -x, cross leg along +y (a sampled mirror
    flips it to -y), spring return along +x. Wall boxes have FIXED sizes; per-reset the
    sampled press-leg length slides their centers (tails always point outside the channel)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    seat_x_margin: float = tunable(0.006)  # knob center past the notch mouth by this = "in notch"
    seat_speed: float = tunable(0.02)  # max |v| for the seated knob (m/s)
    seat_hold_steps: int = tunable(30)  # consecutive substeps the seat must be sustained
    escape_z: float = tunable(0.045)  # knob center above this (console-local) latches `escaped`
    plane_z_tol: float = tunable(0.008)  # |z - rest| within this = knob riding in the channel
    frac_deadband: float = tunable(0.05)  # progress fractions below this score nothing

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    press_leg_range: tuple = tunable((0.055, 0.080))  # press-leg length L (m), sampled U
    console_jitter: float = tunable(0.05)  # console xy jitter around console_pos (m)
    console_yaw_deg: float = tunable(180.0)  # console yaw, uniform +/- this
    mirror_prob: float = tunable(0.5)  # P(cross leg flipped to -y)

    # --- tunable: placement ------------------------------------------------------------------
    console_pos: tuple = tunable((0.0, 0.0))  # canonical-origin nominal spot on the ground

    # --- info: channel geometry (canonical frame; see class docstring) ------------------------
    knob_r: float = info(0.016)  # knob cylinder radius (m)
    knob_h: float = info(0.030)  # knob cylinder height
    knob_mass: float = info(0.06)
    channel_w: float = info(0.046)  # slot width -> 7 mm clearance per side around the knob
    wall_t: float = info(0.012)
    wall_h: float = info(0.035)  # wall tops at base_t + wall_h = 47 mm
    base_t: float = info(0.012)
    base_size: tuple = info((0.240, 0.200))  # base slab footprint (canonical x, y)
    base_center: tuple = info((-0.045, 0.030))
    cross_len: float = info(0.060)  # cross-leg length (home-axis to notch-band center)
    notch_len: float = info(0.028)  # notch depth back toward the OFF end (+x)
    home_x: float = info(0.006)  # spring rest point BEHIND the OFF stop -> ~0.06 N preload
    spring_k: float = info(10.0)  # return spring (N/m); full-depth pull ~0.9 N
    spring_c: float = info(0.8)  # press-axis damping (N*s/m), ~half critical for the knob
    lateral_c: float = info(0.4)  # cross-axis damper (kills slosh, no restoring force)
    friction: tuple = info((0.04, 0.03))  # static, dynamic — partial strokes must glide back
    contact_offset: float = info(0.0015)
    knob_color: tuple = info((0.95, 0.45, 0.08))
    wall_color: tuple = info((0.30, 0.33, 0.40))
    base_color: tuple = info((0.20, 0.22, 0.27))
    tile_on_color: tuple = info((0.10, 0.85, 0.15))  # green ON tile beside the notch
    tile_off_color: tuple = info((0.85, 0.12, 0.10))  # red OFF tile beside the home stop

    # Derived (filled in __post_init__).
    knob_rest_z: float = field(default=None, init=False)  # knob center z when riding the base
    y_notch: float = field(default=None, init=False)  # notch-band center y (canonical)
    x_stop: float = field(default=None, init=False)  # OFF stop interior face

    def __post_init__(self) -> None:
        self.knob_rest_z = round(self.base_t + self.knob_h / 2, 4)
        self.y_notch = self.cross_len
        self.x_stop = self.knob_r

    # L-dependent interior faces (canonical frame), L = press-leg length of the episode.
    def x_west(self, L: float) -> float:
        return -L - self.knob_r

    def x_notch_mouth(self, L: float) -> float:  # the cross leg's east face = notch entrance
        return -L - self.knob_r + self.channel_w

    def x_notch_end(self, L: float) -> float:
        return self.x_notch_mouth(L) + self.notch_len


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bayonet_switch")
class BayonetSwitchScene(BaseScene):
    cfg: BayonetSwitchSceneCfg

    def __init__(self, cfg: BayonetSwitchSceneCfg | None = None) -> None:
        super().__init__(cfg or BayonetSwitchSceneCfg())

    # Fixed wall-box sizes (x, y, z) — centers are computed per reset. Sizes are symmetric,
    # so the mirror is a pure center-reflection (y -> -y) with unchanged orientation.
    def _wall_specs(self) -> dict[str, tuple]:
        c = self.cfg
        t, h = c.wall_t, c.wall_h
        return {
            "wall_s": (0.150, t, h),  # south of the press leg
            "wall_stop": (t, 0.070, h),  # the OFF end stop
            "wall_an": (0.090, t, h),  # north of the press leg (east of the cross leg)
            "wall_w": (t, 0.140, h),  # west face: press-leg end + cross-leg outer wall
            "wall_e1": (t, 0.014, h),  # cross-leg east face between press leg and notch shelf
            "wall_shelf": (0.048, t, h),  # the notch shelf — blocks the seated knob's return
            "wall_ne": (t, 0.070, h),  # notch end stop
            "wall_nw": (0.110, t, h),  # north face over cross leg + notch
        }

    def _wall_centers(self, L: torch.Tensor) -> dict[str, tuple]:
        """Canonical (x, y, z) center of each wall for press-leg length `L` (per-env tensor).
        Every variable-span wall is anchored so its tails point OUTSIDE the channel."""
        c = self.cfg
        t = c.wall_t
        w2 = c.channel_w / 2  # 0.023: interior half-width
        zc = c.base_t + c.wall_h / 2
        xm = -L - c.knob_r + c.channel_w  # notch mouth face x_e1
        return {
            "wall_s": (-L / 2, -w2 - t / 2, zc),
            "wall_stop": (c.x_stop + t / 2 + 0.0 * L, 0.0 * L, zc),
            "wall_an": (xm + 0.045, w2 + t / 2 + 0.0 * L, zc),
            "wall_w": (-L - c.knob_r - t / 2, 0.030 + 0.0 * L, zc),
            "wall_e1": (xm + t / 2, 0.030 + 0.0 * L, zc),
            "wall_shelf": (xm + 0.024, c.y_notch - w2 - t / 2 + 0.0 * L, zc),
            "wall_ne": (xm + c.notch_len + t / 2, c.y_notch + 0.0 * L, zc),
            "wall_nw": (-L + 0.021, c.y_notch + w2 + t / 2 + 0.0 * L, zc),
        }

    def _tile_centers(self, L: torch.Tensor) -> dict[str, tuple]:
        c = self.cfg
        zc = c.base_t + 0.002
        xm = -L - c.knob_r + c.channel_w
        return {
            "tile_on": (xm + c.notch_len + c.wall_t + 0.020, c.y_notch + 0.0 * L, zc),
            "tile_off": (0.048 + 0.0 * L, 0.0 * L, zc),
        }

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic base slab + 8 channel walls + 2 indicator tiles, and the
        dynamic knob at its OFF home (reset() re-places everything from the sampled episode)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction[0], dynamic_friction=c.friction[1], restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)

        def kin_box(size, color, pos):
            return RigidObjectCfg(
                prim_path=None,  # set by caller
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )

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
        }

        L0 = torch.tensor(sum(c.press_leg_range) / 2)  # nominal layout; reset re-places all
        specs = self._wall_specs()
        centers = self._wall_centers(L0)
        out["base"] = kin_box((c.base_size[0], c.base_size[1], c.base_t), c.base_color,
                              (c.base_center[0], c.base_center[1], c.base_t / 2))
        out["base"].prim_path = "{ENV_REGEX_NS}/Base"
        for name, size in specs.items():
            cx, cy, cz = (float(v) for v in centers[name])
            out[name] = kin_box(size, c.wall_color, (cx, cy, cz))
            out[name].prim_path = "{ENV_REGEX_NS}/" + name.title().replace("_", "")
        for name, color in (("tile_on", c.tile_on_color), ("tile_off", c.tile_off_color)):
            cx, cy, cz = (float(v) for v in self._tile_centers(L0)[name])
            out[name] = kin_box((0.030, 0.030, 0.004), color, (cx, cy, cz))
            out[name].prim_path = "{ENV_REGEX_NS}/" + name.title().replace("_", "")

        out["knob"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Knob",
            spawn=sim_utils.CylinderCfg(
                radius=c.knob_r, height=c.knob_h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.knob_mass),
                collision_props=coll, physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.knob_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.001, 0.0, c.knob_rest_z)),
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
        """Grab handles + allocate the per-env episode parameters and latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.knob: RigidObject = env.iscene["knob"]
        self.base: RigidObject = env.iscene["base"]
        self.walls: dict[str, RigidObject] = {
            name: env.iscene[name] for name in self._wall_specs()}
        self.tiles: dict[str, RigidObject] = {
            name: env.iscene[name] for name in ("tile_on", "tile_off")}
        self.env_origins = env.iscene.env_origins
        # Sampled episode parameters (canonical-frame layout).
        self._console = torch.zeros(n, 2, device=dev)  # canonical origin, env-local xy
        self._yaw = torch.zeros(n, device=dev)
        self._mirror = torch.ones(n, device=dev)  # +1 = cross leg to +y, -1 = mirrored
        self._L = torch.full((n,), sum(c.press_leg_range) / 2, device=dev)
        # Latches + the sustained-seat counter.
        self._depth_max = torch.zeros(n, device=dev)
        self._cross_max = torch.zeros(n, device=dev)
        self._notched = torch.zeros(n, dtype=torch.bool, device=dev)
        self._escaped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold = torch.zeros(n, dtype=torch.long, device=dev)

    # --- frame plumbing ---------------------------------------------------------------------
    def local_to_world_xy(self, x_c: torch.Tensor, y_c: torch.Tensor,
                          env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical (x, y) -> env-local world xy (mirror then yaw then translate)."""
        ids = slice(None) if env_ids is None else env_ids
        cy, sy = torch.cos(self._yaw[ids]), torch.sin(self._yaw[ids])
        ym = y_c * self._mirror[ids]
        return (self._console[ids, 0] + cy * x_c - sy * ym,
                self._console[ids, 1] + sy * x_c + cy * ym)

    def knob_canon(self) -> tuple[torch.Tensor, ...]:
        """Knob state in the canonical frame: (x, y, z_local, vx, vy, |v|)."""
        p = self.knob.data.root_pos_w - self.env_origins
        v = self.knob.data.root_lin_vel_w
        cy, sy = torch.cos(self._yaw), torch.sin(self._yaw)
        dx, dy = p[:, 0] - self._console[:, 0], p[:, 1] - self._console[:, 1]
        x_c = cy * dx + sy * dy
        y_c = (-sy * dx + cy * dy) * self._mirror
        vx_c = cy * v[:, 0] + sy * v[:, 1]
        vy_c = (-sy * v[:, 0] + cy * v[:, 1]) * self._mirror
        return x_c, y_c, p[:, 2], vx_c, vy_c, v.norm(dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample (console xy + yaw, mirror handedness, press-leg length),
        re-pose the whole kinematic channel from those parameters, seat the knob at its OFF
        home, clear latches, zero the spring's force buffer."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)

        self._console[env_ids, 0] = c.console_pos[0] \
            + (torch.rand(m, device=dev) * 2 - 1) * c.console_jitter
        self._console[env_ids, 1] = c.console_pos[1] \
            + (torch.rand(m, device=dev) * 2 - 1) * c.console_jitter
        self._yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) \
            * math.radians(c.console_yaw_deg)
        self._mirror[env_ids] = torch.where(
            torch.rand(m, device=dev) < c.mirror_prob,
            torch.full((m,), -1.0, device=dev), torch.ones(m, device=dev))
        lo, hi = c.press_leg_range
        self._L[env_ids] = lo + torch.rand(m, device=dev) * (hi - lo)

        L = self._L[env_ids]
        qw = torch.cos(self._yaw[env_ids] / 2)
        qz = torch.sin(self._yaw[env_ids] / 2)
        origin = self.env_origins[env_ids]

        def write(body, x_c, y_c, z_c) -> None:
            st = torch.zeros(m, 13, device=dev)
            wx, wy = self.local_to_world_xy(x_c + 0.0 * L, y_c + 0.0 * L, env_ids)
            st[:, 0], st[:, 1], st[:, 2] = wx, wy, z_c + 0.0 * L
            st[:, 3], st[:, 6] = qw, qz
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.base, torch.full_like(L, c.base_center[0]),
              torch.full_like(L, c.base_center[1]), torch.full_like(L, c.base_t / 2))
        centers = self._wall_centers(L)
        for name, body in self.walls.items():
            cx, cy_, cz = centers[name]
            write(body, cx + 0.0 * L, cy_ + 0.0 * L, cz + 0.0 * L)
        tiles = self._tile_centers(L)
        for name, body in self.tiles.items():
            cx, cy_, cz = tiles[name]
            write(body, cx + 0.0 * L, cy_ + 0.0 * L, cz + 0.0 * L)
        write(self.knob, torch.full_like(L, -0.001), torch.zeros_like(L),
              torch.full_like(L, c.knob_rest_z))

        self._depth_max[env_ids] = 0.0
        self._cross_max[env_ids] = 0.0
        self._notched[env_ids] = False
        self._escaped[env_ids] = False
        self._hold[env_ids] = 0
        n = self.env.num_envs
        self.knob.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), torch.zeros(n, 1, 3, device=dev))

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs
        x_c, y_c, z, vx_c, vy_c, speed = self.knob_canon()

        # Return spring along the canonical press axis (+x toward home) + lateral damper.
        # NOTE set_external_force_and_torque applies forces in the BODY frame, so the
        # world-frame spring force must be rotated by the knob's own (free) orientation.
        from isaaclab.utils.math import quat_apply_inverse

        fx_c = c.spring_k * (c.home_x - x_c) - c.spring_c * vx_c
        fy_c = -c.lateral_c * vy_c
        cy, sy = torch.cos(self._yaw), torch.sin(self._yaw)
        fym = fy_c * self._mirror
        f_w = torch.zeros(n, 3, device=dev)
        f_w[:, 0] = cy * fx_c - sy * fym
        f_w[:, 1] = sy * fx_c + cy * fym
        f_b = quat_apply_inverse(self.knob.data.root_quat_w, f_w)
        self.knob.set_external_force_and_torque(
            f_b.view(n, 1, 3), torch.zeros(n, 1, 3, device=dev))

        # Latches, all gated on the knob riding IN the channel plane.
        in_plane = (z - c.knob_rest_z).abs() < c.plane_z_tol
        self._escaped |= z > c.escape_z
        depth = ((-x_c) / self._L).clamp(0.0, 1.0)
        self._depth_max = torch.maximum(self._depth_max, torch.where(
            in_plane, depth, torch.zeros_like(depth)))
        at_depth = x_c < -self._L + 0.020
        cross = (y_c / c.cross_len).clamp(0.0, 1.0)
        self._cross_max = torch.maximum(self._cross_max, torch.where(
            in_plane & at_depth, cross, torch.zeros_like(cross)))
        x_mouth = -self._L - c.knob_r + c.channel_w
        in_notch = in_plane & (x_c > x_mouth + c.seat_x_margin) \
            & ((y_c - c.y_notch).abs() < 0.020)
        self._notched |= in_notch & ~self._escaped
        seated_now = in_notch & (speed < c.seat_speed) & ~self._escaped
        self._hold = torch.where(seated_now, self._hold + 1, torch.zeros_like(self._hold))

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "knob": self.knob.data.root_state_w[env_ids].clone(),
            "walls": {name: b.data.root_state_w[env_ids].clone()
                      for name, b in {**self.walls, **self.tiles, "base": self.base}.items()},
            "params": {k: getattr(self, k)[env_ids].clone()
                       for k in ("_console", "_yaw", "_mirror", "_L")},
            "latches": {k: getattr(self, k)[env_ids].clone()
                        for k in ("_depth_max", "_cross_max", "_notched", "_escaped", "_hold")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {**self.walls, **self.tiles, "base": self.base}
        for name, b in bodies.items():
            b.write_root_state_to_sim(state["walls"][name], env_ids)
        self.knob.write_root_state_to_sim(state["knob"], env_ids)
        for group in ("params", "latches"):
            for k, v in state[group].items():
                getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A console block on the ground carries a safety interlock switch: a bright-orange "
            f"knob ({2 * c.knob_r * 1000:.0f} mm across) captive in an L-shaped guide slot. A "
            "return spring preloads the knob against its OFF stop (beside the red tile). The "
            "slot runs a press leg away from the stop, then turns into a cross leg — to the "
            "left or the right, it varies — ending at a locking notch cut back toward the OFF "
            "end, beside the green ON tile.\n"
            "Goal: switch the machine ON by seating the knob in the locking notch so the "
            "spring itself holds it there. A straight push springs back the moment you let go "
            "— you must press the knob the FULL depth of the press leg, sweep it along the "
            "cross leg to the far end, and release so the spring pulls it into the notch "
            "behind the shelf. The knob must stay in the slot: lifting it out and dropping it "
            "in from above voids the interlock."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the knob has rested in the locking notch — held against the live return
        spring by the notch shelf — for `seat_hold_steps` consecutive substeps, and it never
        left the slot. Physically latched: the state persists with no actor present."""
        return (self._hold >= self.cfg.seat_hold_steps) & ~self._escaped

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for nothing; 0..0.25 with the deepest press-leg excursion;
        0.25..0.55 with the farthest cross-leg sweep (geometrically requires full depth);
        0.70 latched once the knob entered the notch; 1.0 iff success. An `escaped` episode
        (knob lifted out of the slot) is capped at 0.30."""
        c = self.cfg
        s = torch.where(self._depth_max > c.frac_deadband,
                        0.25 * self._depth_max, torch.zeros_like(self._depth_max))
        cross = torch.where(self._cross_max > c.frac_deadband,
                            0.25 + 0.30 * self._cross_max, torch.zeros_like(s))
        s = torch.maximum(s, cross)
        s = torch.where(self._notched, torch.maximum(s, torch.full_like(s, 0.70)), s)
        s = torch.where(self.success(), torch.ones_like(s), s)
        return torch.where(self._escaped, s.clamp(max=0.30), s)


register_env("sim_gen", lambda: EnvCfg(scene="bayonet_switch", robot="null", env_spacing=3))
