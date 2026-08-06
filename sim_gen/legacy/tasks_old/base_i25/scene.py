"""ToppleOrderScene — controlled demolition: fell three pillars IN ORDER, each into its
own outward sector, without touching the fragile vase. Derived from pick_place/base but
STRATEGICALLY INVERTED: the seed's whole plan is prehensile transport — grasp one cube,
carry it along a prescribed 5-waypoint free-space trajectory, hold it at the goal. Here
NOTHING is transported and nothing ends up somewhere else: every judged object must stay
on its own footprint and merely change from standing to LYING, by being pushed past its
tipping point so gravity finishes the work. The plan is aimed instability (choose the
right pillar, push it over in the right direction, in the right sequence, without
collateral damage), not trajectory tracking; the seed's carry plan — pick a pillar up and
lay it where it should end — is expressible and is rejected by a dedicated latch.

The arena: three colored pillars (red "first", amber "second", blue "third" — tall
50 x 50 x 260 mm boxes, tipping angle atan(25/130) ~= 10.9 deg) stand on an arc around a
slender porcelain vase that sits INSIDE the arc, within falling reach of one arc slot
(the guarded slot, sampled per episode). Goal, three ordered stages:

  stage 1  topple RED   outward into its sector (within `sector_half_deg` of its own
           outward radial), landing on its footprint (centre within `fall_radius`);
  stage 2  topple AMBER the same way — but only AFTER red is down (order latch);
  stage 3  topple BLUE  the same way;
  always   the vase must remain standing — knocking it over (tilt or displacement) is a
           PERMANENT loss latch that zeroes the score forever.

Honesty devices (all latched in post_step from live physics state):
  - order latch: a pillar's down-crossing (axis passing `down_deg` from vertical) earns
    `felled_ok` only when it is the next expected pillar AND its centre is still within
    `home_radius` of its pedestal at the crossing — a genuine in-place fall;
  - anti-carry latch (`uprooted`): a pillar translated more than `uproot_move` (or lifted
    more than `uproot_lift`) while still upright can never earn credit — this is exactly
    the seed's pick-and-carry plan, and it is smoke-tested as a failing negative control;
  - loss latch (`vase_broken`): vase tilted past `vase_tilt_deg` or displaced more than
    `vase_move` — terminal, score pinned to 0.

Rubric: 0.30 per pillar that is felled_ok AND currently lying in its sector on its
footprint, settled (0.30 / 0.60 / 0.90-equivalent progression, latching via felled_ok);
1.0 iff success (all three credited + vase standing); 0 for doing nothing; 0 forever
after breaking the vase. Success judges PHYSICAL final state: real lying poses from real
falls (the oracle only tips a pillar a little past its critical angle — gravity, not the
teleport, produces the outcome that is judged).

Per-episode randomization: the color->slot assignment is permuted (a memorized motion
sequence fails), slot azimuths/radii jitter, every pillar gets a free yaw, and the
guarded slot (which pillar the vase sits behind) is sampled.

All assets are procedural primitives (dynamic cuboid pillars, dynamic cylinder vase).
Heavy imports (isaaclab) are deferred so importing this module stays app-free.
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
class ToppleOrderSceneCfg(BaseCfg):
    """Config for `ToppleOrderScene`. Geometry invariants (vase within toppling reach of
    the guarded slot; outward sectors clear of the vase and of neighbor slots) are
    asserted in __post_init__ — honesty by construction."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    down_deg: float = tunable(60.0)  # tilt beyond this = the down-crossing (order latch fires)
    lying_axis_z: float = tunable(0.35)  # |axis . z| below this = lying flat
    sector_half_deg: float = tunable(45.0)  # fall heading within this of the outward radial
    fall_radius: float = tunable(0.30)  # lying centre within this of the pedestal (m)
    home_radius: float = tunable(0.22)  # centre must be this close to home AT the crossing (m)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: anti-carry latch (rejects the seed's transport plan) -----------------------
    uproot_move: float = tunable(0.10)  # xy drift while upright that voids the pillar (m)
    uproot_lift: float = tunable(0.05)  # centre-height gain while upright that voids it (m)
    uproot_tilt_deg: float = tunable(35.0)  # "still upright" bound for the two clauses above
    # (an in-place fall pivots about the base edge: centre xy drift at 35 deg tilt is
    #  (h/2)*sin(35) ~= 75 mm < uproot_move, so honest falls never trip the latch)

    # --- tunable: vase loss latch ------------------------------------------------------------
    vase_tilt_deg: float = tunable(30.0)  # vase axis past this from vertical = broken
    vase_move: float = tunable(0.05)  # vase xy drift past this = broken (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    az_jitter_deg: float = tunable(8.0)  # uniform +/- azimuth jitter per slot
    radial_jitter: float = tunable(0.02)  # uniform +/- slot radius jitter (m)
    yaw_deg: float = tunable(180.0)  # uniform +/- free yaw per pillar at reset
    vase_jitter: float = tunable(0.015)  # uniform +/- xy jitter of the vase (m)

    # --- tunable: placement ------------------------------------------------------------------
    slot_radius: float = tunable(0.35)  # arc radius of the pillar slots (m)
    slot_azimuths: tuple = tunable((-55.0, 0.0, 55.0))  # arc slots (deg, arena frame)
    vase_radius_pos: float = tunable(0.15)  # vase distance from arena centre (m)

    # --- info: bodies ------------------------------------------------------------------------
    pillar_size: tuple = info((0.05, 0.05, 0.26))  # (x, y, height) — tip angle ~10.9 deg
    pillar_mass: float = info(0.15)
    # (name, rgb) in REQUIRED TOPPLE ORDER: red first, amber second, blue third.
    pillars: tuple = info((
        ("first", (0.85, 0.15, 0.12)),
        ("second", (0.92, 0.72, 0.10)),
        ("third", (0.15, 0.35, 0.85)),
    ))
    vase_r: float = info(0.022)
    vase_h: float = info(0.16)
    vase_mass: float = info(0.05)
    vase_color: tuple = info((0.96, 0.96, 0.92))
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    n_pillars: int = field(default=None, init=False)
    tip_angle_deg: float = field(default=None, init=False)  # flat-edge critical tilt

    def __post_init__(self) -> None:
        self.n_pillars = len(self.pillars)
        w, _d, h = self.pillar_size[0], self.pillar_size[1], self.pillar_size[2]
        self.tip_angle_deg = round(math.degrees(math.atan((w / 2) / (h / 2))), 2)
        # Vase within toppling reach of the guarded slot in the WORST jitter case: the
        # pillar (length h) must span the largest possible pillar-vase gap with margin.
        max_gap = (self.slot_radius + self.radial_jitter) - (self.vase_radius_pos
                                                             - self.vase_jitter)
        assert max_gap < h - 0.02, "vase can drift out of toppling reach"
        # ... and outside the OUTWARD sectors: an outward fall points away from the centre,
        # so anything strictly inside the arc is safe; just require the vase truly inside.
        assert self.vase_radius_pos + self.vase_jitter < self.slot_radius - self.radial_jitter
        # Neighbor slots must be clear of each other's outward falls: worst-case heading
        # (sector edge toward the neighbor) still misses the neighbor's base.
        span = math.radians(min(abs(b - a) for a, b in zip(self.slot_azimuths,
                                                           self.slot_azimuths[1:])))
        chord = 2 * (self.slot_radius - self.radial_jitter) * math.sin(span / 2)
        assert chord > w + 0.02, "neighbor slots closer than a pillar width"
        # The anti-carry latch must not fire on an honest pivot fall (see tunable note).
        assert (h / 2) * math.sin(math.radians(self.uproot_tilt_deg)) < self.uproot_move - 0.02


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("topple_order")
class ToppleOrderScene(BaseScene):
    cfg: ToppleOrderSceneCfg

    def __init__(self, cfg: ToppleOrderSceneCfg | None = None) -> None:
        super().__init__(cfg or ToppleOrderSceneCfg())

    # ----- assets ------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, three dynamic pillar boxes at their nominal slots, and the vase
        (reset() re-places everything with the episode's sampled layout)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        w, d, h = c.pillar_size

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

        for i, (name, rgb) in enumerate(c.pillars):
            az = math.radians(c.slot_azimuths[i])
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pillar_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(w, d, h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.05, angular_damping=0.20,
                        max_depenetration_velocity=0.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pillar_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_radius * math.cos(az), c.slot_radius * math.sin(az),
                         h / 2 + 0.003)),
            )

        out["vase"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Vase",
            spawn=sim_utils.CylinderCfg(
                radius=c.vase_r, height=c.vase_h,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.30,
                    max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.vase_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.vase_color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.vase_radius_pos, 0.0, c.vase_h / 2 + 0.003)),
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

    # ----- lifecycle ---------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        n, p = env.num_envs, c.n_pillars
        dev = env.device
        self.pillars: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _rgb in c.pillars}
        self.vase: RigidObject = env.iscene["vase"]
        self.env_origins = env.iscene.env_origins
        # Episode layout (env-local): pedestal xy and outward target azimuth per pillar.
        self.home_xy = torch.zeros(n, p, 2, device=dev)
        self.target_az = torch.zeros(n, p, device=dev)
        self.guard_pillar = torch.zeros(n, dtype=torch.long, device=dev)  # pillar at guarded slot
        self.vase_home = torch.zeros(n, 2, device=dev)
        # Latches (post_step).
        self.felled_ok = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.violated = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.uprooted = torch.zeros(n, p, dtype=torch.bool, device=dev)
        self.vase_broken = torch.zeros(n, dtype=torch.bool, device=dev)
        self.next_expected = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_down = torch.zeros(n, p, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: permute the color->slot assignment, jitter slot azimuth + radius,
        stand every pillar upright with free yaw, place the vase behind a sampled guarded
        slot, clear all latches, record the layout (homes + outward target azimuths)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        w, _d, h = c.pillar_size

        self.felled_ok[env_ids] = False
        self.violated[env_ids] = False
        self.uprooted[env_ids] = False
        self.vase_broken[env_ids] = False
        self.next_expected[env_ids] = 0
        self._prev_down[env_ids] = False

        # perm[:, i] = slot index of pillar i (color->slot permutation).
        perm = torch.rand(m, c.n_pillars, device=dev).argsort(dim=1)
        base_az = torch.tensor([math.radians(a) for a in c.slot_azimuths], device=dev)
        az_jit = (torch.rand(m, c.n_pillars, device=dev) * 2 - 1) * math.radians(c.az_jitter_deg)
        r_jit = (torch.rand(m, c.n_pillars, device=dev) * 2 - 1) * c.radial_jitter
        az = base_az[perm] + az_jit  # (m, p): actual azimuth of pillar i
        rad = c.slot_radius + r_jit

        self.home_xy[env_ids, :, 0] = rad * torch.cos(az)
        self.home_xy[env_ids, :, 1] = rad * torch.sin(az)
        self.target_az[env_ids] = az  # outward radial = the required fall heading

        yaw = (torch.rand(m, c.n_pillars, device=dev) * 2 - 1) * math.radians(c.yaw_deg) / 2
        for i, (name, _rgb) in enumerate(c.pillars):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = self.home_xy[env_ids, i]
            st[:, 2] = h / 2 + 0.003
            st[:, 3] = torch.cos(yaw[:, i])
            st[:, 6] = torch.sin(yaw[:, i])
            st[:, 0:3] += origin
            self.pillars[name].write_root_state_to_sim(st, env_ids)

        # Vase: inside the arc, behind a sampled guarded slot (within toppling reach).
        gs = torch.randint(0, c.n_pillars, (m,), device=dev)
        inv = perm.argsort(dim=1)  # inv[:, s] = pillar index occupying slot s
        self.guard_pillar[env_ids] = inv.gather(1, gs.unsqueeze(1)).squeeze(1)
        g_az = az.gather(1, self.guard_pillar[env_ids].unsqueeze(1)).squeeze(1)
        vx = c.vase_radius_pos * torch.cos(g_az)
        vy = c.vase_radius_pos * torch.sin(g_az)
        vst = torch.zeros(m, 13, device=dev)
        vst[:, 0] = vx
        vst[:, 1] = vy
        vst[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.vase_jitter
        vst[:, 2] = c.vase_h / 2 + 0.003
        vst[:, 3] = 1.0
        self.vase_home[env_ids] = vst[:, :2]
        vst[:, 0:3] += origin
        self.vase.write_root_state_to_sim(vst, env_ids)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch machinery, evaluated on live physics state every step: the anti-carry
        latch, the ordered down-crossing latch, and the vase loss latch."""
        c = self.cfg
        h = c.pillar_size[2]
        pos, axis_z, _vel = self._pillar_tensors()
        pos_loc = pos - self.env_origins[:, None, :]
        disp = (pos_loc[:, :, :2] - self.home_xy).norm(dim=-1)

        upright = axis_z > math.cos(math.radians(c.uproot_tilt_deg))
        self.uprooted |= upright & ((disp > c.uproot_move)
                                    | (pos_loc[:, :, 2] > h / 2 + c.uproot_lift))

        down = axis_z < math.cos(math.radians(c.down_deg))
        crossing = down & ~self._prev_down
        # Sequential over pillar index so a same-step double crossing latches in order.
        for i in range(c.n_pillars):
            ok = (crossing[:, i] & (self.next_expected == i)
                  & (disp[:, i] < c.home_radius) & ~self.uprooted[:, i])
            self.felled_ok[:, i] |= ok
            self.violated[:, i] |= crossing[:, i] & ~ok
            self.next_expected += ok.long()
        self._prev_down = down

        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        v_axis = quat_apply(self.vase.data.root_quat_w, ez)
        v_loc = self.vase.data.root_pos_w - self.env_origins
        v_disp = (v_loc[:, :2] - self.vase_home).norm(dim=-1)
        self.vase_broken |= ((v_axis[:, 2] < math.cos(math.radians(c.vase_tilt_deg)))
                             | (v_disp > c.vase_move))

    # ----- state (full, restorable) ------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pillars": {n: b.data.root_state_w[env_ids].clone()
                        for n, b in self.pillars.items()},
            "vase": self.vase.data.root_state_w[env_ids].clone(),
            "home_xy": self.home_xy[env_ids].clone(),
            "target_az": self.target_az[env_ids].clone(),
            "guard_pillar": self.guard_pillar[env_ids].clone(),
            "vase_home": self.vase_home[env_ids].clone(),
            "felled_ok": self.felled_ok[env_ids].clone(),
            "violated": self.violated[env_ids].clone(),
            "uprooted": self.uprooted[env_ids].clone(),
            "vase_broken": self.vase_broken[env_ids].clone(),
            "next_expected": self.next_expected[env_ids].clone(),
            "prev_down": self._prev_down[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.pillars.items():
            b.write_root_state_to_sim(state["pillars"][n], env_ids)
        self.vase.write_root_state_to_sim(state["vase"], env_ids)
        self.home_xy[env_ids] = state["home_xy"]
        self.target_az[env_ids] = state["target_az"]
        self.guard_pillar[env_ids] = state["guard_pillar"]
        self.vase_home[env_ids] = state["vase_home"]
        self.felled_ok[env_ids] = state["felled_ok"]
        self.violated[env_ids] = state["violated"]
        self.uprooted[env_ids] = state["uprooted"]
        self.vase_broken[env_ids] = state["vase_broken"]
        self.next_expected[env_ids] = state["next_expected"]
        self._prev_down[env_ids] = state["prev_down"]

    # ----- description -------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        w, _d, h = c.pillar_size
        return (
            f"A demolition range: three tall colored pillars ({w * 1000:.0f} mm square, "
            f"{h * 1000:.0f} mm high — red, amber, blue) stand upright on an arc, and a "
            f"slender porcelain vase stands on the floor just inside the arc, within falling "
            f"reach of one of them.\n"
            f"Goal: knock the pillars over IN ORDER — red first, then amber, then blue — "
            f"each toppling OUTWARD, away from the arena centre (its fall heading within "
            f"{c.sector_half_deg:.0f} deg of its own outward radial), and each falling in "
            f"place on its own base spot. Push a pillar past its tipping point and let "
            f"gravity fell it: a pillar that is picked up, dragged, or laid down somewhere "
            f"else earns nothing, and a pillar felled out of turn earns nothing. The vase "
            f"must remain standing untouched — knocking it over ends the job with a zero "
            f"score, permanently."
        )

    # ----- predicates / rubric -----------------------------------------------------------------
    def _pillar_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_w (N,P,3), axis_z (N,P) = pillar-up . world-up, |lin_vel| (N,P))."""
        from isaaclab.utils.math import quat_apply

        pos = torch.stack([b.data.root_pos_w for b in self.pillars.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.pillars.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.pillars.values()], dim=1)
        n, p = pos.shape[0], pos.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * p, 3)
        axis = quat_apply(quat.reshape(n * p, 4), ez).reshape(n, p, 3)
        self._axis_cache = axis
        return pos, axis[:, :, 2], vel

    def lying(self) -> torch.Tensor:
        """(N,P) bool: pillar flat on the ground (|axis . z| < lying_axis_z)."""
        _pos, axis_z, _vel = self._pillar_tensors()
        return axis_z.abs() < self.cfg.lying_axis_z

    def in_sector(self) -> torch.Tensor:
        """(N,P) bool: the lying pillar's heading (base->top azimuth) within
        `sector_half_deg` of its outward target azimuth."""
        self._pillar_tensors()
        axis = self._axis_cache
        az = torch.atan2(axis[:, :, 1], axis[:, :, 0])
        delta = (az - self.target_az + math.pi) % (2 * math.pi) - math.pi
        return delta.abs() < math.radians(self.cfg.sector_half_deg)

    def near_home(self) -> torch.Tensor:
        """(N,P) bool: pillar centre within `fall_radius` of its pedestal."""
        pos, _axis_z, _vel = self._pillar_tensors()
        pos_loc = pos - self.env_origins[:, None, :]
        return (pos_loc[:, :, :2] - self.home_xy).norm(dim=-1) < self.cfg.fall_radius

    def settled(self) -> torch.Tensor:
        """(N,P) bool: pillar |lin vel| below `settle_speed`."""
        return self._pillar_tensors()[2] < self.cfg.settle_speed

    def vase_standing(self) -> torch.Tensor:
        """(N,) bool: the vase currently upright and unbroken (latch + live pose)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        v_axis = quat_apply(self.vase.data.root_quat_w, ez)
        return (~self.vase_broken
                & (v_axis[:, 2] >= math.cos(math.radians(self.cfg.vase_tilt_deg))))

    def credit(self) -> torch.Tensor:
        """(N,P) bool: what the rubric counts — cleanly felled in order (latch) AND
        currently lying in its sector on its footprint, settled."""
        return (self.felled_ok & self.lying() & self.in_sector()
                & self.near_home() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30 per credited pillar; 1.0 iff success; pinned to 0
        forever once the vase is broken. Doing nothing scores 0; the seed's carry plan
        (uprooted latch) and out-of-order falls (order latch) earn nothing."""
        s = 0.30 * self.credit().float().sum(dim=1)
        s = torch.where(self.success(), torch.ones_like(s), s)
        return torch.where(self.vase_broken, torch.zeros_like(s), s)

    def success(self) -> torch.Tensor:
        """(N,) bool: all three pillars credited + the vase still standing."""
        return self.credit().all(dim=1) & self.vase_standing()


# Scene-level env binding (robot embodiments are a later stage).
register_env("simgen", lambda: EnvCfg(scene="topple_order", robot="null", env_spacing=3))
