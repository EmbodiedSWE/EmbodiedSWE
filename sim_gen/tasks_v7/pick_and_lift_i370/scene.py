"""RollerRelayScene — move an over-heavy freight slab between two loading docks by
laying loose ROLLERS under it, ancient-Egypt style (sim_gen task `pick_and_lift_i370`).

Derived from rlbench/pick_and_lift, but STRATEGICALLY different: the seed grasps a
5 cm cube that is chosen to fit the jaw and carries it up to a marked height — the
lift IS the grasp. Here the judged object is a 7 kg orange SLAB (260 x 160 x 50 mm)
that no Franka can grasp, lift, or drag: its weight is ~69 N (far past the arm's
payload), and the transit corridor between the two docks is floored with high-grip
RUBBER (mu ~1.2, ~82 N to drag). The docks themselves are SLICK (mu ~0.05, ~3.4 N
to slide), so the slab glides freely on a dock but is immovable the moment it sits
on rubber. The only way across is to build rolling infrastructure first: three loose
blue steel ROLLERS (36 mm diameter — these DO fit the jaw) start in a staging area
beside the track and must be laid across the rubber channel between two guide curbs.
The channel depth is tuned so a roller's crown is exactly FLUSH with the start-dock
top: the slab slides straight off the dock onto the rollers and can then be pushed
along, riding at walking friction while the rollers revolve beneath it (a rolling
relay — rollers emerge at the rear and must be carried round to the front). At the
far end a shallow PIT in front of the goal dock swallows spent rollers below ride
height, and the slab steps 2 mm DOWN onto the slick goal dock, where light nudges
park it against the end bumper. A solver therefore needs a different PLAN (build
the rolling path, then push; never lift the judged object) and different CODE
(support-span reasoning, tool staging, ride-height-gated progress) — not a
grasp-and-raise checker. Pushing the slab off the dock BEFORE laying rollers is an
irreversible dead-end: it drops onto the rubber and can never be dragged out.

success() iff, settled: all four bottom corners of the slab are over the GREEN goal
dock, the slab is flat (within `flat_deg` of level) with its underside at dock-top
height (rejects "still riding on rollers" ~36 mm high and "dropped in the channel"
~36 mm low), regardless of where the rollers ended up. score() latches every physics
substep: 0.10 * (rollers ever staged in the channel / 3) + 0.15 for the slab ever
riding aboard the rollers + 0.55 * farthest ride-height advance toward the goal dock
(gated on ride height and flatness, so dragging through the channel or carrying the
slab farms nothing), capped at 0.80; exactly 1.0 iff success(). Doing nothing ~0.

FRICTION SCHEME (PhysX combine-mode precedence: average < min < multiply < max —
the higher-precedence mode of the pair wins): docks + bumper declare combine MIN
with mu 0.06/0.05 (any contact with a dock is slick), the rubber mat + curbs + pit
floor declare combine MAX with mu 1.2/1.1 (any contact with rubber grips), slab and
rollers declare AVERAGE with mu 0.9/0.8 (slab-on-roller ~0.85 traction). Asserted
consequences in `__post_init__`: sliding the slab on a dock needs ~4 N, dragging it
on rubber needs ~82 N (> any honest push), its dead lift is ~69 N.

Assets are fully procedural (no external files): two kinematic dock slabs, rubber
mat + pit floor + two guide curbs + end bumper (all static colliders), a dynamic
slab, three dynamic rollers. Per-episode randomization (verified by readback in
smoke): slab xy + yaw on the start dock, WHICH SIDE of the track the staging area
is on (left/right coin flip), per-roller staging slot jitter + yaw. Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- spawn-cfg helpers -----------------------------------------------------------------------
def _mat(kind: str) -> Any:
    """Physics material by role. Combine-mode precedence (average < min < max) is the
    load-bearing trick: dock MIN beats slab AVERAGE (slick), rubber MAX beats slab
    AVERAGE (grip), slab-roller AVERAGE-AVERAGE averages (traction)."""
    import isaaclab.sim as sim_utils

    if kind == "slick":
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=0.06, dynamic_friction=0.05, restitution=0.0,
            friction_combine_mode="min", restitution_combine_mode="min")
    if kind == "rubber":
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2, dynamic_friction=1.1, restitution=0.0,
            friction_combine_mode="max", restitution_combine_mode="min")
    # "body": the slab and the rollers
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=0.9, dynamic_friction=0.8, restitution=0.0,
        friction_combine_mode="average", restitution_combine_mode="min")


def _static_box(size: tuple, color: tuple, kind: str, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=contact_offset, rest_offset=0.0),
        physics_material=_mat(kind),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


def _rigid_props() -> Any:
    import isaaclab.sim as sim_utils

    # velocity_iteration_count=4: kills the GPU cylinder-on-box phantom creep band.
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16, solver_velocity_iteration_count=4,
        max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.10,
        disable_gravity=False)


def _slab_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=(c.slab_l, c.slab_w, c.slab_t),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0),
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.slab_m),
        physics_material=_mat("body"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.slab_color),
    )


def _roller_cfg(c: Any) -> Any:
    import isaaclab.sim as sim_utils

    return sim_utils.CylinderCfg(
        radius=c.roller_r, height=c.roller_len, axis="Y",
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0),
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=c.roller_m),
        physics_material=_mat("body"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.roller_color),
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class RollerRelaySceneCfg(BaseCfg):
    """Config for `RollerRelayScene`. Honesty knobs asserted in `__post_init__`:
    flush boarding (roller crown on the mat == start-dock top), 2 mm step DOWN onto
    the goal dock (no lip to snag), the pit swallows rollers below both ride height
    and the goal-dock top, the slab is un-liftable / un-draggable-on-rubber /
    easily-slidable-on-dock by force arithmetic, rollers fit the Franka jaw, the
    channel is too long for the slab to bridge, and a slab parked against the end
    bumper always satisfies the goal rectangle."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    flat_deg: float = tunable(6.0)  # max tilt from level for "flat"
    z_tol: float = tunable(0.008)  # +/- band around goal-dock top for the slab underside
    rect_margin: float = tunable(0.005)  # corners must be this far inside the goal-dock ends
    settle_lin: float = tunable(0.08)  # max slab |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.40)  # max slab |ang vel| when judging (rad/s)
    roller_slow: float = tunable(0.10)  # a roller must be slower than this to latch "staged"
    roller_axis_deg: float = tunable(25.0)  # staged roller axis within this of the y axis

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    slab_jx: float = tunable(0.020)  # uniform +/- slab x jitter on the start dock (m)
    slab_jy: float = tunable(0.020)  # uniform +/- slab y jitter (m)
    slab_yaw_deg: float = tunable(4.0)  # uniform +/- slab yaw
    stage_jx: float = tunable(0.030)  # uniform +/- roller slot x jitter (m)
    stage_jy: float = tunable(0.050)  # uniform +/- staging-band y jitter (m)
    roller_yaw_deg: float = tunable(30.0)  # uniform +/- roller yaw in the staging area

    # --- info: track geometry (env-local; +x = travel direction) -----------------------------
    dock_l: float = info(0.33)  # each dock along x; start spans [-0.33, 0]
    dock_w: float = info(0.210)  # track width (docks, mat, bumper)
    start_top: float = info(0.042)  # start-dock top == RIDE PLANE (roller crown on mat)
    goal_top: float = info(0.040)  # goal-dock top: 2 mm BELOW ride plane (step down)
    goal_x0: float = info(0.46)  # goal dock spans [0.46, 0.79]
    mat_len: float = info(0.40)  # rubber mat spans [0, 0.40]
    mat_t: float = info(0.006)
    pit_len: float = info(0.06)  # roller pit spans [0.40, 0.46] in front of the goal dock
    pit_t: float = info(0.002)  # thin rubber pit floor (keeps the channel undraggable)
    curb_len: float = info(0.46)  # curbs run the whole channel
    curb_w: float = info(0.018)
    curb_h: float = info(0.026)  # curb top 26 mm, well under the 42 mm ride plane
    curb_y: float = info(0.084)  # curb centres: inner faces at +/- 0.075
    bumper_t: float = info(0.030)  # end bumper spans x [0.79, 0.82]
    bumper_h: float = info(0.100)
    # payload + tools
    slab_l: float = info(0.260)
    slab_w: float = info(0.160)
    slab_t: float = info(0.050)
    slab_m: float = info(7.0)  # ~69 N dead weight: no Franka lifts this
    roller_r: float = info(0.018)  # 36 mm diameter: fits the 80 mm jaw
    roller_len: float = info(0.140)  # fits the 150 mm curb gap with 10 mm slack
    roller_m: float = info(0.18)
    n_rollers: int = info(3)
    # nominal placements
    slab_x0: float = info(-0.175)  # slab centre spawn on the start dock
    stage_x: tuple = info((0.08, 0.23, 0.38))  # staging slots along x, beside the track
    stage_y: float = info(0.27)  # staging band |y| (side flipped per episode)
    # colors
    start_color: tuple = info((0.45, 0.45, 0.50))
    goal_color: tuple = info((0.10, 0.60, 0.20))
    mat_color: tuple = info((0.30, 0.09, 0.07))
    curb_color: tuple = info((0.20, 0.06, 0.05))
    bumper_color: tuple = info((0.16, 0.16, 0.18))
    slab_color: tuple = info((0.92, 0.45, 0.05))
    roller_color: tuple = info((0.15, 0.35, 0.85))
    # friction (mirrors _mat(); asserted below)
    mu_dock: float = info(0.06)
    mu_rubber: float = info(1.2)
    mu_body: float = info(0.9)
    contact_offset: float = info(0.0015)

    # Derived (filled in __post_init__).
    ride_z: float = field(default=None, init=False)  # slab underside height when riding
    goal_x1: float = field(default=None, init=False)  # goal dock far edge (bumper face)
    goal_cx: float = field(default=None, init=False)  # goal dock centre x
    curb_gap: float = field(default=None, init=False)  # clear width between curb inner faces
    cos_flat: float = field(default=None, init=False)
    cos_axis: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.ride_z = self.start_top
        self.goal_x1 = self.goal_x0 + self.dock_l
        self.goal_cx = self.goal_x0 + self.dock_l / 2
        self.curb_gap = 2 * self.curb_y - self.curb_w
        self.cos_flat = math.cos(math.radians(self.flat_deg))
        self.cos_axis = math.cos(math.radians(self.roller_axis_deg))

        # ---- flush boarding + step-down landing ----
        crown = self.mat_t + 2 * self.roller_r  # roller crown height on the mat
        assert abs(self.start_top - crown) < 1e-9, "roller crown must be FLUSH with start dock"
        step = self.ride_z - self.goal_top
        assert 0.001 <= step <= 0.004, "goal dock must sit 1-4 mm BELOW the ride plane"

        # ---- the pit swallows spent rollers ----
        pit_crown = self.pit_t + 2 * self.roller_r
        assert pit_crown <= self.goal_top - 0.001, "pit roller crown must duck the goal dock"
        assert pit_crown <= self.ride_z - 0.003, "pit roller crown must duck the ride plane"
        assert self.pit_len >= 2 * self.roller_r + 0.020, "a spent roller fits the pit floor"
        # Support-span timeline (rollers translate at HALF slab speed): the lead roller,
        # laid at ~0.215, must still be ON the mat (< mat_len) when the slab's front
        # first reaches the goal dock — after that the dock carries the front and pit
        # entry is unloaded. Front reaches the dock at slab centre goal_x0 - slab_l/2;
        # the lead roller mounted at centre ~(0.215 - slab_l/2) has then travelled half
        # the slab's travel from there.
        c0 = 0.215 - self.slab_l / 2  # slab centre when the lead roller mounts
        cf = self.goal_x0 - self.slab_l / 2  # slab centre when its front reaches the dock
        lead_at_cf = 0.215 + (cf - c0) / 2
        assert lead_at_cf < self.mat_len - 0.01, \
            "lead roller must still be on the mat when the slab front reaches the dock"
        assert lead_at_cf > cf, \
            "lead roller must still be AHEAD of the slab CoM when the front reaches the dock"
        assert abs(self.mat_len + self.pit_len - self.goal_x0) < 1e-9, \
            "channel tiles [0, goal_x0] exactly"

        # ---- rollers between the curbs; slab spans over them ----
        assert self.roller_len + 0.004 <= self.curb_gap, "roller drops between the curbs"
        assert self.curb_gap <= self.slab_w - 0.006, "slab always spans the curb gap"
        assert self.curb_h + 0.008 <= self.ride_z, "riding slab clears the curb tops"

        # ---- force arithmetic (the strategic core) ----
        w = self.slab_m * 9.81
        assert w >= 45.0, f"slab dead lift {w:.0f} N must exceed any honest lift"
        assert self.mu_rubber * w >= 55.0, "dragging the slab on rubber must be hopeless"
        assert self.mu_dock * w <= 6.0, "sliding the slab on a dock must be easy (~4 N)"
        assert 2 * self.roller_r <= 0.075, "rollers must fit the Franka jaw"

        # ---- reachability + no-bridging + bumper-park honesty ----
        assert self.dock_l >= self.slab_l + 0.06, "slab fits on each dock with slack"
        assert self.goal_x0 >= self.slab_l + 0.12, "slab can NEVER bridge the channel"
        assert self.mat_len >= self.slab_l + 0.06, "the mat can carry the whole slab"
        assert self.goal_x1 - self.slab_l >= self.goal_x0 + self.rect_margin + 0.005, \
            "a slab parked against the bumper satisfies the goal rectangle"
        assert self.slab_x0 - self.slab_jx - self.slab_l / 2 >= -self.dock_l + 0.005, \
            "slab spawn stays on the start dock (rear)"
        assert self.slab_x0 + self.slab_jx + self.slab_l / 2 <= -0.015, \
            "slab spawn stays clear of the dock lip (front)"
        assert self.stage_y - self.stage_jy - self.roller_len / 2 >= self.dock_w / 2 + 0.02, \
            "staging area stays clear of the track"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("roller_relay_freight")
class RollerRelayScene(BaseScene):
    cfg: RollerRelaySceneCfg

    def __init__(self, cfg: RollerRelaySceneCfg | None = None) -> None:
        super().__init__(cfg or RollerRelaySceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        co = c.contact_offset
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
            "start_dock": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/StartDock",
                spawn=_static_box((c.dock_l, c.dock_w, c.start_top), c.start_color,
                                  "slick", co),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-c.dock_l / 2, 0.0, c.start_top / 2)),
            ),
            "goal_dock": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/GoalDock",
                spawn=_static_box((c.dock_l, c.dock_w, c.goal_top), c.goal_color,
                                  "slick", co),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.goal_cx, 0.0, c.goal_top / 2)),
            ),
            "mat": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Mat",
                spawn=_static_box((c.mat_len, c.dock_w, c.mat_t), c.mat_color,
                                  "rubber", co),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.mat_len / 2, 0.0, c.mat_t / 2)),
            ),
            "pit_floor": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/PitFloor",
                spawn=_static_box((c.pit_len, c.dock_w, c.pit_t), c.mat_color,
                                  "rubber", co),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.mat_len + c.pit_len / 2, 0.0, c.pit_t / 2)),
            ),
            "bumper": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bumper",
                spawn=_static_box((c.bumper_t, c.dock_w, c.bumper_h), c.bumper_color,
                                  "slick", co),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.goal_x1 + c.bumper_t / 2, 0.0, c.bumper_h / 2)),
            ),
            "slab": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slab",
                spawn=_slab_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slab_x0, 0.0, c.start_top + c.slab_t / 2 + 0.002)),
            ),
        }
        for tag, sgn in (("curb_yp", 1.0), ("curb_yn", -1.0)):
            out[tag] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + tag.title().replace("_", ""),
                spawn=_static_box((c.curb_len, c.curb_w, c.curb_h), c.curb_color,
                                  "rubber", co),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.curb_len / 2, sgn * c.curb_y, c.curb_h / 2)),
            )
        for i in range(c.n_rollers):
            out[f"roller_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Roller_" + str(i),
                spawn=_roller_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stage_x[i], c.stage_y, c.roller_r + 0.0015)),
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
        c = self.cfg
        self.slab: RigidObject = env.iscene["slab"]
        self.rollers: list[RigidObject] = [env.iscene[f"roller_{i}"] for i in range(c.n_rollers)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.x0 = torch.full((n,), c.slab_x0, device=dev)  # slab spawn x (advance baseline)
        self.side = torch.ones(n, device=dev)  # staging side (+1 / -1), for readback
        self.staged_latch = torch.zeros(n, c.n_rollers, device=dev)
        self.aboard_latch = torch.zeros(n, device=dev)
        self.advance_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: slab on the start dock with xy + yaw jitter, the three
        rollers dealt into jittered slots in a staging band whose SIDE of the track
        is a per-episode coin flip, each with free yaw about vertical; latches zeroed
        and the advance baseline captured. (A few RNG draws are burnt first — the
        first post-seed draw is degenerate across seeds on this stack.)"""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, 3, device=dev)  # burn: first post-seed draws are degenerate
        side = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0).to(dev)
        self.side[env_ids] = side

        # --- slab on the start dock ---
        x = c.slab_x0 + (torch.rand(m, device=dev) * 2 - 1) * c.slab_jx
        y = (torch.rand(m, device=dev) * 2 - 1) * c.slab_jy
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.slab_yaw_deg) / 2
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = c.start_top + c.slab_t / 2 + 0.002
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.slab.write_root_state_to_sim(st, env_ids)
        self.x0[env_ids] = x

        # --- rollers in the staging band (side flipped per episode) ---
        for i, roller in enumerate(self.rollers):
            rx = c.stage_x[i] + (torch.rand(m, device=dev) * 2 - 1) * c.stage_jx
            ry = side * (c.stage_y + (torch.rand(m, device=dev) * 2 - 1) * c.stage_jy)
            rhalf = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.roller_yaw_deg) / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = rx
            st[:, 1] = ry
            st[:, 2] = c.roller_r + 0.0015
            st[:, 3] = torch.cos(rhalf)
            st[:, 6] = torch.sin(rhalf)
            st[:, 0:3] += origin
            roller.write_root_state_to_sim(st, env_ids)

        self.staged_latch[env_ids] = 0.0
        self.aboard_latch[env_ids] = 0.0
        self.advance_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "slab": self.slab.data.root_state_w[env_ids].clone(),
            "rollers": [r.data.root_state_w[env_ids].clone() for r in self.rollers],
            "x0": self.x0[env_ids].clone(),
            "side": self.side[env_ids].clone(),
            "staged_latch": self.staged_latch[env_ids].clone(),
            "aboard_latch": self.aboard_latch[env_ids].clone(),
            "advance_latch": self.advance_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.slab.write_root_state_to_sim(state["slab"], env_ids)
        for r, s in zip(self.rollers, state["rollers"]):
            r.write_root_state_to_sim(s, env_ids)
        self.x0[env_ids] = state["x0"]
        self.side[env_ids] = state["side"]
        self.staged_latch[env_ids] = state["staged_latch"]
        self.aboard_latch[env_ids] = state["aboard_latch"]
        self.advance_latch[env_ids] = state["advance_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A freight track runs along +x. On the near end is a slick GRAY loading "
            f"dock (top {c.start_top * 1000:.0f} mm high), and on it rests a heavy ORANGE "
            f"slab ({c.slab_l * 1000:.0f} x {c.slab_w * 1000:.0f} x {c.slab_t * 1000:.0f} "
            f"mm, {c.slab_m:.0f} kg). The dock surface is nearly frictionless, so the slab "
            f"slides there under a light push (~4 N) — but it weighs ~{c.slab_m * 9.81:.0f} "
            f"N, far too heavy to lift, and between the docks lies a sunken channel floored "
            f"with high-grip RUBBER (friction ~1.2): anything resting on the rubber needs "
            f"~{c.mu_rubber * c.slab_m * 9.81:.0f} N to drag, which is hopeless. The "
            f"channel is {c.goal_x0 * 1000:.0f} mm long — much longer than the slab, which "
            f"can never bridge it. Across the channel is the GREEN goal dock (top "
            f"{c.goal_top * 1000:.0f} mm), ending at a dark bumper wall. Two low rubber "
            f"curbs (tops {c.curb_h * 1000:.0f} mm) flank the channel, "
            f"{c.curb_gap * 1000:.0f} mm apart. Beside the track, on the open floor, lie "
            f"three loose BLUE steel rollers (cylinders, {2 * c.roller_r * 1000:.0f} mm "
            f"diameter x {c.roller_len * 1000:.0f} mm — light, and slim enough to grasp).\n"
            f"Goal: deliver the orange slab flat onto the green goal dock. The intended "
            f"method is a ROLLING RELAY: lay the rollers across the channel between the "
            f"curbs, axes across the track (the curb gap fits them with ~10 mm slack); a "
            f"roller resting on the rubber stands exactly as tall as the gray dock top, so "
            f"the slab slides straight off the dock onto the rollers and rides at "
            f"{c.ride_z * 1000:.0f} mm, clear of the rubber. Stage rollers under the whole "
            f"slab footprint before pushing it off the dock — WARNING: pushing the slab "
            f"off the dock with no rollers beneath drops it onto the rubber, an "
            f"IRREVERSIBLE dead end. While pushing, rollers emerge at the rear (they "
            f"travel at half the slab's speed); carry each spent roller round to the front "
            f"and re-lay it so the slab is always supported. A shallow pit in front of the "
            f"goal dock swallows spent rollers below ride height, and the goal dock top "
            f"sits 2 mm below the ride plane, so the slab steps smoothly DOWN onto the "
            f"slick green dock; light nudges then park it against the bumper. Judged only "
            f"when settled: all four corners of the slab over the green dock, slab flat "
            f"(within {c.flat_deg:.0f} deg) with its underside at dock-top height "
            f"(+/- {c.z_tol * 1000:.0f} mm — still riding on rollers, or dropped into the "
            f"channel, does not count). Where the rollers end up does not matter. Partial "
            f"credit: rollers staged in the channel, the slab riding aboard, and forward "
            f"progress made while riding at height; dragging or carrying earns nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Deliver the heavy orange slab onto the green goal dock: lay the three blue "
            "rollers across the rubber channel between the curbs, slide the slab off the "
            "gray dock onto them, and keep pushing while relaying spent rollers from the "
            "rear back to the front until the slab rides across and parks flat on the "
            "green dock. Never push the slab off a dock without rollers beneath it."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def slab_pos(self) -> torch.Tensor:
        """(N,3) env-local slab centre."""
        return self.slab.data.root_pos_w - self.env_origins

    def slab_axes(self) -> torch.Tensor:
        """(N,3,3) slab body axes as world columns [ex, ey, ez]."""
        from isaaclab.utils.math import matrix_from_quat

        return matrix_from_quat(self.slab.data.root_quat_w)

    def slab_flat(self) -> torch.Tensor:
        """(N,) bool: slab within flat_deg of level."""
        return self.slab_axes()[:, 2, 2] > self.cfg.cos_flat

    def slab_bottom_z(self) -> torch.Tensor:
        """(N,) env-local height of the slab underside (centre minus half thickness,
        valid while flat — used only under flatness gates)."""
        return self.slab_pos()[:, 2] - self.cfg.slab_t / 2

    def slab_corners_xy(self) -> torch.Tensor:
        """(N,4,2) env-local xy of the four bottom-face corners."""
        c = self.cfg
        rot = self.slab_axes()  # (N,3,3) world_from_body
        p = self.slab_pos()
        signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]],
                             device=p.device, dtype=p.dtype)
        local = torch.zeros(4, 3, device=p.device, dtype=p.dtype)
        local[:, 0] = signs[:, 0] * c.slab_l / 2
        local[:, 1] = signs[:, 1] * c.slab_w / 2
        world = torch.einsum("nij,kj->nki", rot, local) + p[:, None, :]
        return world[:, :, :2]

    def corners_on_goal(self) -> torch.Tensor:
        """(N,) bool: all four slab corners inside the goal-dock rectangle."""
        c = self.cfg
        xy = self.slab_corners_xy()
        ok_x = (xy[:, :, 0] > c.goal_x0 + c.rect_margin) \
            & (xy[:, :, 0] < c.goal_x1 + c.rect_margin)
        ok_y = xy[:, :, 1].abs() < c.dock_w / 2 - 0.002
        return (ok_x & ok_y).all(dim=1)

    def roller_pos(self) -> torch.Tensor:
        """(N,R,3) env-local roller centres."""
        return torch.stack([r.data.root_pos_w for r in self.rollers], dim=1) \
            - self.env_origins[:, None, :]

    def rollers_staged(self) -> torch.Tensor:
        """(N,R) bool: roller lying in the channel, axis across the track, slow."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos = self.roller_pos()
        ok = (pos[:, :, 0] > 0.005) & (pos[:, :, 0] < c.goal_x0 - 0.005) \
            & (pos[:, :, 1].abs() < 0.045) & (pos[:, :, 2] < 0.045)
        ey = torch.tensor([0.0, 1.0, 0.0], device=pos.device)
        for i, r in enumerate(self.rollers):
            axis = quat_apply(r.data.root_quat_w, ey.expand(self.env.num_envs, 3))
            ok[:, i] &= axis[:, 1].abs() > c.cos_axis
            ok[:, i] &= r.data.root_lin_vel_w.norm(dim=-1) < c.roller_slow
        return ok

    def aboard_now(self) -> torch.Tensor:
        """(N,) bool: slab flat, riding at roller height, centred over the channel."""
        c = self.cfg
        x = self.slab_pos()[:, 0]
        bz = self.slab_bottom_z()
        return self.slab_flat() & (bz > c.ride_z - 0.006) & (bz < c.ride_z + 0.014) \
            & (x > 0.01) & (x < c.goal_x0 - 0.01)

    def advance_frac_now(self) -> torch.Tensor:
        """(N,) in [0,1]: forward progress of the slab centre toward the goal-dock
        centre, GATED on flat + at ride height (or on the goal dock, 2 mm lower) +
        laterally on the track — dragging through the channel or hoisting the slab
        through the air farms nothing."""
        c = self.cfg
        p = self.slab_pos()
        bz = self.slab_bottom_z()
        gate = self.slab_flat() & (bz > c.goal_top - 0.006) & (bz < c.ride_z + 0.014) \
            & (p[:, 1].abs() < 0.12)
        frac = (p[:, 0] - self.x0 - 0.01) / (c.goal_cx - self.x0)
        return frac.clamp(0.0, 1.0) * gate.float()

    def settled(self) -> torch.Tensor:
        """(N,) bool: slab quiet."""
        c = self.cfg
        return (self.slab.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.slab.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch per-roller staging, ever-aboard, and best gated advance each physics
        substep, so honest transient progress keeps its credit."""
        self.staged_latch = torch.maximum(self.staged_latch, self.rollers_staged().float())
        self.aboard_latch = torch.maximum(self.aboard_latch, self.aboard_now().float())
        self.advance_latch = torch.maximum(self.advance_latch, self.advance_frac_now())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: slab settled FLAT on the goal dock — all four corners over the
        dock, underside at dock-top height (rejects riding-on-rollers ~36 mm high and
        dropped-in-channel ~36 mm low)."""
        c = self.cfg
        bz = self.slab_bottom_z()
        return self.corners_on_goal() & self.slab_flat() \
            & ((bz - c.goal_top).abs() < c.z_tol) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 * (rollers ever staged / 3) + 0.15 * ever aboard
        + 0.55 * latched gated advance, capped at 0.80; exactly 1.0 iff success().
        Doing nothing ~0; the seed's strategy (grasp and lift the judged object) is
        physically unavailable (7 kg, 69 N) and earns nothing."""
        c = self.cfg
        base = (0.10 * self.staged_latch.sum(dim=1) / c.n_rollers
                + 0.15 * self.aboard_latch
                + 0.55 * self.advance_latch).clamp(0.0, 0.80)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="roller_relay_freight", robot="null"))
