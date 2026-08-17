"""CorbelReachScene — build a freestanding corbelled shelf off a cliff edge until it
projects past a red beacon line (sim_gen task `approach_grasp_screwdriver_i428`).

Derived from pick_place/approach_grasp_screwdriver, but STRATEGICALLY different: the
seed is a prehensile pick-and-place — approach a screwdriver in tabletop clutter,
close the parallel jaw on its handle, and carry it to a basket; one grasp affordance,
free-space transport, and a CONTAINER goal that accepts the object anywhere inside
it. Here there is NO container and no goal pose for any single object: the goal is a
STRUCTURE. Four identical amber planks start flat on a high pedestal whose front face
is a sheer cliff; a red beacon line hangs in OPEN AIR beyond the cliff, farther out
than any single plank can reach (the line sits past L/2, so one plank alone tips off
the edge — its centre of mass leaves the support). Success is a property of the
ASSEMBLY: some plank surface, AT DECK HEIGHT, must project past the beacon line and
the whole stack must stand still, unsupported by anything but the pedestal and the
planks below. The seed's plan (carry the graspable thing to the target) is
meaningless here — carrying any one plank toward the beacon drops it into the void
(the beacon is intangible: it supports nothing), and no placement of a single plank
scores. The solver must instead exploit statics: corbel the planks — each course
offset a bit further out than the one below, tail-weighted so every partial stack's
centre of mass stays over its support (the harmonic-overhang construction) — so the
top plank cantilevers past the line while every plank below it is load-bearing
counterweight infrastructure.

success() (all judged on physical, settled poses):
  - the BEST GATED REACH crosses the line: over planks whose LOWEST oriented-box
    corner is at deck height or above (min corner z >= deck_top - reach_z_tol — a
    plank lying on the ground below, leaning on the cliff, or dangling mid-fall is
    gated OUT), the maximum oriented-box corner x reaches >= R (the beacon line);
  - the whole assembly is SETTLED: a pose-stillness streak — no plank drifted
    > 2 mm or rotated > ~1 deg for `streak_n` consecutive substeps (0.5 s) — a
    stack that is still collapsing, wobbling, or freshly written is not a shelf.
score() = 0.75 * STREAK-GATED latched reach fraction (best gated corner-x / R,
clamped to [0,1], latched in post_step only once the pose-stillness streak matures
— so a teleported statically-infeasible stack, which sits numerically still for a
few substeps before its collapse spins up, earns nothing); exactly 1.0 iff
success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - pedestal: STATIC gray block, top face (the deck) at `ped_h`, front face (the
    cliff) at x = 0, with a visual-only amber edge strip on the cliff edge;
  - planks: 4 identical DYNAMIC amber cuboids, `plank_l x plank_w x plank_t`, width
    strictly inside the Franka jaw (pinch-graspable across the width);
  - beacon: KINEMATIC bright-red vertical blade at x = R marking the goal line —
    NO COLLIDER (it is a light curtain, not a support: nothing can rest on it).

Per-episode randomization (verified by readback in smoke): the beacon line R
(how far the shelf must project), and every plank's spawn xy jitter + FREE YAW in
its deck slot (slot geometry guarantees no spawn interpenetration by a circle
bound — asserted in __post_init__). Heavy imports (isaaclab) are deferred so
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

_PLANKS = ("plank0", "plank1", "plank2", "plank3")


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CorbelReachSceneCfg(BaseCfg):
    """Config for `CorbelReachScene`. Honesty knobs asserted in `__post_init__`: the
    beacon line is strictly beyond a single plank's tipping limit (L/2) yet safely
    inside the 4-plank harmonic-corbel envelope; planks are jaw-graspable; spawn
    slots cannot interpenetrate at any sampled jitter/yaw."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    reach_z_tol: float = tunable(0.004)  # gated plank: min oriented corner z >= deck - this
    still_pos_tol: float = tunable(0.002)  # max plank centre drift (m) within one stillness
    # streak — POSE-based stillness, immune to GPU contact-jitter velocities
    still_quat_dot: float = tunable(0.99996)  # min |quat dot| vs the streak anchor (~1 deg)
    streak_n: int = tunable(60)  # substeps (0.5 s) of continuous pose-stillness before the
    # structure counts as settled and the latch fires (a teleported statically-infeasible
    # stack starts visibly collapsing within ~25 substeps — far short of 60; any state
    # write, slide, or wobble re-anchors the streak)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    reach_min: float = tunable(0.132)  # beacon line R range (m past the cliff at x=0)
    reach_max: float = tunable(0.155)
    slot_jit_x: float = tunable(0.030)  # plank spawn jitter about the deck slots (m)
    slot_jit_y: float = tunable(0.014)

    # --- info: structure ----------------------------------------------------------------------
    ped_l: float = info(0.80)  # pedestal x extent: deck spans [-ped_l, 0], cliff at x=0
    ped_w: float = info(0.60)  # pedestal y extent (deck spans +/- ped_w/2)
    ped_h: float = info(0.42)  # deck (pedestal top) height — the shelf's working height
    plank_l: float = info(0.24)  # plank length (the corbelling axis)
    plank_w: float = info(0.06)  # plank width — strictly inside the Franka jaw
    plank_t: float = info(0.024)  # plank thickness
    plank_mass: float = info(0.20)
    jaw_max: float = info(0.080)  # Franka parallel-jaw max opening (embodiment honesty)
    beacon_t: float = info(0.006)  # beacon blade x thickness (visual only, NO collider)
    beacon_w: float = info(0.44)  # beacon blade y extent
    beacon_h: float = info(0.24)  # beacon blade z extent (straddles deck height)
    slots: tuple = info(((-0.28, 0.14), (-0.28, -0.14), (-0.60, 0.14), (-0.60, -0.14)))
    friction_deck: float = info(0.80)
    friction_plank: float = info(0.80)
    contact_offset: float = info(0.002)
    ped_color: tuple = info((0.55, 0.55, 0.58))
    strip_color: tuple = info((0.95, 0.60, 0.10))
    plank_color: tuple = info((0.90, 0.60, 0.10))
    beacon_color: tuple = info((0.95, 0.08, 0.08))

    # Derived (filled in __post_init__).
    half_diag: float = field(default=None, init=False)  # plank xy half-diagonal

    def __post_init__(self) -> None:
        c = self
        c.half_diag = math.hypot(c.plank_l / 2, c.plank_w / 2)
        assert c.plank_w < c.jaw_max - 0.015, (
            "planks must be pinch-graspable across their width with real jaw margin")
        assert c.reach_min >= c.plank_l / 2 + 0.010, (
            "the beacon line must be STRICTLY beyond a single plank's tipping limit "
            "(max stable single-plank overhang = L/2) — one plank alone must fail")
        h4 = (c.plank_l / 2) * (1 + 1 / 2 + 1 / 3 + 1 / 4)  # 4-plank harmonic envelope
        assert c.reach_max + 0.05 <= 0.85 * h4, (
            "the beacon line must sit safely inside the 4-plank corbel envelope "
            "(reachable with >= 15% static margin plus slide headroom)")
        # Spawn slots can never interpenetrate at any jitter/yaw: circle bound.
        min_sep = 2 * c.half_diag + 0.002
        pts = c.slots
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                worst = math.hypot(max(0.0, abs(pts[i][0] - pts[j][0]) - 2 * c.slot_jit_x),
                                   max(0.0, abs(pts[i][1] - pts[j][1]) - 2 * c.slot_jit_y))
                assert worst >= min_sep, (
                    f"slots {i},{j} may interpenetrate at worst-case jitter "
                    f"({worst:.4f} < {min_sep:.4f})")
        # Slots (with jitter + any yaw) stay on the deck and clear of the cliff edge.
        for sx, sy in pts:
            assert sx + c.slot_jit_x + c.half_diag <= -0.010, (
                "spawned planks must never overhang the cliff edge")
            assert sx - c.slot_jit_x - c.half_diag >= -c.ped_l + 0.010, (
                "spawned planks must stay on the deck (back edge)")
            assert abs(sy) + c.slot_jit_y + c.half_diag <= c.ped_w / 2 - 0.010, (
                "spawned planks must stay on the deck (side edges)")
        assert c.reach_min > 0.05, "reach normalization needs a real positive line"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("corbel_reach")
class CorbelReachScene(BaseScene):
    cfg: CorbelReachSceneCfg

    def __init__(self, cfg: CorbelReachSceneCfg | None = None) -> None:
        super().__init__(cfg or CorbelReachSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            # STATIC pedestal: deck top at ped_h, cliff face at x = 0.
            "pedestal": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=(c.ped_l, c.ped_w, c.ped_h),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.friction_deck,
                        dynamic_friction=c.friction_deck - 0.1, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ped_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-c.ped_l / 2, 0.0, c.ped_h / 2)),
            ),
            # Visual-only cliff-edge strip (no collider): marks the corbelling edge.
            "edge_strip": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/EdgeStrip",
                spawn=sim_utils.CuboidCfg(
                    size=(0.012, c.ped_w, 0.0015),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.strip_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(-0.006, 0.0, c.ped_h + 0.00075)),
            ),
            # KINEMATIC beacon blade at x = R: NO collision_props -> NO collider.
            # It is a goal MARKER (a light curtain): nothing can rest on or against it.
            "beacon": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beacon",
                spawn=sim_utils.CuboidCfg(
                    size=(c.beacon_t, c.beacon_w, c.beacon_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.beacon_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=((c.reach_min + c.reach_max) / 2, 0.0, c.ped_h)),
            ),
        }
        for nm in _PLANKS:
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + nm.capitalize(),
                spawn=sim_utils.CuboidCfg(
                    size=(c.plank_l, c.plank_w, c.plank_t),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10, angular_damping=0.30,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plank_mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.friction_plank,
                        dynamic_friction=c.friction_plank - 0.1, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.plank_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.4, 0.0, c.ped_h + c.plank_t / 2 + 0.001)),
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
                # accurate velocity updates under external wrenches (the solve's slide)
                "enable_external_forces_every_iteration": True,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.planks: list[RigidObject] = [env.iscene[nm] for nm in _PLANKS]
        self.beacon: RigidObject = env.iscene["beacon"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.reach_R = torch.full((n,), (self.cfg.reach_min + self.cfg.reach_max) / 2,
                                  device=dev)
        self.reach_latch = torch.zeros(n, device=dev)
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._anchor_pos = torch.zeros(n, 4, 3, device=dev)
        self._anchor_quat = torch.zeros(n, 4, 4, device=dev)
        self._anchor_quat[..., 0] = 1.0

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the beacon line R and each plank's slot jitter + free
        yaw; zero the latch and the stillness streak."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, x: torch.Tensor, y: torch.Tensor, z: float,
                  yaw: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            if yaw is None:
                st[:, 3] = 1.0
            else:
                st[:, 3] = torch.cos(yaw / 2)
                st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- beacon line ---
        r = c.reach_min + torch.rand(m, device=dev) * (c.reach_max - c.reach_min)
        self.reach_R[env_ids] = r
        write(self.beacon, r, torch.zeros(m, device=dev), c.ped_h)

        # --- planks: one per slot, jittered, FREE yaw (slot geometry guarantees no
        # interpenetration at any sample — asserted in cfg.__post_init__) ---
        for k, (sx, sy) in enumerate(c.slots):
            px = sx + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jit_x
            py = sy + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jit_y
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            write(self.planks[k], px, py, c.ped_h + c.plank_t / 2 + 0.001, yaw)

        self.reach_latch[env_ids] = 0.0
        self._still[env_ids] = 0
        self._anchor_pos[env_ids] = 0.0
        self._anchor_quat[env_ids] = 0.0
        self._anchor_quat[env_ids, :, 0] = 1.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {nm: self.planks[k].data.root_state_w[env_ids].clone()
                               for k, nm in enumerate(_PLANKS)}
        out["beacon"] = self.beacon.data.root_state_w[env_ids].clone()
        for nm in ("reach_R", "reach_latch", "_still", "_anchor_pos", "_anchor_quat"):
            out[nm] = getattr(self, nm)[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, nm in enumerate(_PLANKS):
            self.planks[k].write_root_state_to_sim(state[nm], env_ids)
        self.beacon.write_root_state_to_sim(state["beacon"], env_ids)
        for nm in ("reach_R", "reach_latch", "_still", "_anchor_pos", "_anchor_quat"):
            getattr(self, nm)[env_ids] = state[nm]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A tall gray pedestal ({c.ped_h * 1000:.0f} mm high) ends in a sheer CLIFF: "
            f"its flat top (the deck) stops at an amber edge strip, and beyond that is "
            f"open air all the way to the floor. Four identical AMBER PLANKS "
            f"({c.plank_l * 1000:.0f} x {c.plank_w * 1000:.0f} x {c.plank_t * 1000:.0f} mm, "
            f"each graspable across its {c.plank_w * 1000:.0f} mm width) lie flat on the "
            f"deck. A vertical RED BEACON LINE hangs in mid-air past the cliff edge, "
            f"{c.reach_min * 1000:.0f}-{c.reach_max * 1000:.0f} mm out (it varies per "
            f"episode). The beacon is a light curtain: it is INTANGIBLE and supports "
            f"nothing.\n"
            f"Goal: build a freestanding corbelled shelf — stack the planks at the cliff "
            f"edge, each course shifted a little further out over the void than the one "
            f"below — until some plank surface AT DECK HEIGHT projects past the red line, "
            f"and the whole stack stands still. A single plank cannot do it: the line is "
            f"farther out than half a plank, so one plank slid past the edge tips and "
            f"falls. Planks lying on the floor below, leaning on the cliff, or falling do "
            f"not count — only plank surface at deck height, held up by the pedestal and "
            f"the planks beneath it. Credit grows with how far past the edge the shelf "
            f"stably reaches; full credit only when it crosses the line and holds."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stack the amber planks at the cliff edge into a stepped corbel — each plank "
            "overhanging a little more than the one below — until the top plank reaches "
            "past the red beacon line at deck height, then leave the stack standing "
            "still. The red line is intangible and cannot support anything."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _plank_extremes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,4) max oriented-box corner x and (N,4) min oriented-box corner z for
        each plank, env-local (support function of the box along +x and -z)."""
        from isaaclab.utils.math import matrix_from_quat

        c = self.cfg
        half = torch.tensor([c.plank_l / 2, c.plank_w / 2, c.plank_t / 2],
                            device=self.env.device)
        xs, zs = [], []
        for pl in self.planks:
            p = pl.data.root_pos_w - self.env_origins
            r = matrix_from_quat(pl.data.root_quat_w)  # (N,3,3)
            ext_x = (r[:, 0, :].abs() * half).sum(-1)  # support radius along world x
            ext_z = (r[:, 2, :].abs() * half).sum(-1)  # support radius along world z
            xs.append(p[:, 0] + ext_x)
            zs.append(p[:, 2] - ext_z)
        return torch.stack(xs, dim=-1), torch.stack(zs, dim=-1)

    def reach_now(self) -> torch.Tensor:
        """(N,) best gated reach (m past the cliff at x=0): max corner x over planks
        whose LOWEST corner is at deck height (min corner z >= ped_h - reach_z_tol);
        planks below deck height — floor, cliff face, mid-fall — earn nothing."""
        c = self.cfg
        cx, cz = self._plank_extremes()
        gated = cz >= c.ped_h - c.reach_z_tol
        cx = torch.where(gated, cx, torch.full_like(cx, -1.0))
        return cx.max(dim=-1).values.clamp(min=0.0)

    def _poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,4,3) env-local plank centres and (N,4,4) plank quats."""
        pos = torch.stack([pl.data.root_pos_w - self.env_origins
                           for pl in self.planks], dim=1)
        quat = torch.stack([pl.data.root_quat_w for pl in self.planks], dim=1)
        return pos, quat

    def settled(self) -> torch.Tensor:
        """(N,) bool: the whole assembly has held POSE-STILL for `streak_n`
        consecutive substeps (no plank drifted > still_pos_tol or rotated > ~1 deg
        since the streak anchor). Pose-based, so GPU contact-jitter velocities on a
        resting stack cannot block it — and a collapsing or freshly-written stack
        cannot pass it."""
        return self._still >= self.cfg.streak_n

    # ----- progress latch (step-coupled, stillness-streak gated) ------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Maintain the pose-stillness streak and latch the gated reach fraction
        only once the streak matures. A freshly teleported infeasible stack starts
        visibly collapsing within ~25 substeps — far short of `streak_n` — so it
        never latches; any state write, push, or wobble re-anchors the streak."""
        c = self.cfg
        pos, quat = self._poses()
        drift = (pos - self._anchor_pos).norm(dim=-1).amax(dim=1)  # (N,)
        qdot = (quat * self._anchor_quat).sum(-1).abs().amin(dim=1)  # (N,)
        cont = (drift <= c.still_pos_tol) & (qdot >= c.still_quat_dot)
        self._still = torch.where(cont, self._still + 1, torch.zeros_like(self._still))
        re_anchor = ~cont
        if bool(re_anchor.any()):
            idx = re_anchor.nonzero(as_tuple=True)[0]
            self._anchor_pos[idx] = pos[idx]
            self._anchor_quat[idx] = quat[idx]
        fire = self._still >= c.streak_n
        frac = (self.reach_now() / self.reach_R).clamp(0.0, 1.0)
        self.reach_latch = torch.where(
            fire, torch.maximum(self.reach_latch, frac), self.reach_latch)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: gated reach crosses the beacon line AND every plank settled."""
        return (self.reach_now() >= self.reach_R) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.75 * streak-latched gated reach fraction; exactly
        1.0 iff success(). Doing nothing scores ~0 (planks spawn behind the cliff
        edge, so the gated reach starts at 0); a collapsed stack keeps only what it
        honestly held still earlier."""
        base = (0.75 * self.reach_latch).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="corbel_reach", robot="null"))
