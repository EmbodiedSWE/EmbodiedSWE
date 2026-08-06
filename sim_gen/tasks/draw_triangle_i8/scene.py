"""BarTriangleScene — build a closed triangle FRAME from three loose bars on a build mat.

Derived from the maniskill/draw_triangle seed (a Franka holding a rigid tool TRACES a
triangle outline on a canvas — continuous, contact-maintained path-following, judged on
the traced path). This variant keeps the goal SHAPE (a triangle) and replaces the
mechanism wholesale: nothing is drawn and nothing traces. Three rigid square-section bars
of UNEQUAL lengths (distinct colors) wait in staging spots on the floor beside a dark
square build mat; the goal is a persistent, settled STRUCTURE — the bars arranged flat on
the mat so their ends meet pairwise at three corners, closing a real triangle frame. A
solver needs a construction plan (derive a consistent triangle placement from the three
given side lengths, then pick / orient / place each bar and close per-corner tolerances),
not a path-following plan. Placement order is free.

Judged on PHYSICAL outcomes only (settled poses read back from sim):
  - a CORNER counts when the closest pair of endpoints of two bars is within
    `joint_tol`, both bars lie flat at ground height (no stacking / propping), both are
    settled, both are on the mat (centre and both ends), and the corner point is on the
    mat;
  - SUCCESS additionally requires all three corners simultaneously, a consistent cycle
    (each bar contributes its two DIFFERENT ends to its two corners — rejects
    side-by-side bundles), and an enclosed corner-triangle area of at least
    `area_frac_min` of the ideal Heron area for the three bar lengths (rejects
    degenerate near-flat layouts).

Rubric (graded, latched): 0.05 per bar placed flat + settled + on the mat, 0.20 per
closed corner (partial capped at 0.75, latched in post_step so transient achievements
don't evaporate while adjusting); exactly 1.0 iff success() holds NOW. The null policy
scores 0 by construction (bars spawn in staging spots clear of the mat).

Assets are fully procedural (no external files): three colored cuboid bars (square
cross-section — they cannot roll away after release) and a thin dark kinematic mat plate
whose position randomizes per episode. Heavy imports (isaaclab) are deferred so importing
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

# bar pairs that can form the three corners: (0,1), (1,2), (2,0)
PAIRS = ((0, 1), (1, 2), (2, 0))


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BarTriangleSceneCfg(BaseCfg):
    """Config for `BarTriangleScene`. Bar lengths satisfy the triangle inequality with a
    healthy margin (0.155 + 0.130 = 0.285 vs 0.180), so a gap-closed frame is always a
    genuinely two-dimensional triangle even at the gap tolerance."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    joint_tol: float = tunable(0.040)  # max endpoint-to-endpoint gap for a corner to count (m)
    flat_max_deg: float = tunable(8.0)  # bar long axis within this of horizontal (a bar
    # propped with one end on a neighbour tilts ~6.7 deg AND lifts that end ~18 mm — both
    # gates below reject it; an honestly placed bar reads < 1 deg)
    lift_tol: float = tunable(0.012)  # bar centre AND both ends within bar_w/2 + this of the
    # ground (m); rejects stacked/propped/woven bars (an end resting ON another bar sits a
    # full bar-width high)
    settle_speed: float = tunable(0.05)  # max |lin vel| (every bar) when judging (m/s)
    area_frac_min: float = tunable(0.5)  # enclosed corner area >= this * Heron(bar lengths)
    pad_margin: float = tunable(0.010)  # bar centres/ends/corners this far inside the mat edge

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pad_jitter: float = tunable(0.040)  # uniform +/- xy jitter of the mat centre per episode
    slot_bearing_jitter_deg: float = tunable(12.0)  # per-slot bearing jitter (see reset())
    slot_radial_jitter: float = tunable(0.040)  # per-slot radial depth jitter
    slot_xy_jitter: float = tunable(0.008)  # small residual xy jitter per bar
    slot_yaw_jitter_deg: float = tunable(20.0)  # bar yaw jitter about the tangential heading

    # --- info: structure ---------------------------------------------------------------------
    bar_lengths: tuple = info((0.180, 0.155, 0.130))  # unequal on purpose: the solver must
    # reason about WHICH bar spans WHICH side; the corner gaps only close for a consistent
    # triangle layout of the three given side lengths
    bar_w: float = info(0.018)  # square cross-section side; bars rest flat and cannot roll
    bar_mass: float = info(0.060)
    bar_colors: tuple = info(((0.85, 0.18, 0.18), (0.18, 0.68, 0.24), (0.22, 0.38, 0.90)))
    pad_size: float = info(0.36)  # square mat side; the frame (circumradius ~0.12) fits inside
    pad_thick: float = info(0.0015)
    pad_color: tuple = info((0.10, 0.10, 0.12))
    contact_offset: float = info(0.002)  # small: default ~2 cm offsets would eat the ~2.4 cm
    # corner clearances and glue neighbouring bars together
    surface_z: float = info(0.0)  # ground plane height
    # Staging geometry: three spots on a polar fan about `stage_anchor` (where a tabletop
    # manipulator base plausibly sits — solve.py places its Franka base here; the scene itself
    # is embodiment-free and the smoke runs robot="null"). Spots 1/2 mirror at +/- ~46 deg,
    # spot 3 sits deeper on a random side; all clear of the mat under every jitter draw.
    stage_anchor: tuple = info((-0.52, 0.0))
    slot12_bearing_deg: float = info(40.0)  # + jitter in [0, slot_bearing_jitter_deg]
    slot12_radius: float = info(0.42)  # + jitter in [0, slot_radial_jitter]
    slot3_bearing_deg: float = info(66.0)
    slot3_radius: float = info(0.56)

    # Derived (filled in __post_init__).
    heron_area: float = field(default=None, init=False)  # ideal enclosed area of the frame

    def __post_init__(self) -> None:
        a, b, c = self.bar_lengths
        s = (a + b + c) / 2
        self.heron_area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))


# ----- geometry helper (shared by solve.py's plan and smoke.py's probes) ------------------------
def frame_layout(cfg: BarTriangleSceneCfg, center_xy, rot: float,
                 corner_gap: float = 0.024, deltas=None):
    """Pure-python ideal frame layout: where the three bars sit so the frame closes with
    ~`corner_gap` endpoint separation at every corner.

    Corner k of the VERTEX triangle joins bar k-1 and bar k; each adjacent bar end is
    retracted delta_k from vertex k along its own side, so the endpoint separation there is
    2*delta_k*sin(theta_k/2). Vertex side k therefore measures bar_k + delta_k + delta_{k+1};
    deltas and vertex angles are solved by fixed-point iteration. `deltas` overrides the
    per-corner retractions directly (signed; negative = ends overshoot and cross — used by
    the smoke's pinwheel probe).

    Returns (centers[3](x, y), yaws[3], corners[3](x, y), gaps[3]) in the mat/world frame.
    """
    L = list(cfg.bar_lengths)

    def vertex_tri(d):
        # side k runs V_k -> V_{k+1}; retraction at V_k is d[k]
        S = [L[k] + d[k] + d[(k + 1) % 3] for k in range(3)]
        ang = []
        for k in range(3):
            sk, skm, opp = S[k], S[(k - 1) % 3], S[(k + 1) % 3]
            ang.append(math.acos(max(-1.0, min(1.0, (sk * sk + skm * skm - opp * opp)
                                               / (2 * sk * skm)))))
        return S, ang

    if deltas is None:
        d = [corner_gap / 2.0] * 3
        for _ in range(8):
            _S, ang = vertex_tri(d)
            d = [corner_gap / (2.0 * math.sin(ang[k] / 2.0)) for k in range(3)]
    else:
        d = list(deltas)
    S, ang = vertex_tri(d)

    # vertex coordinates: V0 at origin, V1 along +x, V2 above; then centre + rotate
    v = [(0.0, 0.0), (S[0], 0.0), (0.0, 0.0)]
    x2 = (S[0] ** 2 + S[2] ** 2 - S[1] ** 2) / (2 * S[0])
    v[2] = (x2, math.sqrt(max(S[2] ** 2 - x2 ** 2, 1e-12)))
    cx = sum(p[0] for p in v) / 3.0
    cy = sum(p[1] for p in v) / 3.0
    cr, sr = math.cos(rot), math.sin(rot)

    def xf(p):
        px, py = p[0] - cx, p[1] - cy
        return (center_xy[0] + cr * px - sr * py, center_xy[1] + sr * px + cr * py)

    V = [xf(p) for p in v]
    centers, yaws, gaps = [], [], []
    ends = []
    for k in range(3):
        a, b = V[k], V[(k + 1) % 3]
        ux, uy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        e0 = (a[0] + d[k] * ux, a[1] + d[k] * uy)
        e1 = (b[0] - d[(k + 1) % 3] * ux, b[1] - d[(k + 1) % 3] * uy)
        ends.append((e0, e1))
        centers.append(((e0[0] + e1[0]) / 2, (e0[1] + e1[1]) / 2))
        yaws.append(math.atan2(uy, ux))
    for k in range(3):  # corner k joins bar k-1 (its far end) and bar k (its near end)
        p, q = ends[(k - 1) % 3][1], ends[k][0]
        gaps.append(math.hypot(p[0] - q[0], p[1] - q[1]))
    return centers, yaws, V, gaps


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bar_triangle")
class BarTriangleScene(BaseScene):
    cfg: BarTriangleSceneCfg

    def __init__(self, cfg: BarTriangleSceneCfg | None = None) -> None:
        super().__init__(cfg or BarTriangleSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the mat marker, and the three bars (reset() re-places everything)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
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
            # Build mat: kinematic rigid body, pure visual marker (no collision props ->
            # no collider, so there is no lip for a bar to snag on), teleported per episode.
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(c.pad_size, c.pad_size, c.pad_thick),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.surface_z + c.pad_thick / 2)),
            ),
        }
        for i, (length, rgb) in enumerate(zip(c.bar_lengths, c.bar_colors)):
            out[f"bar_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + str(i),
                spawn=sim_utils.CuboidCfg(
                    size=(length, c.bar_w, c.bar_w),  # long axis = local +x
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.9),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.45, 0.35 * (i - 1), c.surface_z + c.bar_w / 2 + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={"solver_type": 1, "bounce_threshold_velocity": 0.2},
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.pad: RigidObject = env.iscene["pad"]
        self.bars: list[RigidObject] = [env.iscene[f"bar_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        self._half_l = torch.tensor([length / 2 for length in c.bar_lengths], device=env.device)
        # Latched best partial credit per env (transient achievements don't evaporate).
        self.best_partial = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter the mat centre; deal the three bars (random permutation)
        onto the three staging spots of a polar fan about `stage_anchor`, lying flat with
        near-tangential yaw. Spots stay clear of the mat and of each other under every
        jitter draw (worst-case bar-bar centreline distance ~28 mm > bar width; verified
        numerically over the full jitter ranges)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        self.best_partial[env_ids] = 0.0

        # --- mat: kinematic teleport with xy jitter ---
        pad_xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = pad_xy
        st[:, 2] = c.surface_z + c.pad_thick / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(st, env_ids)

        # --- staging spots (base-polar fan about stage_anchor) ---
        anchor = torch.tensor(c.stage_anchor, device=dev)
        bj = math.radians(c.slot_bearing_jitter_deg)
        u = torch.rand(m, 3, device=dev)          # bearing jitters
        r = torch.rand(m, 3, device=dev)          # radial jitters
        side3 = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        b12 = math.radians(c.slot12_bearing_deg)
        b3 = math.radians(c.slot3_bearing_deg)
        bearings = torch.stack([
            b12 + u[:, 0] * bj,
            -(b12 + u[:, 1] * bj),
            side3 * (b3 + u[:, 2] * bj),
        ], dim=1)
        radii = torch.stack([
            c.slot12_radius + r[:, 0] * c.slot_radial_jitter,
            c.slot12_radius + r[:, 1] * c.slot_radial_jitter,
            c.slot3_radius + r[:, 2] * c.slot_radial_jitter,
        ], dim=1)
        slot_xy = anchor[None, None, :] + radii[..., None] * torch.stack(
            [torch.cos(bearings), torch.sin(bearings)], dim=-1)
        slot_xy = slot_xy + (torch.rand(m, 3, 2, device=dev) * 2 - 1) * c.slot_xy_jitter
        # near-tangential yaw (perpendicular to the spot's bearing), +/- yaw jitter
        yaw = (bearings + math.pi / 2
               + (torch.rand(m, 3, device=dev) * 2 - 1) * math.radians(c.slot_yaw_jitter_deg))

        # --- deal bars to spots: random permutation per env ---
        perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)  # bar i -> spot perm[:, i]
        for i in range(3):
            spot = perm[:, i]
            ar = torch.arange(m, device=dev)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slot_xy[ar, spot]
            st[:, 2] = c.surface_z + c.bar_w / 2 + 0.002
            half = yaw[ar, spot] / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.bars[i].write_root_state_to_sim(st, env_ids)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the best partial credit every physics substep. torch.maximum PROPAGATES
        NaN, so a diverged frame is scrubbed before latching — a garbage frame earns no
        credit and the latch keeps its last good value."""
        self.best_partial = torch.maximum(
            self.best_partial,
            torch.nan_to_num(self._partial(), nan=0.0, posinf=0.0, neginf=0.0))

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "bars": [b.data.root_state_w[env_ids].clone() for b in self.bars],
            "best_partial": self.best_partial[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        for b, st in zip(self.bars, state["bars"]):
            b.write_root_state_to_sim(st, env_ids)
        self.best_partial[env_ids] = state["best_partial"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        cols = ("red", "green", "blue")
        bars = " / ".join(f"{col} {int(round(length * 1000))} mm"
                          for col, length in zip(cols, c.bar_lengths))
        return (
            f"A dark square build mat ({c.pad_size * 100:.0f} cm on a side) lies flat on the "
            f"floor, and three rigid square-section bars of different lengths ({bars}, each "
            f"{c.bar_w * 1000:.0f} mm thick) lie flat in staging spots on the floor around "
            f"it, clear of the mat.\nGoal: build a closed triangle FRAME on the mat from the "
            f"three bars — lay them flat on the mat so the ends of neighbouring bars meet "
            f"pairwise at three corners, every corner's endpoint gap under "
            f"{c.joint_tol * 100:.1f} cm. Each bar is one side, identified by its color and "
            f"length; only a consistent triangle placement of the three given side lengths "
            f"closes all three gaps at once. Every bar must lie flat at ground level — level to "
            f"within a few degrees with both ends down, not resting on or leaning against "
            f"another bar — settled, entirely on the mat (both ends and the middle), and "
            f"each bar must contribute both of its ends to two different corners. Bars may "
            f"be placed in any order. A frame with one gapped corner, a frame built beside "
            f"the mat, bars laid side by side, bars radiating from a single hub, or a woven "
            f"frame with ends resting on neighbours do not count."
        )

    # ----- readouts / rubric ----------------------------------------------------------------
    def _bar_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos (N,3,3) local to env origin, quat (N,3,4), |lin vel| (N,3))."""
        pos = torch.stack([b.data.root_pos_w for b in self.bars], dim=1)
        pos = pos - self.env_origins[:, None, :]
        quat = torch.stack([b.data.root_quat_w for b in self.bars], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.bars], dim=1)
        return pos, quat, vel

    def _endpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(ends (N,3,2,3): both endpoints of each bar, local; axis (N,3,3): world long axis)."""
        from isaaclab.utils.math import quat_apply

        pos, quat, _v = self._bar_tensors()
        n = pos.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=pos.device).expand(n * 3, 3)
        axis = quat_apply(quat.reshape(n * 3, 4), ex).reshape(n, 3, 3)
        half = self._half_l[None, :, None]
        ends = torch.stack([pos - axis * half, pos + axis * half], dim=2)
        return ends, axis

    def _pad_xy(self) -> torch.Tensor:
        """(N,2) mat centre, local to env origin."""
        return (self.pad.data.root_pos_w - self.env_origins)[:, :2]

    def _on_pad(self, xy: torch.Tensor) -> torch.Tensor:
        """(...,) bool: xy (local, shape (...,2)) inside the mat minus `pad_margin`. The mat
        never rotates, so the test is axis-aligned."""
        c = self.cfg
        half = c.pad_size / 2 - c.pad_margin
        d = xy - self._pad_xy().reshape((-1,) + (1,) * (xy.dim() - 2) + (2,))
        return (d.abs() <= half).all(dim=-1)

    def bar_valid(self) -> torch.Tensor:
        """(N,3) bool: flat at ground height (long axis near-horizontal, centre at resting
        height — no stacking), settled, and entirely on the mat (centre + both ends)."""
        c = self.cfg
        pos, _q, vel = self._bar_tensors()
        ends, axis = self._endpoints()
        flat = axis[:, :, 2].abs() <= math.sin(math.radians(c.flat_max_deg))
        rest_z = c.surface_z + c.bar_w / 2
        grounded = ((pos[:, :, 2] - rest_z).abs() <= c.lift_tol) \
            & ((ends[:, :, :, 2] - rest_z).abs().amax(dim=2) <= c.lift_tol)
        settled = vel < c.settle_speed
        on = (self._on_pad(pos[:, :, :2])
              & self._on_pad(ends[:, :, 0, :2]) & self._on_pad(ends[:, :, 1, :2]))
        return flat & grounded & settled & on

    def corners(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per bar pair p in PAIRS: (gap (N,3), counted (N,3), corner xy (N,3,2),
        end index used (N,3,2) — which endpoint of (first, second) bar the corner uses)."""
        ends, _axis = self._endpoints()
        n = ends.shape[0]
        valid = self.bar_valid()
        gaps, cnt, xy, use = [], [], [], []
        for a, b in PAIRS:
            d = (ends[:, a, :, None, :] - ends[:, b, None, :, :]).norm(dim=-1).reshape(n, 4)
            gap, arg = d.min(dim=1)
            ia, ib = arg // 2, arg % 2
            ar = torch.arange(n, device=ends.device)
            pa, pb = ends[ar, a, ia], ends[ar, b, ib]
            mid = (pa + pb) / 2
            ok = (gap < self.cfg.joint_tol) & valid[:, a] & valid[:, b] \
                & self._on_pad(mid[:, :2])
            gaps.append(gap)
            cnt.append(ok)
            xy.append(mid[:, :2])
            use.append(torch.stack([ia, ib], dim=1))
        return (torch.stack(gaps, dim=1), torch.stack(cnt, dim=1),
                torch.stack(xy, dim=1), torch.stack(use, dim=1))

    def _cycle_ok(self, use: torch.Tensor) -> torch.Tensor:
        """(N,) bool: each bar contributes DIFFERENT endpoints to its two corners — the
        closure certificate that rejects side-by-side bundles / open chains. PAIRS is the
        cycle (0,1),(1,2),(2,0): bar 0 is 'first' in pair 0 and 'second' in pair 2, etc."""
        s0 = use[:, 0, 0] != use[:, 2, 1]
        s1 = use[:, 1, 0] != use[:, 0, 1]
        s2 = use[:, 2, 0] != use[:, 1, 1]
        return s0 & s1 & s2

    def _area(self, corner_xy: torch.Tensor) -> torch.Tensor:
        """(N,) shoelace area of the three corner points."""
        x, y = corner_xy[:, :, 0], corner_xy[:, :, 1]
        return 0.5 * (x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0])
                      + x[:, 2] * (y[:, 0] - y[:, 1])).abs()

    def success(self) -> torch.Tensor:
        """(N,) bool: all three corners closed on the mat, consistent end-to-end cycle, and
        a genuinely two-dimensional frame (area gate) — judged on the CURRENT settled state."""
        c = self.cfg
        _gaps, cnt, xy, use = self.corners()
        return (cnt.all(dim=1) & self._cycle_ok(use)
                & (self._area(xy) >= c.area_frac_min * c.heron_area))

    def _partial(self) -> torch.Tensor:
        """(N,) float: 0.05 per valid bar on the mat + 0.20 per closed corner (max 0.75)."""
        _gaps, cnt, _xy, _use = self.corners()
        return (0.05 * self.bar_valid().sum(dim=1) + 0.20 * cnt.sum(dim=1)).clamp(max=0.75)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched partial credit; exactly 1.0 iff success() NOW."""
        self.best_partial = torch.maximum(
            self.best_partial,
            torch.nan_to_num(self._partial(), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.where(self.success(),
                           torch.ones_like(self.best_partial), self.best_partial)

    def corner_report(self) -> str:
        """Human-readable env-0 readout (used by solve/smoke prints)."""
        gaps, cnt, _xy, use = self.corners()
        valid = self.bar_valid()
        return (f"gaps_mm={[round(float(g) * 1000, 1) for g in gaps[0]]} "
                f"counted={[bool(v) for v in cnt[0]]} valid={[bool(v) for v in valid[0]]} "
                f"cycle_ok={bool(self._cycle_ok(use)[0])} "
                f"score={float(self.score()[0]):.3f} success={bool(self.success()[0])}")


register_env("simgen", lambda: EnvCfg(scene="bar_triangle", robot="null"))
