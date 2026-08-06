"""StickTriangleScene — assemble a closed triangle FRAME from three loose sticks on a build mat.

Derived from the maniskill/draw_triangle seed (a Franka with a rigid stick TRACES a triangle
outline on a canvas — continuous tool-tip path-following). This variant keeps the goal SHAPE (a
triangle) but flips the mechanism: nothing is drawn, nothing traces. Three rigid sticks of
UNEQUAL lengths lie scattered on the floor around a marked square build mat; the goal is to
ARRANGE the sticks flat on the mat so their ends meet pairwise at three corners, closing a real
triangle frame. A solver needs a construction plan — pick a consistent triangle placement from
the three given side lengths, then place + orient each stick so every corner gap closes within
tolerance — not a path-following plan. Order-free: sticks may be placed in any order.

Judged on PHYSICAL outcomes only (settled, flat-on-the-surface poses read back from sim):
  - a CORNER counts when the closest pair of endpoints of two sticks is within `joint_tol`,
    both sticks lie flat (axis near-horizontal, resting at ground height — no stacking), both
    are settled and on the mat, and the corner point itself is on the mat;
  - SUCCESS additionally requires all three corners simultaneously, a consistent cycle (each
    stick contributes its two DIFFERENT ends to its two corners — rejects side-by-side
    bundles), and an enclosed corner-triangle area of at least `area_frac_min` of the ideal
    Heron area for the three stick lengths (rejects degenerate chains).

Rubric (graded, latched): 0.05 per stick placed flat + settled on the mat, 0.20 per closed
corner (max partial 0.75, latched so transient achievements don't evaporate while adjusting);
exactly 1.0 iff success() holds NOW. Null policy scores 0 by construction (sticks spawn off-mat).

Assets are fully procedural: three plain rigid cylinders (distinct colors + lengths, identity
oracle-visible) and a thin dark kinematic mat plate (visual marker, collision OFF so it never
snags a stick) whose position randomizes per episode. Heavy imports (isaaclab) are deferred so
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

# stick pairs that can form the three corners: (0,1), (1,2), (2,0)
PAIRS = ((0, 1), (1, 2), (2, 0))


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class StickTriangleSceneCfg(BaseCfg):
    """Config for `StickTriangleScene`. Stick lengths satisfy the triangle inequality with a
    healthy margin (0.315 sum of the two short sides vs 0.20 long side), so a gap-closed frame
    is always a genuinely two-dimensional triangle."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    joint_tol: float = tunable(0.035)  # max endpoint-to-endpoint gap for a corner to count (m)
    flat_max_deg: float = tunable(20.0)  # stick axis within this of horizontal
    lift_tol: float = tunable(0.015)  # stick centre within stick_r + this of the ground (m);
    # rejects stacked / propped sticks (a stick resting ON another sits ~2*stick_r higher)
    settle_speed: float = tunable(0.05)  # max |lin vel| (every stick) when judging (m/s)
    area_frac_min: float = tunable(0.5)  # enclosed corner area >= this * Heron(stick lengths)
    pad_margin: float = tunable(0.010)  # sticks/corners must be this far inside the mat edge

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pad_jitter: float = tunable(0.10)  # uniform +/- xy jitter of the mat centre per episode
    scatter_radius: float = tunable(0.34)  # stick spawn ring radius around the mat centre
    scatter_jitter: float = tunable(0.05)  # uniform +/- xy jitter per stick at reset
    scatter_arc_jitter_deg: float = tunable(25.0)  # +/- jitter of each stick's ring angle
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per stick at reset (free)

    # --- info: structure ---------------------------------------------------------------------
    stick_lengths: tuple = info((0.200, 0.170, 0.145))  # unequal on purpose: the solver must
    # reason about WHICH stick spans WHICH side; corner gaps close only for a consistent layout
    stick_r: float = info(0.011)  # barrel radius; 2r = 22 mm < the 26+ mm oracle corner gap,
    # so gap-closed corners never interpenetrate (clearance is honest by construction)
    stick_mass: float = info(0.040)
    stick_colors: tuple = info(((0.85, 0.20, 0.20), (0.20, 0.70, 0.25), (0.25, 0.40, 0.90)))
    pad_size: float = info(0.40)  # square mat side; circumradius of the frame ~0.15 fits inside
    pad_thick: float = info(0.004)
    pad_color: tuple = info((0.12, 0.12, 0.14))
    contact_offset: float = info(0.002)  # small: default ~2 cm offsets would eat the 4 mm
    # corner clearance and glue neighbouring sticks together
    surface_z: float = info(0.0)  # ground plane (null smoke); robot bindings may lift later

    # Derived (filled in __post_init__).
    heron_area: float = field(default=None, init=False)  # ideal enclosed area of the frame

    def __post_init__(self) -> None:
        a, b, c = self.stick_lengths
        s = (a + b + c) / 2
        self.heron_area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stick_triangle")
class StickTriangleScene(BaseScene):
    cfg: StickTriangleSceneCfg

    def __init__(self, cfg: StickTriangleSceneCfg | None = None) -> None:
        super().__init__(cfg or StickTriangleSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the mat marker, and the three sticks (reset() re-places everything)."""
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
            # Build mat: kinematic rigid body with NO collider — a pure visual marker that can
            # still be teleported per episode via write_root_state_to_sim. No collider means no
            # 4 mm lip for a slid stick to snag on (the later robot stage pushes sticks around).
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
        for i, (length, rgb) in enumerate(zip(c.stick_lengths, c.stick_colors)):
            out[f"stick_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stick_" + str(i),
                spawn=sim_utils.CylinderCfg(
                    radius=c.stick_r,
                    height=length,
                    axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,  # resolve any residual overlap gently
                        linear_damping=0.05, angular_damping=0.05,  # cylinders ring/roll forever
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.stick_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.scatter_radius, 0.0, c.surface_z + c.stick_r + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),  # lying flat
                ),
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
        self.sticks: list[RigidObject] = [env.iscene[f"stick_{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        self._half_l = torch.tensor([length / 2 for length in c.stick_lengths], device=env.device)
        # Latched best partial credit per env (transient achievements don't evaporate).
        self.best_partial = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter the mat centre, scatter the sticks lying flat on a ring around
        it (well outside the mat, ~0.6 m arc apart so no corner exists at reset), free yaw."""
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

        # --- sticks: ring slots 120 deg apart around the mat centre, arc + xy jitter, flat
        # with free yaw ---
        arc_amp = math.radians(c.scatter_arc_jitter_deg)
        yaw_amp = math.radians(c.reset_yaw_deg)
        for i in range(3):
            ang = (math.radians(90.0 + 120.0 * i)
                   + (torch.rand(m, device=dev) * 2 - 1) * arc_amp)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = pad_xy[:, 0] + c.scatter_radius * torch.cos(ang)
            st[:, 1] = pad_xy[:, 1] + c.scatter_radius * torch.sin(ang)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 2] = c.surface_z + c.stick_r + 0.003
            # lying flat: q = qz(yaw) * qy(90 deg) -> (cy*c45, -sy*c45, cy*c45, sy*c45)
            c45 = math.cos(math.pi / 4)
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            st[:, 0:3] += origin
            self.sticks[i].write_root_state_to_sim(st, env_ids)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the best partial credit every physics substep.

        A diverged substep must not latch: torch.maximum PROPAGATES NaN, so one
        non-finite frame (agent code has raw sim access and can write a NaN force or
        velocity) would pin best_partial at NaN for the rest of the episode, and
        best_partial is in get_state/set_state so the NaN is checkpointed and restored
        by goto. A garbage frame earns NO credit; the latch keeps its last good value.
        Cost this pattern caused elsewhere: job f1175e3855f6fc25 died in
        wandb.Histogram on a single NaN reward out of 615 episodes."""
        self.best_partial = torch.maximum(
            self.best_partial,
            torch.nan_to_num(self._partial(), nan=0.0, posinf=0.0, neginf=0.0))

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "sticks": [s.data.root_state_w[env_ids].clone() for s in self.sticks],
            "best_partial": self.best_partial[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        for s, st in zip(self.sticks, state["sticks"]):
            s.write_root_state_to_sim(st, env_ids)
        self.best_partial[env_ids] = state["best_partial"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        lens = ", ".join(f"{int(round(length * 1000))} mm" for length in c.stick_lengths)
        cols = ("red", "green", "blue")
        sticks = " / ".join(f"{col} {int(round(length * 1000))} mm"
                            for col, length in zip(cols, c.stick_lengths))
        return (
            f"A dark square build mat ({c.pad_size * 100:.0f} cm on a side) lies on the floor, "
            f"and three rigid sticks of different lengths ({sticks}, each "
            f"{2 * c.stick_r * 1000:.0f} mm thick) lie scattered flat around it, outside the "
            f"mat.\nGoal: arrange the three sticks flat on the mat into a closed triangle "
            f"frame — the ends of neighbouring sticks meeting pairwise at three corners, each "
            f"corner gap under {c.joint_tol * 100:.1f} cm. Every stick must lie flat on the "
            f"surface (not stacked or propped), settled, fully on the mat, and each stick must "
            f"contribute both of its ends to two different corners (side lengths {lens}, so "
            f"only a genuine triangle closes all three gaps). Any placement order works; a "
            f"near-closed frame with a gapped corner, a frame built off the mat, or sticks "
            f"laid side by side do not count."
        )

    # ----- progress / rubric ----------------------------------------------------------------
    def _stick_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos (N,3,3) local to env origin, quat (N,3,4), |lin vel| (N,3))."""
        pos = torch.stack([s.data.root_pos_w for s in self.sticks], dim=1)
        pos = pos - self.env_origins[:, None, :]
        quat = torch.stack([s.data.root_quat_w for s in self.sticks], dim=1)
        vel = torch.stack([s.data.root_lin_vel_w.norm(dim=-1) for s in self.sticks], dim=1)
        return pos, quat, vel

    def _endpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(ends (N,3,2,3): both endpoints of each stick, local; axis (N,3,3): world axis)."""
        from isaaclab.utils.math import quat_apply

        pos, quat, _v = self._stick_tensors()
        n = pos.shape[0]
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * 3, 3)
        axis = quat_apply(quat.reshape(n * 3, 4), ez).reshape(n, 3, 3)
        half = self._half_l[None, :, None]
        ends = torch.stack([pos - axis * half, pos + axis * half], dim=2)
        return ends, axis

    def _pad_xy(self) -> torch.Tensor:
        """(N,2) mat centre, local to env origin."""
        return (self.pad.data.root_pos_w - self.env_origins)[:, :2]

    def _on_pad(self, xy: torch.Tensor) -> torch.Tensor:
        """(...,) bool: xy (local, shape (...,2)) inside the mat minus `pad_margin`."""
        c = self.cfg
        half = c.pad_size / 2 - c.pad_margin
        d = xy - self._pad_xy().reshape((-1,) + (1,) * (xy.dim() - 2) + (2,))
        return (d.abs() <= half).all(dim=-1)

    def stick_valid(self) -> torch.Tensor:
        """(N,3) bool: flat on the ground (axis near-horizontal, centre at resting height —
        no stacking), settled, and centred on the mat."""
        c = self.cfg
        pos, _q, vel = self._stick_tensors()
        _ends, axis = self._endpoints()
        flat = axis[:, :, 2].abs() <= math.sin(math.radians(c.flat_max_deg))
        rest_z = c.surface_z + c.stick_r
        grounded = (pos[:, :, 2] - rest_z).abs() <= c.lift_tol
        settled = vel < c.settle_speed
        return flat & grounded & settled & self._on_pad(pos[:, :, :2])

    def corners(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per stick pair p in PAIRS: (gap (N,3), counted (N,3), corner xy (N,3,2),
        end index used (N,3,2) — which endpoint of (first, second) stick the corner uses)."""
        ends, _axis = self._endpoints()
        n = ends.shape[0]
        valid = self.stick_valid()
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
        """(N,) bool: each stick contributes DIFFERENT endpoints to its two corners — the
        closure certificate that rejects side-by-side bundles / open chains. PAIRS is the
        cycle (0,1),(1,2),(2,0): stick 0 is 'first' in pair 0 and 'second' in pair 2, etc."""
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
        """(N,) bool: all three corners closed on the mat, consistent end-to-end cycle, and a
        genuinely two-dimensional frame (area gate) — all judged on the CURRENT settled state."""
        c = self.cfg
        _gaps, cnt, xy, use = self.corners()
        return (cnt.all(dim=1) & self._cycle_ok(use)
                & (self._area(xy) >= c.area_frac_min * c.heron_area))

    def _partial(self) -> torch.Tensor:
        """(N,) float: 0.05 per valid stick on the mat + 0.20 per closed corner (max 0.75)."""
        _gaps, cnt, _xy, _use = self.corners()
        return (0.05 * self.stick_valid().sum(dim=1) + 0.20 * cnt.sum(dim=1)).clamp(max=0.75)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched partial credit; exactly 1.0 iff success() NOW."""
        self.best_partial = torch.maximum(
            self.best_partial,
            torch.nan_to_num(self._partial(), nan=0.0, posinf=0.0, neginf=0.0))
        return torch.where(self.success(),
                           torch.ones_like(self.best_partial), self.best_partial)


register_env("simgen", lambda: EnvCfg(scene="stick_triangle", robot="null"))
