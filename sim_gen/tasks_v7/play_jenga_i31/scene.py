"""CompassDialsScene — set the three compass dials: swing each colored ARROW about its
fixed pivot pin until its tip points at the same-colored post (sim_gen task
`play_jenga_i31`).

Derived from rlbench/play_jenga, but STRATEGICALLY different: the seed is a
free-body EXTRACTION under a stability constraint — pull one designated block out of
a tower of identical loose cuboids without toppling the rest; every judged quantity
is a POSITION (block out, tower still standing), and the plan is one careful linear
slide through free space. Here NOTHING is transported and nothing may leave its
place: each jenga-bar-shaped body is CAPTIVE on a fixed vertical pin through a square
hole in its body (a free pivot built from contact, no articulation), and the entire
goal is ORIENTATION — three independent headings, each bound by COLOR to a target
post whose azimuth is randomized per episode. A solver needs a different PLAN
(perceive three azimuth targets, swing each arrow about its pivot with tangential
nudges, stop inside an angular tolerance, disturb nothing off its pin) and a
different code structure (heading/angle rubric with per-dial latches, not
position-of-a-block checks). The seed's whole strategy — slide a bar out and carry it
somewhere — is exactly the off-pin end state this rubric rejects (smoke check).

The scene (fully procedural, no external assets):
  - a light-gray kinematic BENCH slab (0.96 x 0.48 m, top at 0.10 m) carrying
    everything;
  - three steel PIVOT PINS (kinematic, 18 mm dia, 55 mm tall) planted upright in a
    row along the bench;
  - three ARROW bars (dynamic), one RED, one GREEN, one BLUE: a flat bar 14 mm thick
    with a 28 mm square hole around its pivot, a shaft, a two-plate ARROWHEAD at the
    long end (tip 110 mm from the pivot) and a short tail — the arrow rests FLAT on
    the bench with the pin through its hole, so it can spin freely about the pin but
    cannot translate away;
  - three TARGET POSTS (kinematic, 30 mm dia, 120 mm tall), one RED, one GREEN, one
    BLUE, each standing on the bench at 0.19 m from its same-colored pin at a random
    azimuth; plus one YELLOW DECOY post that matches no arrow.

Judged when settled. A dial counts iff its arrow's TIP heading is within
`align_tol_deg` of the azimuth from its pin to its SAME-colored post, the arrow is
still seated on its pin (hole around the pin, resting flat in the on-bench height
band), and it is still. success() iff ALL THREE dials count simultaneously. No
execution order is required. score(): per dial, latched credit — 0.10 * best
fractional heading progress toward the target (on-pin gated, normalized by the
episode's own start error) + 0.20 * dial ever aligned-and-settled — capped at 0.90;
exactly 1.0 iff success(). Doing nothing scores ~0 (spawn headings are forced
>= `init_sep_deg` away from their targets); a flipped arrow (tail toward the post),
an arrow aimed at a wrong-color or the decoy post, and an arrow taken OFF its pin and
laid pointing correctly all score nothing for that dial.

Per-episode randomization (verified by readback in smoke): pin xy jitter, each
post's azimuth on its ring (window `post_az_window_deg` on either side of the row,
rejection-sampled so from every pin the azimuth separation between its OWN post and
every OTHER post/decoy is >= `post_sep_deg` — wrong-target rejection is always
well-defined), the decoy's azimuth, and each arrow's spawn heading (forced
>= `init_sep_deg` from its target). Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawner (the arrow bar) --------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float,
         yaw_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        sxf.AddRotateZOp().Set(float(yaw_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _rigid_dynamic(root, mass: float) -> None:
    """Dynamic rigid-body armor on a compound root: mass (PhysX derives inertia from
    the child colliders), damping so the light arrow settles promptly, no sleeping
    while we judge velocities, and the depenetration cap."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _spawn_arrow(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a DYNAMIC arrow bar at `prim_path`. Origin = the PIVOT-HOLE CENTRE at
    mid-thickness; local +x is the POINTING direction (through the arrowhead tip)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_dynamic(root, cfg.mass)

    co, t = cfg.contact_offset, cfg.bar_t
    hh = cfg.hole_half
    eh = cfg.eye_out / 2
    side = eh - hh
    color, hcolor = cfg.color, cfg.head_color
    # square pivot-hole frame (four boxes around the origin)
    _box(stage, f"{prim_path}/eye_yp", (cfg.eye_out, side, t), (0.0, hh + side / 2, 0.0), color, co)
    _box(stage, f"{prim_path}/eye_yn", (cfg.eye_out, side, t), (0.0, -(hh + side / 2), 0.0), color, co)
    _box(stage, f"{prim_path}/eye_xp", (side, 2 * hh, t), (hh + side / 2, 0.0, 0.0), color, co)
    _box(stage, f"{prim_path}/eye_xn", (side, 2 * hh, t), (-(hh + side / 2), 0.0, 0.0), color, co)
    # shaft toward the tip
    x0, x1 = eh, cfg.head_base_x
    _box(stage, f"{prim_path}/shaft", (x1 - x0, cfg.shaft_w, t),
         ((x0 + x1) / 2, 0.0, 0.0), color, co)
    # arrowhead: two angled plates converging at the TIP (local +x = pointing end)
    hl, ha = cfg.head_len, math.radians(cfg.head_ang_deg)
    cx = cfg.tip_x - (hl / 2) * math.cos(ha)
    cy = (hl / 2) * math.sin(ha)
    _box(stage, f"{prim_path}/head_a", (hl, cfg.head_w, t), (cx, -cy, 0.0), hcolor, co,
         yaw_deg=cfg.head_ang_deg)
    _box(stage, f"{prim_path}/head_b", (hl, cfg.head_w, t), (cx, cy, 0.0), hcolor, co,
         yaw_deg=-cfg.head_ang_deg)
    # short tail (the blunt end — visually distinct from the arrowhead)
    _box(stage, f"{prim_path}/tail", (cfg.tail_len, cfg.shaft_w, t),
         (-(eh + cfg.tail_len / 2), 0.0, 0.0), color, co)
    return root


def _arrow_spawner_cfg(*, key: str, hole_half: float, eye_out: float, bar_t: float,
                       shaft_w: float, head_base_x: float, tip_x: float, head_len: float,
                       head_w: float, head_ang_deg: float, tail_len: float, mass: float,
                       color: tuple, head_color: tuple, contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "arrow" not in _SPAWNER_CACHE:

        @configclass
        class ArrowSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_arrow)
            hole_half: float = 0.014
            eye_out: float = 0.056
            bar_t: float = 0.014
            shaft_w: float = 0.024
            head_base_x: float = 0.082
            tip_x: float = 0.110
            head_len: float = 0.055
            head_w: float = 0.012
            head_ang_deg: float = 30.0
            tail_len: float = 0.024
            mass: float = 0.10
            color: tuple = (0.8, 0.2, 0.2)
            head_color: tuple = (0.5, 0.1, 0.1)
            contact_offset: float = 0.001

        _SPAWNER_CACHE["arrow"] = ArrowSpawnerCfg

    return _SPAWNER_CACHE["arrow"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        hole_half=hole_half, eye_out=eye_out, bar_t=bar_t, shaft_w=shaft_w,
        head_base_x=head_base_x, tip_x=tip_x, head_len=head_len, head_w=head_w,
        head_ang_deg=head_ang_deg, tail_len=tail_len, mass=mass, color=color,
        head_color=head_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CompassDialsSceneCfg(BaseCfg):
    """Config for `CompassDialsScene`. Honesty knobs asserted in `__post_init__`: the
    arrow spins on its pin with real clearance, target posts stand beyond the arrow's
    sweep (an arrow can never touch a post), adjacent sweeps cannot clash, and spawn
    headings start far enough from their targets that the null policy scores ~0."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    align_tol_deg: float = tunable(10.0)  # arrow heading within this of the pin->post azimuth
    on_pin_xy_tol: float = tunable(0.012)  # hole centre within this of the pin axis (m)
    flat_max_deg: float = tunable(12.0)  # arrow +z within this of world-up (resting flat)
    settle_lin: float = tunable(0.05)  # max |lin vel| of every arrow when judging (m/s)
    settle_ang: float = tunable(0.30)  # max |ang vel| of every arrow when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    pin_jitter: float = tunable(0.015)  # uniform +/- xy jitter of each pivot pin at reset
    post_az_window_deg: tuple = tunable((50.0, 130.0))  # post azimuth window (mirrored across the row)
    post_sep_deg: float = tunable(25.0)  # min azimuth separation, own post vs every other post
    init_sep_deg: float = tunable(25.0)  # arrow spawn heading at least this far from its target

    # --- tunable: placement ------------------------------------------------------------------
    pin_nominal: tuple = tunable(((-0.28, 0.0), (0.0, 0.0), (0.28, 0.0)))  # RED, GREEN, BLUE
    post_ring_r: float = tunable(0.19)  # post distance from its pin (m)

    # --- info: structure ---------------------------------------------------------------------
    bench_size: tuple = info((0.96, 0.48, 0.10))  # kinematic bench slab; top = surface
    bench_top: float = info(0.10)
    pin_r: float = info(0.009)  # 18 mm pivot pin in the 28 mm hole (5 mm radial play)
    pin_h: float = info(0.055)
    hole_half: float = info(0.014)  # square pivot hole half-width in the arrow
    eye_out: float = info(0.056)  # pivot-hole frame outer square
    bar_t: float = info(0.014)  # arrow thickness (flat underside)
    shaft_w: float = info(0.024)
    head_base_x: float = info(0.082)  # shaft ends where the arrowhead begins
    tip_x: float = info(0.110)  # arrowhead TIP distance from the pivot (the heading arm)
    head_len: float = info(0.055)
    head_w: float = info(0.012)
    head_ang_deg: float = info(30.0)
    tail_len: float = info(0.024)  # short blunt tail: pivot is visibly off-centre
    arrow_mass: float = info(0.10)
    sweep_r: float = info(0.116)  # max radius any arrow point reaches from its pin
    post_r: float = info(0.015)  # 30 mm target post
    post_h: float = info(0.120)
    contact_offset: float = info(0.001)  # default ~2 cm would eat the 5 mm pivot play
    bench_color: tuple = info((0.75, 0.75, 0.78))
    pin_color: tuple = info((0.25, 0.25, 0.28))
    dial_colors: tuple = info((
        ("red", (0.85, 0.15, 0.12), (0.55, 0.08, 0.06)),
        ("green", (0.10, 0.60, 0.25), (0.05, 0.38, 0.15)),
        ("blue", (0.25, 0.45, 0.90), (0.15, 0.28, 0.60)),
    ))
    decoy_color: tuple = info((0.95, 0.85, 0.10))

    # Derived (filled in __post_init__).
    n_dials: int = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.n_dials = len(self.dial_colors)
        assert self.hole_half - self.pin_r >= 0.004, "arrow must spin on its pin with real play"
        assert self.post_ring_r >= self.sweep_r + self.post_r + 0.03, (
            "posts must stand beyond the arrow's sweep — an arrow can never touch a post")
        # adjacent sweeps can never clash, even with worst-case jitter toward each other
        min_gap = min(
            abs(self.pin_nominal[i + 1][0] - self.pin_nominal[i][0])
            for i in range(self.n_dials - 1)) - 2 * self.pin_jitter
        assert min_gap >= 2 * self.sweep_r + 0.01, "adjacent arrow sweeps must be disjoint"
        # a post can never intrude into a NEIGHBOUR's sweep (worst case: azimuth at the
        # window edge closest to the row axis, jitter closing the gap)
        lo = math.radians(self.post_az_window_deg[0])
        worst = math.hypot(min_gap - self.post_ring_r * math.cos(lo),
                           self.post_ring_r * math.sin(lo))
        assert worst >= self.sweep_r + self.post_r + 0.01, (
            "a post at the window edge must stay clear of the neighbouring sweep")
        assert self.init_sep_deg >= self.align_tol_deg + 10.0, (
            "spawn headings must start well outside the alignment tolerance")
        assert self.post_sep_deg >= 2 * self.align_tol_deg + 5.0, (
            "aiming at a wrong post must be unambiguously outside the tolerance")
        # posts and sweeps stay on the bench
        span_x = max(abs(p[0]) for p in self.pin_nominal) + self.pin_jitter
        assert span_x + self.post_ring_r * math.cos(lo) + self.post_r < self.bench_size[0] / 2
        assert self.post_ring_r + self.post_r < self.bench_size[1] / 2
        assert self.tip_x > self.eye_out / 2 + 0.02, "the tip arm dominates: heading is the tip end"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("compass_dials")
class CompassDialsScene(BaseScene):
    cfg: CompassDialsSceneCfg

    def __init__(self, cfg: CompassDialsSceneCfg | None = None) -> None:
        super().__init__(cfg or CompassDialsSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.bench_top
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
            "bench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=c.bench_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.bench_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - c.bench_size[2] / 2)),
            ),
        }
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        col = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        for k, (name, color, hcolor) in enumerate(c.dial_colors):
            px, py = c.pin_nominal[k]
            out[f"pin_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pin_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.pin_r, height=c.pin_h, axis="Z",
                    rigid_props=kin, collision_props=col,
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pin_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, z0 + c.pin_h / 2)),
            )
            out[f"post_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Post_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.post_r, height=c.post_h, axis="Z",
                    rigid_props=kin, collision_props=col,
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py + c.post_ring_r, z0 + c.post_h / 2)),
            )
            out[f"arrow_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Arrow_" + name,
                spawn=_arrow_spawner_cfg(
                    key=name, hole_half=c.hole_half, eye_out=c.eye_out, bar_t=c.bar_t,
                    shaft_w=c.shaft_w, head_base_x=c.head_base_x, tip_x=c.tip_x,
                    head_len=c.head_len, head_w=c.head_w, head_ang_deg=c.head_ang_deg,
                    tail_len=c.tail_len, mass=c.arrow_mass, color=color,
                    head_color=hcolor, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, z0 + c.bar_t / 2 + 0.002)),
            )
        out["post_decoy"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Post_decoy",
            spawn=sim_utils.CylinderCfg(
                radius=c.post_r, height=c.post_h, axis="Z",
                rigid_props=kin, collision_props=col,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.decoy_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -c.post_ring_r, z0 + c.post_h / 2)),
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
        names = [nm for nm, _c, _h in c.dial_colors]
        self.names = names
        self.pins: list[RigidObject] = [env.iscene[f"pin_{nm}"] for nm in names]
        self.posts: list[RigidObject] = [env.iscene[f"post_{nm}"] for nm in names]
        self.arrows: list[RigidObject] = [env.iscene[f"arrow_{nm}"] for nm in names]
        self.decoy: RigidObject = env.iscene["post_decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        d = c.n_dials
        self.target_az = torch.zeros(n, d, device=dev)  # pin->own-post azimuth (world)
        self.err0 = torch.full((n, d), math.pi, device=dev)  # spawn heading error
        self.prog_latch = torch.zeros(n, d, device=dev)
        self.align_latch = torch.zeros(n, d, device=dev)

    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(a), torch.cos(a))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter each pin, sample each post's azimuth in the window
        (mirrored across the row, rejection-resampled for `post_sep_deg` separation
        from every other post as seen from every pin), plant the decoy off the middle
        pin, spawn each arrow ON its pin at a heading forced >= `init_sep_deg` from
        its target; latches zeroed, baselines captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        d = c.n_dials
        origin = self.env_origins[env_ids]
        z0 = c.bench_top

        # --- pins: nominal + jitter ---
        pin_xy = torch.tensor(c.pin_nominal, device=dev).unsqueeze(0).expand(m, d, 2).clone()
        pin_xy += (torch.rand(m, d, 2, device=dev) * 2 - 1) * c.pin_jitter

        # --- post + decoy azimuths: window sample + whole-layout rejection ---
        lo, hi = (math.radians(v) for v in c.post_az_window_deg)
        sep = math.radians(c.post_sep_deg)

        def sample_az(rows: int) -> torch.Tensor:
            az = lo + torch.rand(rows, d + 1, device=dev) * (hi - lo)
            flip = torch.rand(rows, d + 1, device=dev) < 0.5
            return torch.where(flip, az + math.pi, az)

        az = sample_az(m)  # (m, d+1): three posts + decoy (decoy anchored to middle pin)
        anchor = torch.cat([pin_xy, pin_xy[:, 1:2, :]], dim=1)  # (m, d+1, 2)
        for _ in range(40):
            post_xy = anchor + c.post_ring_r * torch.stack(
                [torch.cos(az), torch.sin(az)], dim=-1)
            # from every pin k: azimuth to every post j; own-vs-other separation
            rel = post_xy.unsqueeze(1) - pin_xy.unsqueeze(2)  # (m, d, d+1, 2)
            az_seen = torch.atan2(rel[..., 1], rel[..., 0])  # (m, d, d+1)
            own = az[:, :d].unsqueeze(2)  # target azimuth of pin k (== az_seen[:,k,k])
            diff = self._wrap(az_seen - own).abs()  # (m, d, d+1)
            mask = torch.eye(d, d + 1, device=dev, dtype=torch.bool).unsqueeze(0)
            bad = ((diff < sep) & ~mask).reshape(m, -1).any(dim=1)
            if not bad.any():
                break
            az[bad] = sample_az(int(bad.sum()))
        post_xy = anchor + c.post_ring_r * torch.stack([torch.cos(az), torch.sin(az)], dim=-1)

        # --- arrow spawn headings: forced away from the target ---
        isep = math.radians(c.init_sep_deg)
        yaw = (torch.rand(m, d, device=dev) * 2 - 1) * math.pi
        for _ in range(40):
            bad = self._wrap(yaw - az[:, :d]).abs() < isep
            if not bad.any():
                break
            yaw[bad] = (torch.rand(int(bad.sum()), device=dev) * 2 - 1) * math.pi

        # --- write states ---
        def write(body, xy: torch.Tensor, z: float, yaw_1d: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            if yaw_1d is None:
                st[:, 3] = 1.0
            else:
                st[:, 3] = torch.cos(yaw_1d / 2)
                st[:, 6] = torch.sin(yaw_1d / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        for k in range(d):
            write(self.pins[k], pin_xy[:, k], z0 + c.pin_h / 2)
            write(self.posts[k], post_xy[:, k], z0 + c.post_h / 2)
            write(self.arrows[k], pin_xy[:, k], z0 + c.bar_t / 2 + 0.002, yaw[:, k])
        write(self.decoy, post_xy[:, d], z0 + c.post_h / 2)

        # --- baselines + latches ---
        self.target_az[env_ids] = az[:, :d]
        self.err0[env_ids] = self._wrap(yaw - az[:, :d]).abs().clamp(min=isep * 0.9)
        self.prog_latch[env_ids] = 0.0
        self.align_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pins": [b.data.root_state_w[env_ids].clone() for b in self.pins],
            "posts": [b.data.root_state_w[env_ids].clone() for b in self.posts],
            "arrows": [b.data.root_state_w[env_ids].clone() for b in self.arrows],
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "target_az": self.target_az[env_ids].clone(),
            "err0": self.err0[env_ids].clone(),
            "prog_latch": self.prog_latch[env_ids].clone(),
            "align_latch": self.align_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.pins, state["pins"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.posts, state["posts"]):
            b.write_root_state_to_sim(st, env_ids)
        for b, st in zip(self.arrows, state["arrows"]):
            b.write_root_state_to_sim(st, env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.target_az[env_ids] = state["target_az"]
        self.err0[env_ids] = state["err0"]
        self.prog_latch[env_ids] = state["prog_latch"]
        self.align_latch[env_ids] = state["align_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"Three compass dials stand in a row on a light-gray bench. Each dial is a flat "
            f"ARROW-shaped bar ({c.bar_t * 1000:.0f} mm thick, pointed arrowhead at its long "
            f"end, short blunt tail at the other) resting flat on the bench, held captive by "
            f"a dark steel PIVOT PIN ({2 * c.pin_r * 1000:.0f} mm thick, "
            f"{c.pin_h * 1000:.0f} mm tall) that passes up through a square hole in the bar: "
            f"the arrow can spin freely about its pin like a compass needle but cannot slide "
            f"away. The arrows are colored RED, GREEN and BLUE. Standing upright on the bench "
            f"around the dials are four cylindrical POSTS ({c.post_h * 1000:.0f} mm tall): one "
            f"RED, one GREEN, one BLUE — each placed {c.post_ring_r * 100:.0f} cm from the "
            f"same-colored arrow's pin, at a different direction every episode — and one "
            f"YELLOW post that matches no arrow (a decoy).\n"
            f"Goal: set all three dials — swing each arrow around its pin until its pointed "
            f"ARROWHEAD TIP aims at the post of the SAME color, within about "
            f"{c.align_tol_deg:.0f} degrees. The posts stand beyond the arrows' reach, so an "
            f"arrow never touches its post; push the arrow's shaft, head or tail sideways to "
            f"spin it about the pin. Any dial order works. Judged when everything is still: "
            f"every arrow must end resting FLAT on the bench with its pin still through its "
            f"hole. An arrow aimed at a wrong-colored post or at the yellow decoy, an arrow "
            f"pointing with its TAIL instead of its arrowhead, and an arrow lifted off its "
            f"pin (wherever it is put down) all count for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Spin each colored arrow around its pivot pin until its arrowhead tip points at "
            "the post of the same color. Keep every arrow flat on the bench with the pin "
            "through its hole, and ignore the yellow post."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _stack(self, attr: str) -> torch.Tensor:
        """(N, D, ...) stacked arrow tensors in dial order."""
        return torch.stack([getattr(b.data, attr) for b in self.arrows], dim=1)

    def headings(self) -> torch.Tensor:
        """(N, D) world yaw of each arrow's POINTING axis (local +x, through the tip)."""
        from isaaclab.utils.math import quat_apply

        quat = self._stack("root_quat_w")  # (N, D, 4)
        n, d = quat.shape[0], quat.shape[1]
        ex = torch.tensor([1.0, 0.0, 0.0], device=quat.device).expand(n * d, 3)
        v = quat_apply(quat.reshape(n * d, 4), ex).reshape(n, d, 3)
        return torch.atan2(v[:, :, 1], v[:, :, 0])

    def heading_err(self) -> torch.Tensor:
        """(N, D) |wrapped| angle between each arrow's tip heading and its target."""
        return self._wrap(self.headings() - self.target_az).abs()

    def on_pin(self) -> torch.Tensor:
        """(N, D) bool: hole centre within `on_pin_xy_tol` of the pin axis, resting in
        the on-bench height band, flat (arrow +z near world-up)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        pos = self._stack("root_pos_w")  # (N, D, 3)
        pin = torch.stack([b.data.root_pos_w for b in self.pins], dim=1)
        near = (pos[:, :, :2] - pin[:, :, :2]).norm(dim=-1) < c.on_pin_xy_tol
        z_rel = pos[:, :, 2] - self.env_origins[:, None, 2] - c.bench_top
        in_band = (z_rel > 0.001) & (z_rel < c.bar_t / 2 + 0.010)
        quat = self._stack("root_quat_w")
        n, d = quat.shape[0], quat.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(n * d, 3)
        up = quat_apply(quat.reshape(n * d, 4), ez).reshape(n, d, 3)
        flat = up[:, :, 2] >= math.cos(math.radians(c.flat_max_deg))
        return near & in_band & flat

    def still(self) -> torch.Tensor:
        """(N, D) bool: per-arrow lin AND ang velocity below the settle thresholds."""
        lin = self._stack("root_lin_vel_w").norm(dim=-1)
        ang = self._stack("root_ang_vel_w").norm(dim=-1)
        return (lin < self.cfg.settle_lin) & (ang < self.cfg.settle_ang)

    # ----- predicates -------------------------------------------------------------------------
    def aligned(self) -> torch.Tensor:
        """(N, D) bool: tip heading within `align_tol_deg` of the pin->own-post
        azimuth AND the arrow still seated on its pin."""
        tol = math.radians(self.cfg.align_tol_deg)
        return (self.heading_err() <= tol) & self.on_pin()

    def settled(self) -> torch.Tensor:
        """(N,) bool: every arrow still."""
        return self.still().all(dim=1)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, per dial: best fractional heading progress toward the target
        (on-pin gated — an arrow carried around OFF its pin earns nothing) and
        ever-aligned-and-still. Transient progress keeps its credit."""
        on = self.on_pin()
        prog = ((self.err0 - self.heading_err()) / self.err0).clamp(0.0, 1.0) * on.float()
        self.prog_latch = torch.maximum(self.prog_latch, prog)
        ok_now = (self.aligned() & self.still()).float()
        self.align_latch = torch.maximum(self.align_latch, ok_now)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three dials simultaneously aligned on their pins, everything
        still."""
        return self.aligned().all(dim=1) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: per dial 0.10 * latched heading progress + 0.20 *
        latched aligned-and-still, capped at 0.90; exactly 1.0 iff success(). Doing
        nothing scores ~0 (spawn headings start >= `init_sep_deg` off target); the
        seed's strategy (slide the bar off and carry it somewhere) earns nothing —
        off-pin arrows are gated out of every credit term."""
        base = (0.10 * self.prog_latch + 0.20 * self.align_latch).sum(dim=1).clamp(0.0, 0.90)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="compass_dials", robot="null"))
