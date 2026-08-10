"""PouringScene — metered split-pour of pellets from a cup into two bowls (port).

The object world for the dexmimicgen `TwoArmPouring` port: an open cup filled with a
sampled number of pellets stands on the bench between two pad-seated bowls. **Goal
(carried here, no task layer): pour — never touch a pellet directly — until the MAJOR
bowl holds the episode's sampled target count (within a small band) and the other bowl
holds the rest, with at most a couple of pellets lost to the table, then park the empty
cup upright on the bench.** The pour angle is the flow valve: pellets start streaming
above one tilt and stop below a lower one (hysteresis), and a pellet that comes to rest
outside a vessel STAYS there — counts are irreversible at fingertip scale.

PORT FIDELITY (dexmimicgen `environments/two_arm_pouring.py`:
  - Ported FAITHFULLY: cup -> bowl pour of free ball(s), containment judged by
    ball-in-bowl + bowl seated on a fixed pad + bowl upright (their contact + XY<0.1 +
    pad XY<0.06 + 1-Rzz<0.05 checks become our vessel-frame containment + pad/upright
    seat gates), bimanual embodiment class (GR1 with dex hands = our gr1t2).
  - OUR LABELED EXTENSIONS (the source pours ONE ball into ONE bowl — short-horizon):
    the multi-pellet load (~24 spheres, the granular stand-in for liquid — no liquids
    by suite rule), the SECOND bowl, and the metered SPLIT goal with a sampled per-
    episode target — this is what earns the long-horizon slot and forces a mid-pour
    stop at the hardest point of the hysteresis curve. `goal="pour_all"` keeps the
    faithful v0 (everything into one bowl) as the curriculum floor.
  - Assets re-authored procedurally (the objaverse cup/bowl meshes are CoACD piles;
    our compound-vessel spawner gives the same open cavity from primitive colliders).

Everything task-relevant is VISIBLE in-scene (presentation principle): the pellets are
bright orange against blue/green bowls, so per-bowl counts, spills, and the pour stream
read directly off the video; the target split is stated in `describe()` (nothing is
hidden from the agent).

Bodies are plain rigid objects — no springs, no forced bodies — so `post_step` here is
pure bookkeeping: region counts, the latched spill/tunneling masks, stream (pour-event)
edges, the carry-tilt metric, and the latched stage flags. The counting core is pure
tensor functions (`_classify_regions`, `_stream_update`, `_prefix_score`) — unit-
testable app-free, the microwave `_machine_step` pattern.

Per-episode randomization (task-family knobs): active pellet count, WHICH bowl is the
major target, the target split fraction, cup pose (xy jitter + yaw), bowl jitter —
judged against the sampled episode.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- pure-tensor counting core (unit-testable app-free) -----------------------------------------
def _qrot_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors `v` (..., 3) into the body frame of unit quats `q`
    (..., 4, wxyz). Broadcasts (e.g. q (N, 1, 4) against v (N, P, 3))."""
    w = q[..., :1]
    u = -q[..., 1:]  # conjugate = inverse rotation for unit quats
    c = torch.linalg.cross(u.expand_as(v), v, dim=-1)
    return v + 2.0 * (w * c + torch.linalg.cross(u.expand_as(v), c, dim=-1))


def _tilt_deg(q: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between a body's +z axis and world +z, from quats (..., 4, wxyz):
    R22 = 1 - 2(x^2 + y^2)."""
    r22 = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return torch.rad2deg(torch.acos(r22.clamp(-1.0, 1.0)))


def _in_vessel(p_w: torch.Tensor, v_pos: torch.Tensor, v_quat: torch.Tensor,
               inner_r: float, z_lo: float, z_hi: float) -> torch.Tensor:
    """Bool (N, P): pellet centres inside an open cylindrical vessel, judged in the
    VESSEL'S BODY FRAME (the pen-holder holder-frame lesson — a tilted cup judges like a
    standing one). `z_lo`/`z_hi` bound the local z band (root at the body centre).
    The callers pass the FULL inner radius and a z_hi one pellet radius above the rim
    plane (GPU round 2: a tighter radius left wall-hugging pellets deadlocked as
    "air", and a rim-plane z_hi latched pellets pooled at a paused, tilted cup's lip
    as spilled while still in the cup); a pellet balanced ON the rim top is still
    excluded by radius (the wall's mid-face sits outside inner_r)."""
    local = _qrot_inv(v_quat.unsqueeze(1), p_w - v_pos.unsqueeze(1))
    in_r = local[..., 0] ** 2 + local[..., 1] ** 2 < inner_r ** 2
    return in_r & (local[..., 2] > z_lo) & (local[..., 2] < z_hi)


def _classify_regions(p_w: torch.Tensor,
                      cup_pos: torch.Tensor, cup_quat: torch.Tensor,
                      bowl_pos: torch.Tensor, bowl_quat: torch.Tensor,
                      cup_inner_r: float, cup_z: tuple[float, float],
                      bowl_inner_r: float, bowl_z: tuple[float, float]) -> torch.Tensor:
    """Region code per pellet (N, P) long: 0 = in cup, 1 = in bowl 0, 2 = in bowl 1,
    3 = elsewhere (air / table). `bowl_pos`/`bowl_quat` are (N, 2, 3)/(N, 2, 4)."""
    region = torch.full(p_w.shape[:2], 3, dtype=torch.long, device=p_w.device)
    for b in (1, 0):  # cup wins ties (write it last)
        inside = _in_vessel(p_w, bowl_pos[:, b], bowl_quat[:, b],
                            bowl_inner_r, bowl_z[0], bowl_z[1])
        region = torch.where(inside, torch.full_like(region, b + 1), region)
    in_cup = _in_vessel(p_w, cup_pos, cup_quat, cup_inner_r, cup_z[0], cup_z[1])
    return torch.where(in_cup, torch.zeros_like(region), region)


def _stream_update(in_cup: torch.Tensor, was_in_cup: torch.Tensor,
                   since_out: torch.Tensor, was_active: torch.Tensor,
                   gap: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One substep of pour-stream detection. An OUTFLOW is a pellet leaving the cup
    volume; the stream is ACTIVE while the last outflow is under `gap` substeps old,
    and each idle->active transition is one pour event (start/stop cycles — the mid-
    pour stop the split forces is exactly one extra event).
    Returns (since_out, stream_active, new_event)."""
    outflow = (was_in_cup & ~in_cup).any(dim=1)
    since_out = torch.where(outflow, torch.zeros_like(since_out),
                            (since_out + 1).clamp(max=gap + 1))
    stream_active = since_out <= gap
    return since_out, stream_active, stream_active & ~was_active


def _prefix_score(flags: torch.Tensor, chain: tuple[int, ...]) -> torch.Tensor:
    """(N,) int 0-100: longest latched PREFIX of the goal's stage chain (the spatula rubric
    — parking an unpoured cup earns nothing)."""
    sel = flags[:, list(chain)].long()
    done = sel.cumprod(dim=1).sum(dim=1)
    return done * 100 // len(chain)


# ----- compound vessel spawner (the shared compound-spawner pattern) ------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_vessel(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one open cylindrical vessel (cup or bowl) at `prim_path`: root Xform with
    RigidBodyAPI + explicit MassAPI, a bottom disc collider + n short box wall segments
    (child colliders of one body never self-collide) — an open cavity from primitives,
    no mesh decomposition needed."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    # The pellets live in SUSTAINED contact with a kinematically-driven cup (the spatula
    # lesson 10 regime, not the pen-holder impact-pop one): a low depenetration cap would let
    # a lift out-run the position correction and sink the load into the floor disc.
    pxrb.CreateMaxDepenetrationVelocityAttr(1.0)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSolverPositionIterationCountAttr(8)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = cfg.n_segments
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _vessel_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, color: tuple, n_segments: int,
                        contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vessel" not in _SPAWNER_CACHE:

        @configclass
        class VesselSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vessel)
            inner_r: float = 0.034
            wall_t: float = 0.005
            height: float = 0.11
            bot_t: float = 0.008
            color: tuple = (0.8, 0.8, 0.82)
            n_segments: int = 10
            contact_offset: float = 0.002

        _SPAWNER_CACHE["vessel"] = VesselSpawnerCfg

    return _SPAWNER_CACHE["vessel"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t,
        color=color, n_segments=n_segments, contact_offset=contact_offset,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class PouringSceneCfg(BaseCfg):
    """Config for `PouringScene`."""

    # --- goal + judging (difficulty dials) ------------------------------------------------------
    goal: str = "split"  # "split" (v1 metered, default) | "pour_all" (faithful v0)
    n_lo: int = 18  # sampled active pellet count, inclusive bounds
    n_hi: int = 24
    frac_lo: float = 0.55  # sampled major-bowl target fraction band
    frac_hi: float = 0.72  # (~2/3 split; brief's [0.58, 0.75] = band +/- tol)
    count_tol: int = 2  # |major count - target| <= this
    spill_max: int = 2  # pellets at rest outside all vessels allowed
    spill_grace: int = 60  # displacement-window length (substeps) for the spill latch
    spill_disp: float = 0.006  # moved less than this over a window = at rest (m)
    pour_all_frac: float = 0.90  # v0 gate: >= this fraction in the major bowl
    settle_speed: float = 0.05  # |v| below this = settled (judging gate)
    # A pellet in a bowl counts below this |v|: well under falling-transit speed
    # (~1.3 m/s at the bowl) but ABOVE contact-solver buzz on a 5 g sphere (GPU
    # round 2: a wedged pellet jittering at ~0.2 m/s never counted anywhere).
    count_speed: float = 0.30
    park_tilt_deg: float = 15.0  # cup upright gate when parked
    seat_tilt_deg: float = 18.0  # bowl upright gate (source: 1 - Rzz < 0.05)
    pad_xy_tol: float = 0.06  # bowl centre within this of its pad (source 0.06)
    stream_gap: int = 45  # substeps without an outflow that end a stream
    lift_h: float = 0.04  # cup bottom above surface by this = lifted

    # --- randomization (the task-family knobs) --------------------------------------------------
    reset_pos_jitter: float = 0.03  # uniform +/- xy jitter on the cup at reset
    reset_yaw_deg: float = 180.0  # uniform +/- yaw on the cup at reset
    bowl_jitter: float = 0.012  # uniform +/- xy jitter per bowl (stays on pad)

    # --- placement + embodiment sizing ----------------------------------------------------------
    surface_z: float = 0.0  # work-surface height; 0 = on the ground (null smoke)
    cup_pos: tuple = (0.0, -0.18)  # cup spawn centre on the surface
    bowl_slots: tuple = ((-0.18, 0.12), (0.18, 0.12))  # pad centres (bowl 0, 1)
    park_pos: tuple = (-0.38, -0.20)  # suggested clear parking spot (describe only)
    reserve_pos: tuple = (0.0, 1.05)  # inactive-pellet grid, on the GROUND behind
    cup_inner_r: float = 0.034  # per-embodiment: outer dia 78 mm palms for dex
    cup_h: float = 0.11  # hands; the franka binding shrinks it under the 8 cm jaw

    # --- structure (applied masses + pellet friction) ------------------------------
    bench_size: tuple = (1.2, 0.9)
    cup_wall_t: float = 0.005
    cup_bot_t: float = 0.008
    cup_mass: float = 0.10
    cup_color: tuple = (0.82, 0.82, 0.85)
    # Source-faithful bowl size (objaverse bowl_7 x1.5 is ~19 cm across): GPU round 4
    # showed a 12 cm bowl sheds straggler splash over its rim; the wide basin is part
    # of what makes the source task feasible.
    bowl_inner_r: float = 0.075
    bowl_wall_t: float = 0.007  # rim width — the universal pinch-grasp affordance
    bowl_h: float = 0.055
    bowl_bot_t: float = 0.010
    bowl_mass: float = 0.25
    bowl_colors: tuple = ((0.25, 0.42, 0.72), (0.30, 0.60, 0.35))  # blue / green
    bowl_names: tuple = ("blue", "green")
    pad_size: tuple = (0.20, 0.20, 0.006)
    pad_colors: tuple = ((0.14, 0.22, 0.38), (0.16, 0.32, 0.20))  # darker shades
    n_segments: int = 12  # rounder polygon = shallower corner notches
    max_pellets: int = 24  # authored sphere count; reset() activates n_lo..n_hi
    pellet_r: float = 0.008
    pellet_mass: float = 0.005  # ~source ball scale (density-50 foam, grams)
    pellet_color: tuple = (0.93, 0.55, 0.15)  # bright orange on blue/green — countable
    pellet_friction: tuple = (0.40, 0.35)  # static, dynamic (restitution 0: no popcorn)
    contact_offset: float = 0.002  # vessels (explicit small offsets everywhere)
    pellet_contact_offset: float = 0.0015
    n_flags: int = 7

    # Stage-flag slots (latched) + the two goal chains over them.
    FLAGS = ("lifted", "stream_major", "major_band", "stream_minor",
             "distributed", "parked", "poured_all")
    CHAINS = {"split": (0, 1, 2, 3, 4, 5), "pour_all": (0, 1, 6, 5)}

    # Derived (filled in __post_init__).
    cup_outer_r: float = field(default=None, init=False)
    bowl_outer_r: float = field(default=None, init=False)
    cup_z_band: tuple = field(default=None, init=False)  # local z band "inside the cup"
    bowl_z_band: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.goal not in self.CHAINS:
            raise ValueError(f"goal must be one of {tuple(self.CHAINS)}, got {self.goal!r}")
        self.cup_outer_r = round(self.cup_inner_r + self.cup_wall_t, 4)
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)
        # Inside = above the floor disc, centre up to one pellet radius above the rim
        # plane (pellets pooled at a tilted cup's lip, or riding a pile against a
        # bowl wall, must still read as inside — GPU round 2); rim-top-balanced
        # pellets stay excluded by the RADIUS gate, not the z one.
        self.cup_z_band = (round(-self.cup_h / 2 + self.cup_bot_t, 4),
                           round(self.cup_h / 2 + self.pellet_r, 4))
        self.bowl_z_band = (round(-self.bowl_h / 2 + self.bowl_bot_t, 4),
                            round(self.bowl_h / 2 + self.pellet_r, 4))


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("pouring")
class PouringScene(BaseScene):
    cfg: PouringSceneCfg

    def __init__(self, cfg: PouringSceneCfg | None = None) -> None:
        super().__init__(cfg or PouringSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                 rest_offset=0.0)

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
        if z0 > 0:
            out["bench"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0 / 2)),
            )

        # Two kinematic pads (the source's fixed bowl pads) + one free bowl seated on each.
        for b in range(2):
            px, py = c.bowl_slots[b]
            out[f"pad_{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad_" + str(b),
                spawn=sim_utils.CuboidCfg(
                    size=c.pad_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_colors[b]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, z0 + c.pad_size[2] / 2)),
            )
            out[f"bowl_{b}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl_" + str(b),
                spawn=_vessel_spawner_cfg(
                    inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
                    bot_t=c.bowl_bot_t, mass=c.bowl_mass, color=c.bowl_colors[b],
                    n_segments=c.n_segments, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(px, py, z0 + c.pad_size[2] + c.bowl_h / 2 + 0.002)),
            )

        out["cup"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            spawn=_vessel_spawner_cfg(
                inner_r=c.cup_inner_r, wall_t=c.cup_wall_t, height=c.cup_h,
                bot_t=c.cup_bot_t, mass=c.cup_mass, color=c.cup_color,
                n_segments=c.n_segments, contact_offset=c.contact_offset,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.cup_pos[0], c.cup_pos[1], z0 + c.cup_h / 2 + 0.002)),
        )

        # Pellets: plain spheres, explicit mass, tight offsets, zero restitution and
        # moderate friction (controllable flow, not popcorn), depenetration cap 1.0 +
        # extra position iterations (sustained contact with a driven cup — spatula lesson).
        for i in range(c.max_pellets):
            gx, gy = self._reserve_slot(i)
            out[f"pellet_{i:02d}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pellet_" + f"{i:02d}",
                spawn=sim_utils.SphereCfg(
                    radius=c.pellet_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                        # Heavy ANGULAR damping = rolling resistance: PhysX spheres
                        # have no rolling friction, and a pellet that hops out rolls
                        # forever (GPU round 3: escapees 5 m away, still 0.24 m/s).
                        # 2.0 stops a free roller in ~0.5 s; sliding flow down the
                        # tilted cup is friction-dominated and barely affected.
                        linear_damping=0.10, angular_damping=2.0,
                        solver_position_iteration_count=8,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pellet_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.pellet_contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.pellet_friction[0],
                        dynamic_friction=c.pellet_friction[1],
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pellet_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(gx, gy, c.pellet_r + 0.001)),
            )
        return out

    def _reserve_slot(self, i: int) -> tuple[float, float]:
        """Ground-level parking grid behind the bench for pellets not in the episode's
        sampled count (visible but clearly aside; excluded from every predicate)."""
        rx, ry = self.cfg.reserve_pos
        return (rx + (i % 6 - 2.5) * 0.05, ry + (i // 6) * 0.05)

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
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.cup: RigidObject = env.iscene["cup"]
        self.bowls: list[RigidObject] = [env.iscene["bowl_0"], env.iscene["bowl_1"]]
        self.pellets: list[RigidObject] = [env.iscene[f"pellet_{i:02d}"]
                                           for i in range(c.max_pellets)]
        self.env_origins = env.iscene.env_origins
        # Episode sample.
        self._active = torch.zeros(n, c.max_pellets, dtype=torch.bool, device=dev)
        self._n_active = torch.zeros(n, dtype=torch.long, device=dev)
        self._major = torch.zeros(n, dtype=torch.long, device=dev)  # 0 = blue, 1 = green
        self._target = torch.zeros(n, dtype=torch.long, device=dev)  # major-bowl count
        # Latched pellet masks + counters/metrics.
        self._spilled = torch.zeros(n, c.max_pellets, dtype=torch.bool, device=dev)
        self._ref_pos = torch.zeros(n, c.max_pellets, 3, device=dev)  # spill-window ref
        self._ref_age = torch.zeros(n, dtype=torch.long, device=dev)
        self._lost = torch.zeros(n, c.max_pellets, dtype=torch.bool, device=dev)
        self._was_in_cup = torch.zeros(n, c.max_pellets, dtype=torch.bool, device=dev)
        self._since_out = torch.full((n,), 10_000, dtype=torch.long, device=dev)
        self._stream_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._pour_events = torch.zeros(n, dtype=torch.long, device=dev)
        self._carry_tilt_max = torch.zeros(n, device=dev)
        self._flags = torch.zeros(n, c.n_flags, dtype=torch.bool, device=dev)
        # Per-substep cache (filled by post_step; queries read it).
        self._region = torch.full((n, c.max_pellets), 3, dtype=torch.long, device=dev)

    # ----- reset --------------------------------------------------------------------------------
    def _fill_slots_local(self, k: int) -> list[tuple[float, float, float]]:
        """Cup-local pellet stack for k pellets: layers of a 3-ring + centre (max
        separation at the cup's radius), z from the inner floor upward. 24 pellets of
        r=8 mm fill ~9 cm of the 10.2 cm cavity."""
        c = self.cfg
        rr = c.cup_inner_r - c.pellet_r - 0.002
        dz = 2 * c.pellet_r * 1.03
        z_base = -c.cup_h / 2 + c.cup_bot_t + c.pellet_r + 0.0005
        out = []
        for j in range(k):
            layer, m = divmod(j, 4)
            if m < 3:
                ang = 2 * math.pi * m / 3 + layer * 0.9
                out.append((rr * math.cos(ang), rr * math.sin(ang), z_base + layer * dz))
            else:
                out.append((0.0, 0.0, z_base + layer * dz + c.pellet_r * 0.4))
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        """Bowls seated on their pads (jittered), cup at its spot (jitter + yaw) filled
        with a freshly sampled pellet count; sampled major bowl + target split."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        z0 = c.surface_z

        for b, bowl in enumerate(self.bowls):
            px, py = c.bowl_slots[b]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = px
            st[:, 1] = py
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bowl_jitter
            st[:, 2] = z0 + c.pad_size[2] + c.bowl_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += self.env_origins[env_ids]
            bowl.write_root_state_to_sim(st, env_ids)

        cup = torch.zeros(m, 13, device=dev)
        cup[:, 0] = c.cup_pos[0]
        cup[:, 1] = c.cup_pos[1]
        cup[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        cup[:, 2] = z0 + c.cup_h / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.reset_yaw_deg) / 2
        cup[:, 3] = torch.cos(half)
        cup[:, 6] = torch.sin(half)
        cup[:, 0:3] += self.env_origins[env_ids]
        self.cup.write_root_state_to_sim(cup, env_ids)

        # Episode sample: count, major bowl, target (clamped so both bowls get >= 1).
        n_act = torch.randint(c.n_lo, c.n_hi + 1, (m,), device=dev)
        frac = c.frac_lo + torch.rand(m, device=dev) * (c.frac_hi - c.frac_lo)
        target = (frac * n_act).round().long().clamp(min=1)
        target = torch.minimum(target, n_act - 1)
        self._n_active[env_ids] = n_act
        self._target[env_ids] = target
        self._major[env_ids] = torch.randint(0, 2, (m,), device=dev)
        idx = torch.arange(c.max_pellets, device=dev)
        self._active[env_ids] = idx.unsqueeze(0) < n_act.unsqueeze(1)

        # Pellets: active ones stacked inside the (jittered, yawed) cup; the rest on
        # the reserve grid. Stack coordinates are cup-local -> rotate by the cup yaw.
        cup_xy = cup[:, 0:2]
        cw, cz = cup[:, 3], cup[:, 6]
        cup_cz = z0 + c.cup_h / 2 + 0.002
        slots = self._fill_slots_local(c.max_pellets)
        for i, pellet in enumerate(self.pellets):
            st = torch.zeros(m, 13, device=dev)
            lx, ly, lz = slots[i]
            # yaw-rotate the local xy (quat about z: cos/sin of half-angle -> full angle)
            cos_y = cw * cw - cz * cz
            sin_y = 2 * cw * cz
            st[:, 0] = cup_xy[:, 0] - self.env_origins[env_ids, 0] \
                + lx * cos_y - ly * sin_y
            st[:, 1] = cup_xy[:, 1] - self.env_origins[env_ids, 1] \
                + lx * sin_y + ly * cos_y
            st[:, 2] = cup_cz + lz
            st[:, 3] = 1.0
            gx, gy = self._reserve_slot(i)
            on_reserve = ~self._active[env_ids, i]
            st[:, 0] = torch.where(on_reserve, torch.full((m,), gx, device=dev), st[:, 0])
            st[:, 1] = torch.where(on_reserve, torch.full((m,), gy, device=dev), st[:, 1])
            st[:, 2] = torch.where(on_reserve,
                                   torch.full((m,), c.pellet_r + 0.001, device=dev),
                                   st[:, 2])
            st[:, 0:3] += self.env_origins[env_ids]
            pellet.write_root_state_to_sim(st, env_ids)

        self._spilled[env_ids] = False
        self._ref_pos[env_ids] = 0.0  # first window sees a huge disp -> no false latch
        self._ref_age[env_ids] = 0
        self._lost[env_ids] = False
        self._was_in_cup[env_ids] = self._active[env_ids]
        self._since_out[env_ids] = 10_000
        self._stream_active[env_ids] = False
        self._pour_events[env_ids] = 0
        self._carry_tilt_max[env_ids] = 0.0
        self._flags[env_ids] = False
        self._region[env_ids] = 3

    # ----- geometry queries -----------------------------------------------------------------------
    def _pellet_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N, P, 3) world positions, (N, P) speeds)."""
        pos = torch.stack([p.data.root_pos_w for p in self.pellets], dim=1)
        vel = torch.stack([p.data.root_lin_vel_w.norm(dim=-1) for p in self.pellets], dim=1)
        return pos, vel

    def cup_tilt_deg(self) -> torch.Tensor:
        return _tilt_deg(self.cup.data.root_quat_w)

    def bowl_tilt_deg(self) -> torch.Tensor:
        return torch.stack([_tilt_deg(b.data.root_quat_w) for b in self.bowls], dim=1)

    def counts(self) -> dict[str, torch.Tensor]:
        """Live per-region counts (N,) — the dense signal the agent can query: in_cup,
        in each bowl (settled, spill-excluded), major/minor aliases, spilled, air."""
        _pos, vel = self._pellet_tensors()
        act = self._active
        slow = vel < self.cfg.count_speed
        in_cup = (self._region == 0) & act
        in_b = [(self._region == b + 1) & act & slow & ~self._spilled for b in range(2)]
        spilled = self._spilled & act
        air = act & ~in_cup & ~in_b[0] & ~in_b[1] & ~spilled
        in_bowls = torch.stack([v.sum(dim=1) for v in in_b], dim=1)
        maj = self._major.unsqueeze(1)
        return {
            "in_cup": in_cup.sum(dim=1),
            "in_bowl": in_bowls,
            "in_major": in_bowls.gather(1, maj).squeeze(1),
            "in_minor": in_bowls.gather(1, 1 - maj).squeeze(1),
            "spilled": spilled.sum(dim=1),
            "air": air.sum(dim=1),
            "lost": (self._lost & act).sum(dim=1),
        }

    def cup_parked(self) -> torch.Tensor:
        """(N,) bool: cup upright, at rest, bottom on the work surface, on the bench
        and clear of both bowls."""
        c = self.cfg
        p = self.cup.data.root_pos_w - self.env_origins
        upright = self.cup_tilt_deg() < c.park_tilt_deg
        bottom = p[:, 2] - c.cup_h / 2
        on_z = (bottom - c.surface_z).abs() < 0.015
        on_bench = (p[:, 0].abs() < c.bench_size[0] / 2) \
            & (p[:, 1].abs() < c.bench_size[1] / 2)
        clear = torch.ones_like(upright)
        for b in range(2):
            bx, by = c.bowl_slots[b]
            d = (p[:, :2] - torch.tensor([bx, by], device=p.device)).norm(dim=-1)
            clear &= d > c.bowl_outer_r + c.cup_outer_r
        still = self.cup.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return upright & on_z & on_bench & clear & still

    def bowls_seated(self) -> torch.Tensor:
        """(N, 2) bool: each bowl upright, at rest, centred on its pad (the source's
        bowl-on-pad + upright success predicates)."""
        c = self.cfg
        out = []
        for b, bowl in enumerate(self.bowls):
            p = bowl.data.root_pos_w - self.env_origins
            px, py = c.bowl_slots[b]
            on_pad = (p[:, :2] - torch.tensor([px, py], device=p.device)).norm(dim=-1) \
                < c.pad_xy_tol
            upright = _tilt_deg(bowl.data.root_quat_w) < c.seat_tilt_deg
            still = bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
            out.append(on_pad & upright & still)
        return torch.stack(out, dim=1)

    # ----- bookkeeping (every substep) ------------------------------------------------------------
    def post_step(self) -> None:
        c = self.cfg
        pos, vel = self._pellet_tensors()
        pos = pos - self.env_origins.unsqueeze(1)
        act = self._active

        bowl_pos = torch.stack([b.data.root_pos_w for b in self.bowls], dim=1) \
            - self.env_origins.unsqueeze(1)
        bowl_quat = torch.stack([b.data.root_quat_w for b in self.bowls], dim=1)
        cup_pos = self.cup.data.root_pos_w - self.env_origins
        cup_quat = self.cup.data.root_quat_w
        self._region = _classify_regions(
            pos, cup_pos, cup_quat, bowl_pos, bowl_quat,
            c.cup_inner_r, c.cup_z_band, c.bowl_inner_r, c.bowl_z_band)

        # Latched spill — DISPLACEMENT-WINDOW latch: a pellet that moved less than
        # `spill_disp` over a `spill_grace` window while outside every vessel is at
        # rest there (table, pad, ground, perched on a rim) and latches spilled.
        # Position windows, not velocity (GPU rounds 1-2: rim/corner pellets buzz at
        # solver-jitter speeds and never pass a velocity rest gate; a consecutive
        # counter resets on every flicker). Latency is 1-2 windows. Irreversible by
        # rule: a spilled pellet knocked into a bowl later still counts as spilled.
        self._ref_age += 1
        due = self._ref_age >= c.spill_grace
        if bool(due.any()):
            disp = (pos - self._ref_pos).norm(dim=-1)
            self._spilled |= due.unsqueeze(1) & (disp < c.spill_disp) \
                & (self._region == 3) & act
            self._ref_pos = torch.where(due.view(-1, 1, 1), pos, self._ref_pos)
            self._ref_age = torch.where(due, torch.zeros_like(self._ref_age),
                                        self._ref_age)
        off_bench = (pos[..., 0].abs() > c.bench_size[0] / 2) \
            | (pos[..., 1].abs() > c.bench_size[1] / 2)
        # Off the workspace footprint = spilled IMMEDIATELY, no rest needed: an
        # escapee can never legally return (no touching pellets), and waiting for it
        # to stop parks it in "air" — where it blocks success AND inflates any
        # committed-count feedback watching the pour (GPU round 3: escapees rolled
        # meters off the bench and never stopped — PhysX spheres have no rolling
        # friction).
        self._spilled |= off_bench & act
        # Tunneling detector (the feasibility spike's conservation check): through the
        # ground, or inside the bench footprint below its top.
        thru_ground = pos[..., 2] < -0.02
        if c.surface_z > 0:
            thru_ground = thru_ground | (~off_bench
                                         & (pos[..., 2] < c.surface_z - 2 * c.pellet_r))
        self._lost |= thru_ground & act

        # Pour-stream events (start/stop cycles — the metering metric).
        in_cup_now = (self._region == 0) & act
        self._since_out, self._stream_active, new_event = _stream_update(
            in_cup_now, self._was_in_cup & act, self._since_out,
            self._stream_active, c.stream_gap)
        self._pour_events += new_event.long()
        self._was_in_cup = in_cup_now

        # Carry-tilt margin: max cup tilt while carrying pellets AWAY from both bowls
        # (over a bowl, tilt is the pour valve, not a spill risk).
        cnt = self.counts()
        cup_bottom = cup_pos[:, 2] - c.cup_h / 2
        over = torch.zeros_like(self._stream_active)
        for b in range(2):
            d = (cup_pos[:, :2] - bowl_pos[:, b, :2]).norm(dim=-1)
            over |= d < c.bowl_inner_r + c.cup_outer_r
        carrying = (cnt["in_cup"] > 0) & (cup_bottom > c.surface_z + c.lift_h) & ~over
        tilt = self.cup_tilt_deg()
        self._carry_tilt_max = torch.where(
            carrying, torch.maximum(self._carry_tilt_max, tilt), self._carry_tilt_max)

        # Latched stage flags (both chains' slots always maintained; the goal knob only
        # selects which chain scores/succeeds — runtime-switchable, the spatula pattern).
        n_act = self._n_active
        band = (cnt["in_major"] - self._target).abs() <= c.count_tol
        settled_out = (cnt["air"] == 0) & (cnt["in_cup"] == 0)
        self._flags[:, 0] |= (cup_bottom > c.surface_z + c.lift_h) \
            & (cnt["in_cup"] * 2 >= n_act)
        self._flags[:, 1] |= cnt["in_major"] >= 1
        self._flags[:, 2] |= band
        self._flags[:, 3] |= cnt["in_minor"] >= 1
        self._flags[:, 4] |= settled_out & band & (cnt["spilled"] <= c.spill_max)
        self._flags[:, 5] |= self.cup_parked() & (cnt["in_cup"] == 0)
        need = (c.pour_all_frac * n_act).ceil().long()
        self._flags[:, 6] |= cnt["in_major"] >= need

    # ----- state (full, restorable) --------------------------------------------------------------
    def _bodies(self) -> dict[str, RigidObject]:
        d = {"cup": self.cup, "bowl_0": self.bowls[0], "bowl_1": self.bowls[1]}
        d.update({f"pellet_{i:02d}": p for i, p in enumerate(self.pellets)})
        return d

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("_active", "_n_active", "_major", "_target", "_spilled",
                               "_ref_pos", "_ref_age", "_lost", "_was_in_cup",
                               "_since_out", "_stream_active", "_pour_events",
                               "_carry_tilt_max", "_flags")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][n], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        bound = hasattr(self, "_target")
        n = int(self._n_active[0]) if bound else c.n_hi
        tgt = int(self._target[0]) if bound else round(0.66 * n)
        maj = int(self._major[0]) if bound else 0
        major_name, minor_name = c.bowl_names[maj], c.bowl_names[1 - maj]
        base = (
            f"An open cup ({2 * c.cup_outer_r * 100:.0f} cm across, {c.cup_h * 100:.0f} cm "
            f"tall) holding {n} orange pellets stands on the bench in front of two empty "
            f"bowls, each seated on its own matching pad: a blue bowl on the dark-blue pad "
            f"and a green bowl on the dark-green pad.\n"
            f"RULES: pellets may only be moved by POURING from the cup — never touch a "
            f"pellet with a hand or another object. Tilting the cup is the flow valve: "
            f"pellets start streaming above one angle and stop below a lower one, so a "
            f"stream in progress overshoots unless you back off early. A pellet that comes "
            f"to rest anywhere outside a vessel — table, pad, even balanced on a rim — is "
            f"SPILLED after a moment (leaving the bench entirely spills it instantly) and "
            f"stays spilled forever, even if later knocked into a bowl. Both bowls must "
            f"remain upright and seated on their pads.\n"
        )
        if c.goal == "split":
            goal = (
                f"Goal: pour so the {major_name.upper()} bowl ends up holding "
                f"{tgt} pellets (within +/-{c.count_tol}) and the {minor_name} bowl all "
                f"the rest, with at most {c.spill_max} pellets spilled and the cup emptied "
                f"and parked upright on a clear spot of the bench (e.g. near "
                f"({c.park_pos[0]:.2f}, {c.park_pos[1]:.2f})). Live per-bowl counts can "
                f"be queried at any time."
            )
        else:
            goal = (
                f"Goal: pour at least {c.pour_all_frac * 100:.0f}% of the pellets into "
                f"the {major_name.upper()} bowl, then park the emptied cup upright on a "
                f"clear spot of the bench."
            )
        return base + goal

    # ----- progress -------------------------------------------------------------------------------
    def stage_flags(self) -> torch.Tensor:
        return self._flags.clone()

    def pour_events(self) -> torch.Tensor:
        return self._pour_events.clone()

    def carry_tilt_max(self) -> torch.Tensor:
        return self._carry_tilt_max.clone()

    def score(self) -> torch.Tensor:
        """(N,) int 0-100: longest latched prefix of the active goal's stage chain."""
        return _prefix_score(self._flags, self.cfg.CHAINS[self.cfg.goal])

    def success(self) -> torch.Tensor:
        """(N,) bool, live (spill mask is the one latched input): the sampled split
        delivered (or the v0 pour-all gate), everything settled, cup parked upright,
        both bowls still seated on their pads."""
        c = self.cfg
        cnt = self.counts()
        seated = self.bowls_seated().all(dim=1)
        parked = self.cup_parked() & (cnt["in_cup"] == 0)
        if c.goal == "split":
            band = (cnt["in_major"] - self._target).abs() <= c.count_tol
            done = band & (cnt["air"] == 0) & (cnt["spilled"] <= c.spill_max)
        else:
            need = (c.pour_all_frac * self._n_active).ceil().long()
            done = (cnt["in_major"] >= need) & (cnt["air"] == 0)
        return done & parked & seated & (cnt["lost"] == 0)
