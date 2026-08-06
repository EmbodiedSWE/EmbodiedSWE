"""ScaffoldLiftScene — build a block scaffold so the red cube RESTS at the marked height.

Derived from rlbench/pick_and_lift but strategically inverted. The seed rewards
PREHENSILE TRANSPORT: identify the red target cube among colored distractor cubes,
grasp it, and hoist it to a floating target sphere — the cube is judged at the goal
point while held, and the distractors are pure clutter. Here the same cast of objects
(a red payload cube, blocks, a floating goal marker) is rearranged so that the seed's
plan is physically incapable of succeeding:

  - the goal is judged on a FREE-STANDING outcome: the red cube's center must REST
    inside a height band marked by a floating ring, directly above a round plinth,
    settled and persistently so (30 consecutive settled in-band steps). A held or
    hovering cube never satisfies it — release it and it simply falls;
  - nothing in the scene is tall enough to reach the band on its own, so the ONLY way
    up is CONSTRUCTION: the seed's distractor blocks are promoted to essential
    building material. Exactly one PAIR of the three scaffold blocks (heights 60 / 90 /
    120 mm) stacks to put the red cube's center in the band — the ring height is
    re-sampled per episode among the three pair sums, so the solver must read the
    marker height and choose the matching subset;
  - execution order is forced bottom-up: plinth -> first block -> second block ->
    payload on top.

Four stages: (1) read the ring height, select the correct block pair; (2) place the
first scaffold block on the plinth; (3) stack the second block on it; (4) set the red
cube on top and let it settle inside the band.

Rubric (graded, transients latched in post_step so momentary achievements are kept):
  0.00  nothing (null policy; blocks and payload spawn scattered on open ground)
  0.20  latched: a scaffold block placed ON the plinth (settled, on the plinth top)
  0.45  latched: a two-block scaffold over the plinth (a block settled, elevated)
  0.70  latched: the red cube elevated on structure over the plinth (any height)
  1.00  iff success(): red cube center inside the band (|z - band| < band_tol), over
        the plinth (xy within xy_tol), everything settled, and the state has PERSISTED
        for hold_steps consecutive substeps. Success is a physical outcome: a cube
        held at the ring, or a collapsed tower, keeps only latched partial credit.

Assets are fully procedural, one rigid body each: a KINEMATIC plinth (built-in
cylinder), a KINEMATIC visual-only ring marker (custom compound spawner, NO colliders
— the draw_triangle mat precedent — so it can be re-posed per episode yet never
touched), three dynamic scaffold cuboids and the dynamic red payload cube (built-in
cuboid spawners, high friction, zero restitution). Real per-episode randomization
(verified by readback in the smoke): plinth xy, ring height (3 discrete bands = the
required pair), block/payload scatter jitter + free yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- custom spawner: the visual-only ring marker ---------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the goal ring at `prim_path`: root Xform with KINEMATIC RigidBodyAPI +
    explicit mass, and 8 thin box segments forming an octagonal halo. NO CollisionAPI on
    any child — the ring is a pure visual marker (a held/toppling block passes through
    it) that can still be re-posed per episode via write_root_state_to_sim."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.05)

    color = Gf.Vec3f(*cfg.color)
    n = 8
    r_mid = cfg.ring_r
    seg_len = 2 * r_mid * math.tan(math.pi / n) + 0.004
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/seg_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.tube, seg_len, cfg.tube))
        seg.CreateDisplayColorAttr([color])
        # no CollisionAPI: visual only
    return root


def _ring_spawner_cfg(c: ScaffoldLiftSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "ring" not in _SPAWNER_CACHE:

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            ring_r: float = 0.13
            tube: float = 0.012
            color: tuple = (0.95, 0.80, 0.15)

        _SPAWNER_CACHE["ring"] = RingSpawnerCfg

    return _SPAWNER_CACHE["ring"](
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ring_r=c.ring_r, tube=c.ring_tube, color=c.ring_color,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ScaffoldLiftSceneCfg(BaseCfg):
    """Config for `ScaffoldLiftScene`. Band centers are derived from the block sizes in
    __post_init__; separations between every reachable wrong configuration and every
    band edge are >= 20 mm by construction (block edges 30 mm apart)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    band_tol: float = tunable(0.010)  # |red center z - band center| below this when judged
    xy_tol: float = tunable(0.07)  # red center within this of the plinth axis when judged
    settle_speed: float = tunable(0.05)  # max |lin vel| for 'settled' (m/s); < g*dt at 120 Hz
    hold_steps: int = tunable(30)  # consecutive settled-in-band substeps before success
    elev_margin: float = tunable(0.045)  # 'elevated' = this far above the body's on-plinth rest

    # --- tunable: randomization (task-family knobs) ------------------------------------------
    pad_pos: tuple = tunable((0.22, 0.0))  # plinth center (xy)
    pad_jitter: float = tunable(0.06)  # uniform +/- xy jitter of the plinth at reset
    scatter_jitter: float = tunable(0.02)  # uniform +/- xy jitter of blocks/payload at reset
    reset_yaw_deg: float = tunable(180.0)  # uniform +/- yaw per block at reset

    # --- info: structure ---------------------------------------------------------------------
    pad_r: float = info(0.10)  # plinth radius; the whole scaffold must stand on it
    pad_t: float = info(0.020)  # plinth height — thick enough that on-plinth vs on-ground
    # resting heights are 20 mm apart (unambiguous placement predicate, and a ground-built
    # tower misses every band by the same 20 mm)
    pad_color: tuple = info((0.20, 0.20, 0.24))
    scaffold_sizes: tuple = info((0.060, 0.090, 0.120))  # block edges; 30 mm apart so every
    # subset sum (singles 60/90/120, pairs 150/180/210, triple 270) is >= 30 mm from its
    # neighbors -> with band_tol 10 mm no wrong subset can land in a band
    scaffold_colors: tuple = info(((0.10, 0.70, 0.20), (0.92, 0.92, 0.92), (0.90, 0.75, 0.10)))
    scaffold_masses: tuple = info((0.12, 0.25, 0.40))
    red_size: float = info(0.050)  # the payload cube (the seed's pick target)
    red_mass: float = info(0.08)
    red_color: tuple = info((0.90, 0.08, 0.08))
    ring_r: float = info(0.13)  # halo radius > plinth radius: surrounds, never occludes
    ring_tube: float = info(0.012)
    ring_color: tuple = info((0.95, 0.80, 0.15))
    contact_offset: float = info(0.002)
    # scatter slots (x, y): payload + blocks staggered on the far side of the plinth, spaced
    # so worst-case jitter + yaw never overlaps neighbors (min center gap 0.206 m vs max
    # needed half-diagonal sum 0.184 m + 2*jitter*sqrt(2) = 0.057 m -> safe by design)
    slot_red: tuple = info((-0.17, -0.10))
    slot_blocks: tuple = info(((-0.17, 0.30), (-0.22, -0.30), (-0.23, 0.10)))
    pairs: tuple = info(((0, 1), (0, 2), (1, 2)))  # candidate scaffold pairs (block indices)

    # Derived (filled in __post_init__).
    band_centers: tuple = field(default=None, init=False)  # red CENTER z per pair, above ground

    def __post_init__(self) -> None:
        s = self.scaffold_sizes
        self.band_centers = tuple(
            round(self.pad_t + s[a] + s[b] + self.red_size / 2, 4) for a, b in self.pairs)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("scaffold_lift")
class ScaffoldLiftScene(BaseScene):
    cfg: ScaffoldLiftSceneCfg

    def __init__(self, cfg: ScaffoldLiftSceneCfg | None = None) -> None:
        super().__init__(cfg or ScaffoldLiftSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def cube_cfg(size: float, mass: float, rgb: tuple, path: str,
                     pos: tuple) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + path,
                spawn=sim_utils.CuboidCfg(
                    size=(size, size, size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(pos[0], pos[1], size / 2 + 0.002)),
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
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CylinderCfg(
                    radius=c.pad_r, height=c.pad_t, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.8, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
            "ring": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ring",
                spawn=_ring_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.pad_pos[0], c.pad_pos[1], c.band_centers[0])),
            ),
            "payload": cube_cfg(c.red_size, c.red_mass, c.red_color, "Payload", c.slot_red),
        }
        for i, size in enumerate(c.scaffold_sizes):
            out[f"block_{i}"] = cube_cfg(size, c.scaffold_masses[i], c.scaffold_colors[i],
                                         f"Block_{i}", c.slot_blocks[i])
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**21,
                "gpu_max_rigid_patch_count": 2**21,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.pad: RigidObject = env.iscene["pad"]
        self.ring: RigidObject = env.iscene["ring"]
        self.red: RigidObject = env.iscene["payload"]
        self.blocks: list[RigidObject] = [
            env.iscene[f"block_{i}"] for i in range(len(c.scaffold_sizes))]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self._sizes = torch.tensor(c.scaffold_sizes, device=dev)
        self._bands = torch.tensor(c.band_centers, device=dev)
        self.required_pair = torch.zeros(n, dtype=torch.long, device=dev)
        self.band_z = torch.full((n,), c.band_centers[0], device=dev)
        self._latch_base = torch.zeros(n, dtype=torch.bool, device=dev)
        self._latch_stack = torch.zeros(n, dtype=torch.bool, device=dev)
        self._latch_red = torch.zeros(n, dtype=torch.bool, device=dev)
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: plinth re-posed with xy jitter, required pair sampled -> ring
        teleported to the matching band height (kinematic, verified by readback), blocks
        and payload scattered at their slots with jitter + free yaw, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        yaw_amp = math.radians(c.reset_yaw_deg)

        # --- plinth ---
        pj = (torch.rand(m, 2, device=dev) * 2 - 1) * c.pad_jitter
        pad_st = torch.zeros(m, 13, device=dev)
        pad_st[:, 0] = c.pad_pos[0]
        pad_st[:, 1] = c.pad_pos[1]
        pad_st[:, :2] += pj
        pad_st[:, 2] = c.pad_t / 2
        pad_st[:, 3] = 1.0
        pad_st[:, 0:3] += origin
        self.pad.write_root_state_to_sim(pad_st, env_ids)

        # --- required pair + ring at the matching band height ---
        pair = torch.randint(0, len(c.pairs), (m,), device=dev)
        self.required_pair[env_ids] = pair
        self.band_z[env_ids] = self._bands[pair]
        ring_st = torch.zeros(m, 13, device=dev)
        ring_st[:, 0:2] = pad_st[:, 0:2]
        ring_st[:, 2] = origin[:, 2] + self._bands[pair]
        ring_st[:, 3] = 1.0
        self.ring.write_root_state_to_sim(ring_st, env_ids)

        # --- blocks + payload at their slots, jitter + free yaw ---
        def scatter(body, slot: tuple, size: float) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = slot[0]
            st[:, 1] = slot[1]
            st[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.scatter_jitter
            st[:, 2] = size / 2 + 0.002
            half = (torch.rand(m, device=dev) * 2 - 1) * yaw_amp / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        scatter(self.red, c.slot_red, c.red_size)
        for i, b in enumerate(self.blocks):
            scatter(b, c.slot_blocks[i], c.scaffold_sizes[i])

        self._latch_base[env_ids] = False
        self._latch_stack[env_ids] = False
        self._latch_red[env_ids] = False
        self._streak[env_ids] = 0

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch transient achievements + advance the persistence streak at sim rate
        (buffers are fresh — env.step updates the scene before calling this)."""
        self._latch_base |= self._base_now()
        self._latch_stack |= self._stack_now()
        self._latch_red |= self._red_up_now()
        core = self.seated()
        self._streak = torch.where(core, self._streak + 1,
                                   torch.zeros_like(self._streak))

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "ring": self.ring.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blocks": [b.data.root_state_w[env_ids].clone() for b in self.blocks],
            "required_pair": self.required_pair[env_ids].clone(),
            "band_z": self.band_z[env_ids].clone(),
            "latch_base": self._latch_base[env_ids].clone(),
            "latch_stack": self._latch_stack[env_ids].clone(),
            "latch_red": self._latch_red[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.ring.write_root_state_to_sim(state["ring"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        for b, st in zip(self.blocks, state["blocks"]):
            b.write_root_state_to_sim(st, env_ids)
        self.required_pair[env_ids] = state["required_pair"]
        self.band_z[env_ids] = state["band_z"]
        self._latch_base[env_ids] = state["latch_base"]
        self._latch_stack[env_ids] = state["latch_stack"]
        self._latch_red[env_ids] = state["latch_red"]
        self._streak[env_ids] = state["streak"]

    # ----- description ----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        sizes = " / ".join(f"{s * 1000:.0f}" for s in c.scaffold_sizes)
        return (
            f"A round dark plinth ({2 * c.pad_r * 1000:.0f} mm wide, "
            f"{c.pad_t * 1000:.0f} mm high) stands on the ground, with a floating golden "
            f"ring hovering directly above it — the ring is intangible, a height marker "
            f"only, and its height changes between episodes. Scattered on the other side "
            f"lie a small red cube ({c.red_size * 1000:.0f} mm) and three scaffold blocks "
            f"of different heights ({sizes} mm).\n"
            f"Goal: make the red cube REST with its center at the ring's height, directly "
            f"above the plinth, standing freely — nothing may hold it there. No single "
            f"block is tall enough: exactly one PAIR of scaffold blocks stacks to the "
            f"right height. Read the ring, choose that pair, stack it on the plinth "
            f"(larger block first), and set the red cube on top. A cube merely lifted to "
            f"the ring falls when released and scores nothing; a tower built beside the "
            f"plinth, or with the wrong blocks, does not count."
        )

    # ----- predicates / rubric --------------------------------------------------------------
    def _block_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,B,3), |lin_vel| (N,B)) for the scaffold blocks."""
        pos = torch.stack([b.data.root_pos_w for b in self.blocks], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.blocks], dim=1)
        return pos, vel

    def _base_now(self) -> torch.Tensor:
        """(N,) bool: some scaffold block settled ON the plinth top (resting height is
        20 mm above its on-ground height — unambiguous by construction)."""
        c = self.cfg
        pos, vel = self._block_tensors()
        pad_xy = self.pad.data.root_pos_w[:, None, :2]
        z_rel = pos[:, :, 2] - self.env_origins[:, None, 2]
        on_pad_z = c.pad_t + self._sizes[None, :] / 2
        ok = ((pos[:, :, :2] - pad_xy).norm(dim=-1) < c.pad_r) \
            & ((z_rel - on_pad_z).abs() < 0.010) & (vel < c.settle_speed)
        return ok.any(dim=1)

    def _stack_now(self) -> torch.Tensor:
        """(N,) bool: some scaffold block settled ELEVATED over the plinth (at least
        `elev_margin` above its own on-plinth resting height -> it stands on a block)."""
        c = self.cfg
        pos, vel = self._block_tensors()
        pad_xy = self.pad.data.root_pos_w[:, None, :2]
        z_rel = pos[:, :, 2] - self.env_origins[:, None, 2]
        thresh = c.pad_t + self._sizes[None, :] / 2 + c.elev_margin
        ok = ((pos[:, :, :2] - pad_xy).norm(dim=-1) < c.pad_r) \
            & (z_rel > thresh) & (vel < c.settle_speed)
        return ok.any(dim=1)

    def _red_up_now(self) -> torch.Tensor:
        """(N,) bool: the red cube settled ELEVATED on structure over the plinth."""
        c = self.cfg
        rp = self.red.data.root_pos_w
        rv = self.red.data.root_lin_vel_w.norm(dim=-1)
        pad_xy = self.pad.data.root_pos_w[:, :2]
        z_rel = rp[:, 2] - self.env_origins[:, 2]
        thresh = c.pad_t + c.red_size / 2 + c.elev_margin
        return ((rp[:, :2] - pad_xy).norm(dim=-1) < c.pad_r) & (z_rel > thresh) \
            & (rv < c.settle_speed)

    def settled(self) -> torch.Tensor:
        """(N,) bool: payload AND every scaffold block below `settle_speed`."""
        _, bv = self._block_tensors()
        rv = self.red.data.root_lin_vel_w.norm(dim=-1)
        return (rv < self.cfg.settle_speed) & (bv < self.cfg.settle_speed).all(dim=1)

    def seated(self) -> torch.Tensor:
        """(N,) bool, the success CORE (instantaneous): red center inside the band, over
        the plinth axis within `xy_tol`, everything settled."""
        c = self.cfg
        rp = self.red.data.root_pos_w
        pad_xy = self.pad.data.root_pos_w[:, :2]
        z_rel = rp[:, 2] - self.env_origins[:, 2]
        in_band = (z_rel - self.band_z).abs() < c.band_tol
        over_pad = (rp[:, :2] - pad_xy).norm(dim=-1) < c.xy_tol
        return in_band & over_pad & self.settled()

    def success(self) -> torch.Tensor:
        """(N,) bool: the core has held for `hold_steps` consecutive substeps — a
        free-standing physical outcome. A kinematically held cube never accumulates the
        streak (its post-step speed is g*dt ≈ 0.082 m/s > settle_speed)."""
        return self.seated() & (self._streak >= self.cfg.hold_steps)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 nothing; 0.20 block on plinth; 0.45 two-block
        scaffold; 0.70 red elevated on structure; 1.0 iff success(). Latched."""
        base = self._latch_base | self._base_now()
        stack = self._latch_stack | self._stack_now()
        red_up = self._latch_red | self._red_up_now()
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = torch.where(base, torch.full_like(s, 0.20), s)
        s = torch.where(stack, torch.full_like(s, 0.45), s)
        s = torch.where(red_up, torch.full_like(s, 0.70), s)
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("sim_gen", lambda: EnvCfg(scene="scaffold_lift", robot="null"))
