"""VaultUnstackScene — un-stack the tower capping a vault, store the blocks tidily,
deliver the buried prize (sim_gen task `stack_pyramid_i17`, derived from
maniskill/stack_pyramid).

The seed BUILDS a pyramid: pick scattered cubes and stack them (blue on top of red +
green); success is a purely constructive on-top geometric check. This task is the
strategic inverse — SUBTRACTIVE, access-driven, order-constrained:

  A red/green/blue tower already stands, capping the square well of a low vault in
  which a golden prize cube is buried (the cap cube spans the aperture — the prize is
  physically imprisoned while the tower stands). The solver must (1) un-stack the tower
  top-down, (2) store each block resting flat ON THE FLOOR of a walled storage tray —
  stacking blocks inside the tray does NOT count, and blocks dumped on the ground next
  to the tray do NOT count — and (3) extract the prize and set it on the delivery pad.

  Stacking — the seed's entire skill — is rewarded NOWHERE: re-building the tower
  anywhere scores ~0.1, and a block placed on top of another block inside the tray is
  rejected by the rest-on-tray-floor gate. Toppling the tower to "free" the prize fails
  too: the blocks end up outside the tray and the score stalls.

Judged on PHYSICAL outcomes only:
  - success(): each of the 3 tower blocks settled, resting on the tray floor inside the
    tray walls; the prize settled on the pad. (The cover is off by necessity — all
    blocks are in the tray.)
  - score() in [0,1]: 0.15 per block stored (settled on the tray floor) + 0.10 for the
    latched "well uncovered" transient + 0.15 while the prize is out of the well +
    0.25 for the prize settled on the pad; exactly 1.0 iff success (max partial 0.95;
    doing nothing scores 0).

Per-episode randomization: vault position, tray position + yaw, pad position, per-block
stack-alignment jitter + yaw, prize position inside the well.

Assets are fully procedural: one custom compound spawner (open box = floor + 4 walls,
one kinematic rigid body — used for both the vault and the tray) plus plain CuboidCfg
blocks. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawner -------------------------------------------------------------------
# One KINEMATIC rigid body: a floor slab + 4 wall boxes around a rectangular aperture
# (the pen_holder cup pattern: child colliders of one body never self-collide; only
# `isaaclab.sim.utils.clone` is borrowed for the per-env replicate machinery).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_open_box(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author an open-topped box at `prim_path`: root Xform with a KINEMATIC RigidBodyAPI,
    a floor slab (top at z=floor_t in the root frame, root origin at ground level) and 4
    wall boxes rising `wall_h` above the floor around the `inner_hx` x `inner_hy` aperture.
    Explicit small contact offsets (the default ~2 cm would eat the well clearances)."""
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    color = Gf.Vec3f(*cfg.color)

    def box(name: str, size, center) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        sxf.AddScaleOp().Set(Gf.Vec3f(*size))
        seg.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    ohx = cfg.inner_hx + cfg.wall_t
    ohy = cfg.inner_hy + cfg.wall_t
    zc = cfg.floor_t + cfg.wall_h / 2
    box("floor", (2 * ohx, 2 * ohy, cfg.floor_t), (0.0, 0.0, cfg.floor_t / 2))
    box("wall_xp", (cfg.wall_t, 2 * ohy, cfg.wall_h), (cfg.inner_hx + cfg.wall_t / 2, 0.0, zc))
    box("wall_xn", (cfg.wall_t, 2 * ohy, cfg.wall_h), (-(cfg.inner_hx + cfg.wall_t / 2), 0.0, zc))
    box("wall_yp", (2 * cfg.inner_hx, cfg.wall_t, cfg.wall_h), (0.0, cfg.inner_hy + cfg.wall_t / 2, zc))
    box("wall_yn", (2 * cfg.inner_hx, cfg.wall_t, cfg.wall_h), (0.0, -(cfg.inner_hy + cfg.wall_t / 2), zc))
    return root


def _open_box_cfg(*, inner_hx: float, inner_hy: float, wall_t: float, floor_t: float,
                  wall_h: float, color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the open-box spawner cfg — `clone` wraps
    `_spawn_open_box` exactly like `spawn_cuboid` is wrapped."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "open_box" not in _SPAWNER_CACHE:

        @configclass
        class OpenBoxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_open_box)
            inner_hx: float = 0.05
            inner_hy: float = 0.05
            wall_t: float = 0.012
            floor_t: float = 0.012
            wall_h: float = 0.045
            color: tuple = (0.3, 0.3, 0.35)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["open_box"] = OpenBoxSpawnerCfg

    return _SPAWNER_CACHE["open_box"](
        inner_hx=inner_hx, inner_hy=inner_hy, wall_t=wall_t, floor_t=floor_t,
        wall_h=wall_h, color=color, contact_offset=contact_offset,
    )


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class VaultUnstackSceneCfg(BaseCfg):
    """Config for `VaultUnstackScene`. Env-local frame: ground at z=0; the vault root sits
    at ground level with the well rim at z = `rim_top` (derived)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.04)     # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.60)     # max |ang vel| when judging settled (rad/s)
    tray_xy_margin: float = tunable(0.002)  # slack on the block-inside-tray xy bound
    tray_bot_lo: float = tunable(0.008)   # block bottom may sit this far BELOW the tray
    # floor top (contact-offset slack) ...
    tray_bot_hi: float = tunable(0.012)   # ... and at most this far ABOVE it: a block on
    # top of another block (bottom +65..80 mm) or perched on the walls NEVER counts.
    pad_xy_tol: float = tunable(0.032)    # prize center within this of the pad center
    # (per axis); pad half 55 mm - prize half 18 mm - 5 mm => fully on the pad, honest.
    pad_z_tol: float = tunable(0.008)     # prize center height band around pad_top + half

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fixture_jitter: float = tunable(0.03)   # uniform +/- xy jitter: vault, tray, pad
    tray_yaw_deg: float = tunable(20.0)     # uniform +/- tray yaw
    block_yaw_deg: float = tunable(15.0)    # uniform +/- yaw per tower block
    stack_jitter: float = tunable(0.004)    # uniform +/- xy misalignment per stack level
    prize_jitter: float = tunable(0.006)    # uniform +/- xy of the prize inside the well

    # --- info: structure ----------------------------------------------------------------------
    # Vault: square well, aperture 64 mm — the 36 mm prize rattles freely, but the 80 mm
    # cap block spans the aperture (>= 5 mm of rim support per side under max jitter), so
    # the prize is physically imprisoned while the tower stands (asserted by a kick probe).
    well_half: float = info(0.032)
    vault_wall_t: float = info(0.012)
    well_depth: float = info(0.045)   # interior depth; prize top (48 mm) stays below the rim
    vault_floor_t: float = info(0.012)
    vault_pos: tuple = info((0.32, 0.0))
    vault_color: tuple = info((0.28, 0.30, 0.38))
    # Tower blocks, bottom -> top (seed colors: red cap, green mid, blue top). Sizes
    # DESCEND so the reset tower is boringly stable; the cap is the only block that can
    # cover the aperture.
    blocks: tuple = info((
        ("cap", 0.080, 0.15, (0.85, 0.15, 0.15)),
        ("mid", 0.065, 0.08, (0.15, 0.70, 0.20)),
        ("top", 0.050, 0.05, (0.15, 0.30, 0.85)),
    ))
    prize_size: float = info(0.036)
    prize_mass: float = info(0.04)
    prize_color: tuple = info((0.95, 0.78, 0.15))
    # Storage tray: inner 270 x 140 mm — all three blocks fit side by side ON THE FLOOR
    # (80+65+50 = 195 mm + gaps); walls 40 mm above the floor, so a block must be lifted in.
    tray_inner_hx: float = info(0.135)
    tray_inner_hy: float = info(0.070)
    tray_wall_t: float = info(0.012)
    tray_floor_t: float = info(0.012)
    tray_wall_h: float = info(0.040)
    tray_pos: tuple = info((-0.02, 0.30))
    tray_color: tuple = info((0.55, 0.40, 0.22))
    # Delivery pad: a thin slab; the prize must rest fully ON it.
    pad_half: float = info(0.055)
    pad_t: float = info(0.012)
    pad_pos: tuple = info((-0.02, -0.30))
    pad_color: tuple = info((0.48, 0.20, 0.62))
    contact_offset: float = info(0.002)
    stack_gap: float = info(0.0015)  # reset-time inter-block air gap (settles in a few steps)

    # Derived (filled in __post_init__).
    rim_top: float = field(default=None, init=False)
    tray_floor_top: float = field(default=None, init=False)
    pad_top: float = field(default=None, init=False)
    halves: tuple = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.rim_top = round(self.vault_floor_t + self.well_depth, 4)
        self.tray_floor_top = self.tray_floor_t
        self.pad_top = self.pad_t
        self.halves = tuple(s / 2 for _n, s, _m, _c in self.blocks)


# ----- scene ---------------------------------------------------------------------------------------
@SCENES.register("vault_unstack")
class VaultUnstackScene(BaseScene):
    cfg: VaultUnstackSceneCfg

    def __init__(self, cfg: VaultUnstackSceneCfg | None = None) -> None:
        super().__init__(cfg or VaultUnstackSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic vault / tray / pad, the 3 tower blocks (stacked on
        the vault rim) and the prize (inside the well). reset() re-places everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6, dynamic_friction=0.5, restitution=0.0)

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
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault",
                spawn=_open_box_cfg(
                    inner_hx=c.well_half, inner_hy=c.well_half, wall_t=c.vault_wall_t,
                    floor_t=c.vault_floor_t, wall_h=c.well_depth, color=c.vault_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.vault_pos[0], c.vault_pos[1], 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_open_box_cfg(
                    inner_hx=c.tray_inner_hx, inner_hy=c.tray_inner_hy, wall_t=c.tray_wall_t,
                    floor_t=c.tray_floor_t, wall_h=c.tray_wall_h, color=c.tray_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.tray_pos[0], c.tray_pos[1], 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.pad_half, 2 * c.pad_half, c.pad_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.pad_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.pad_pos[0], c.pad_pos[1], c.pad_t / 2)),
            ),
        }
        z = c.rim_top
        for name, size, mass, rgb in c.blocks:
            z += size / 2
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(size, size, size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.vault_pos[0], c.vault_pos[1], z)),
            )
            z += size / 2 + c.stack_gap
        out["prize"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Prize",
            spawn=sim_utils.CuboidCfg(
                size=(c.prize_size, c.prize_size, c.prize_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05, angular_damping=0.10),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.prize_mass),
                collision_props=coll, physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.prize_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.vault_pos[0], c.vault_pos[1], c.vault_floor_t + c.prize_size / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode latch the rubric depends on."""
        super().bind(env)
        c = self.cfg
        n, dev = env.num_envs, env.device
        self.vault: RigidObject = env.iscene["vault"]
        self.tray: RigidObject = env.iscene["tray"]
        self.pad: RigidObject = env.iscene["pad"]
        self.prize: RigidObject = env.iscene["prize"]
        self.block_bodies: dict[str, RigidObject] = {
            name: env.iscene[name] for name, _s, _m, _c in c.blocks}
        self.env_origins = env.iscene.env_origins
        self._halves = torch.tensor(c.halves, device=dev)
        # uncovered[e]: the well aperture was observed clear of every tower block at some
        # substep this episode (latched transient achievement).
        self.uncovered = torch.zeros(n, dtype=torch.bool, device=dev)

    def _local(self, body) -> torch.Tensor:
        return body.data.root_pos_w - self.env_origins

    def _block_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(pos_local (N,3,3), |lin vel| (N,3), |ang vel| (N,3)) for the tower blocks."""
        pos = torch.stack([self._local(b) for b in self.block_bodies.values()], dim=1)
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.block_bodies.values()], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                           for b in self.block_bodies.values()], dim=1)
        return pos, lin, ang

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter the vault / tray / pad poses (kinematic pose writes),
        rebuild the tower on the vault rim (per-level alignment jitter + yaw), drop the
        prize into the well, clear the uncovered latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def jit(k: int = 2) -> torch.Tensor:
            return (torch.rand(m, k, device=dev) * 2 - 1) * c.fixture_jitter

        # --- kinematic fixtures: pose writes only ---
        vxy = torch.tensor(c.vault_pos, device=dev) + jit()
        pose = torch.zeros(m, 7, device=dev)
        pose[:, 0:2] = vxy
        pose[:, 3] = 1.0
        pose[:, 0:3] += origin
        self.vault.write_root_pose_to_sim(pose, env_ids)

        txy = torch.tensor(c.tray_pos, device=dev) + jit()
        tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        pose = torch.zeros(m, 7, device=dev)
        pose[:, 0:2] = txy
        pose[:, 3] = torch.cos(tyaw / 2)
        pose[:, 6] = torch.sin(tyaw / 2)
        pose[:, 0:3] += origin
        self.tray.write_root_pose_to_sim(pose, env_ids)

        gxy = torch.tensor(c.pad_pos, device=dev) + jit()
        pose = torch.zeros(m, 7, device=dev)
        pose[:, 0:2] = gxy
        pose[:, 2] = c.pad_t / 2
        pose[:, 3] = 1.0
        pose[:, 0:3] += origin
        self.pad.write_root_pose_to_sim(pose, env_ids)

        # --- tower: stacked on the vault rim, cumulative alignment jitter + per-block yaw ---
        z = torch.full((m,), c.rim_top, device=dev)
        xy = vxy.clone()
        for i, (name, size, _mass, _rgb) in enumerate(c.blocks):
            if i > 0:
                xy = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.stack_jitter
            z = z + size / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z + c.stack_gap
            half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.block_yaw_deg) / 2
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            self.block_bodies[name].write_root_state_to_sim(st, env_ids)
            z = z + size / 2 + c.stack_gap

        # --- prize: inside the well, xy jitter + free yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = vxy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.prize_jitter
        st[:, 2] = c.vault_floor_t + c.prize_size / 2 + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.prize.write_root_state_to_sim(st, env_ids)

        self.uncovered[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, every physics substep: the well aperture clear of every tower block."""
        self.uncovered |= ~self.covered()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "blocks": {n: b.data.root_state_w[env_ids].clone()
                       for n, b in self.block_bodies.items()},
            "prize": self.prize.data.root_state_w[env_ids].clone(),
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "uncovered": self.uncovered[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n, b in self.block_bodies.items():
            b.write_root_state_to_sim(state["blocks"][n], env_ids)
        self.prize.write_root_state_to_sim(state["prize"], env_ids)
        self.vault.write_root_pose_to_sim(state["vault"][:, 0:7], env_ids)
        self.tray.write_root_pose_to_sim(state["tray"][:, 0:7], env_ids)
        self.pad.write_root_pose_to_sim(state["pad"][:, 0:7], env_ids)
        self.uncovered[env_ids] = state["uncovered"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low vault (a {2 * c.well_half * 1000:.0f} mm square well sunk in a slate "
            f"plinth) holds a golden prize cube ({c.prize_size * 1000:.0f} mm), imprisoned "
            f"under a tower of three blocks stacked over the opening: red "
            f"({c.blocks[0][1] * 1000:.0f} mm) at the bottom capping the well, green "
            f"({c.blocks[1][1] * 1000:.0f} mm), then blue ({c.blocks[2][1] * 1000:.0f} mm) on "
            f"top. To one side stands an open storage tray (inner "
            f"{2 * c.tray_inner_hx * 1000:.0f} x {2 * c.tray_inner_hy * 1000:.0f} mm, walls "
            f"{c.tray_wall_h * 1000:.0f} mm); to the other, a flat violet delivery pad. All "
            f"positions change every episode.\n"
            f"Goal: dismantle the tower from the top down and store each block resting flat "
            f"on the TRAY FLOOR (blocks stacked on each other inside the tray do not count, "
            f"nor do blocks dumped on the ground beside it), then take the golden prize out "
            f"of the well and set it on the delivery pad. Re-stacking the blocks anywhere is "
            f"worthless, and toppling the tower just scatters blocks you must still store."
        )

    # ----- progress / rubric --------------------------------------------------------------------
    def covered(self) -> torch.Tensor:
        """(N,) bool: some tower block still covers (or hovers over) the well aperture —
        its footprint can overlap the aperture and it is at rim height or above."""
        c = self.cfg
        v = self._local(self.vault)
        pos, _lin, _ang = self._block_tensors()
        d = (pos[:, :, 0:2] - v[:, None, 0:2]).abs()
        reach = c.well_half + self._halves[None, :]
        near = (d[:, :, 0] < reach - 0.005) & (d[:, :, 1] < reach - 0.005)
        z = pos[:, :, 2]
        over = (z > c.rim_top - 0.5 * self._halves[None, :]) & (z < c.rim_top + 0.35)
        return (near & over).any(dim=1)

    def prize_in_well(self) -> torch.Tensor:
        """(N,) bool: prize center inside the well interior volume."""
        c = self.cfg
        v = self._local(self.vault)
        p = self._local(self.prize)
        d = (p[:, 0:2] - v[:, 0:2]).abs()
        return (d[:, 0] < c.well_half) & (d[:, 1] < c.well_half) & (p[:, 2] < c.rim_top)

    def blocks_settled(self) -> torch.Tensor:
        """(N,3) bool: per-block velocity gates."""
        _pos, lin, ang = self._block_tensors()
        return (lin < self.cfg.settle_lin) & (ang < self.cfg.settle_ang)

    def prize_settled(self) -> torch.Tensor:
        lin = self.prize.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin
        ang = self.prize.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang
        return lin & ang

    def in_tray(self) -> torch.Tensor:
        """(N,3) bool, geometric: block center inside the tray walls (tray body frame,
        the tray yaws) AND its bottom face resting ON THE TRAY FLOOR — a block stacked on
        another block, or perched on the walls, is rejected by the bottom-height gate."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        pos, _lin, _ang = self._block_tensors()
        n = pos.shape[0]
        tq = self.tray.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
        tp = self._local(self.tray)[:, None, :]
        loc = quat_apply_inverse(tq, (pos - tp).reshape(n * 3, 3)).reshape(n, 3, 3)
        h = self._halves[None, :]
        in_x = loc[:, :, 0].abs() < c.tray_inner_hx - h + c.tray_xy_margin
        in_y = loc[:, :, 1].abs() < c.tray_inner_hy - h + c.tray_xy_margin
        bot = loc[:, :, 2] - h - c.tray_floor_top
        on_floor = (bot > -c.tray_bot_lo) & (bot < c.tray_bot_hi)
        return in_x & in_y & on_floor

    def stored(self) -> torch.Tensor:
        """(N,3) bool: in the tray, on its floor, settled — what the rubric counts."""
        return self.in_tray() & self.blocks_settled()

    def prize_on_pad(self) -> torch.Tensor:
        """(N,) bool: prize settled flat on the delivery pad (fully on it, top height)."""
        c = self.cfg
        p = self._local(self.prize)
        g = self._local(self.pad)
        d = (p[:, 0:2] - g[:, 0:2]).abs()
        xy = (d[:, 0] < c.pad_xy_tol) & (d[:, 1] < c.pad_xy_tol)
        z_tgt = g[:, 2] + c.pad_t / 2 + c.prize_size / 2
        z_ok = (p[:, 2] - z_tgt).abs() < c.pad_z_tol
        return xy & z_ok & self.prize_settled()

    def success(self) -> torch.Tensor:
        """(N,) bool: all 3 blocks stored on the tray floor + prize settled on the pad.
        (The well is uncovered by necessity: every block is in the tray.)"""
        return self.stored().all(dim=1) & self.prize_on_pad()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 per stored block + 0.10 latched well-uncovered +
        0.15 prize out of the well (current state) + 0.25 prize on the pad; exactly 1.0
        iff success (max partial 0.95). Doing nothing scores 0."""
        part = (0.15 * self.stored().float().sum(dim=1)
                + 0.10 * self.uncovered.float()
                + 0.15 * (~self.prize_in_well()).float()
                + 0.25 * self.prize_on_pad().float())
        return torch.where(self.success(), torch.ones_like(part), part)


# ----- env registration ----------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="vault_unstack", robot="null", env_spacing=4.0))
