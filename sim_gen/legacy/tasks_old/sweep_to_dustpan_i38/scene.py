"""SowingTrayScene — de-aggregate a clumped hopper of pellets: EXACTLY ONE per planter cell
(sim_gen task `sweep_to_dustpan_i38`, derived from rlbench/sweep_to_dustpan).

The seed is TOOL-MEDIATED AGGREGATION: five loose dirt cubes are swept with a broom across
the plane into one dustpan — an indiscriminate many-to-one collection where the whole pile
is one undifferentiated blob and the plane itself is the transport channel. This task keeps
the seed's vocabulary (a scatter of small granular objects, a container, a tabletop) and
inverts every load-bearing pillar of that plan:

  - The material starts ALREADY AGGREGATED: 4-6 seed pellets lie clumped inside an open
    hopper cup (the dustpan-analog is the SOURCE, not the goal).
  - The goal is DISPERSAL with EXACT COUNTING: an elevated planting tray (a walled bed,
    3x2 grid of cells) must end with exactly ONE pellet resting on the floor of each used
    cell — one pellet per cell, every present pellet planted, no cell double-seeded.
  - A cell containing two or more pellets is VOIDED (scores nothing until fixed), so the
    seed's plan class — move the whole pile to the container in bulk (sweep it, pour the
    hopper, dump the clump) — is a tested failing control, not merely suboptimal.
  - The tray bed is raised 55 mm off the ground and ringed by walls: planar pushing /
    sweeping along the ground can NEVER score (a pellet swept flush against the planter's
    base is still at ground level, outside every cell) — the seed's transport channel is
    physically disconnected from the goal region.

So a solver needs a different PLAN: repeated prehensile de-aggregation — extract ONE
pellet at a time out of a deep narrow cup, carry it over the bed, and precision-drop it
into a chosen empty cell (18 mm radial funnel per cell), keeping count across 4-6 cycles.
Cell order is free; the horizon is 4-6 extract->transport->drop stages.

Judged on PHYSICAL outcomes only:
  - `good_cells()`: a cell counts iff exactly one present pellet occupies its airspace AND
    that pellet rests ON THE CELL FLOOR (per-axis xy gate 28 mm — honest by construction:
    the max physical floor offset is cell/2 - r = 18 mm, while a pellet perched on a wall
    top sits at 36 mm; z gate rejects wall-perches and pellets riding on top of others)
    and is settled.
  - success(): number of good cells == number of present pellets (each pellet is then the
    sole floor occupant of its own cell; cells are disjoint so this is exact).
  - score(): 0 for doing nothing; +w_extract (0.10, latched in post_step) once any pellet
    is genuinely raised clear above the hopper's rim height; +w_frac (0.70) * fraction of
    present pellets planted-alone; exactly 1.0 iff success. Max non-success ~ 0.68.

Per-episode randomization: pellet count (4-6), tray pose (xy jitter + FULL yaw — cell
axes rotate, so memorized drop points fail), hopper pose (xy + yaw), clump arrangement
phase. Absent pellets park in an off-camera ground depot and are masked everywhere.

Assets are fully procedural (no external files): a dynamic octagonal hopper cup
(pen_holder compound-spawner pattern), a KINEMATIC raised planting tray (bed slab + grid
walls, one body, re-posed per reset by pose writes), six dynamic sphere pellets. Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners ----------------------------------------------------------------
# One rigid body each, authored with raw pxr APIs; `isaaclab.sim.utils.clone` supplies the
# regex-resolve + per-env replication (each cloned prim is authored fresh — idempotent, no
# duplicate-xformOp trap).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the open hopper cup at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI (overlapping wall segments would double-count density), a bottom disc collider
    and 8 box wall segments forming an octagonal shell of inner inradius `inner_r`."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    # Gentle overlap resolution (the factory-env insertion trick): pellets teleport-dropped
    # into the cup must not be ejected ballistically by the depenetration solver.
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateMaxDepenetrationVelocityAttr(0.5)

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

    n = 8
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


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC raised planting tray at `prim_path`. Body origin = centre of the
    bed floor's TOP face: the bed slab hangs below (local z in [-bed_t, 0], resting on the
    ground), the grid walls rise above (local z in [0, wall_h]). (nx+1) x-boundary walls and
    (ny+1) y-boundary walls partition the bed into an nx x ny grid of square cells of inner
    side `cell` at pitch `cell + wall_t` (overlaps at wall crossings are harmless inside one
    body)."""
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

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    def box(name: str, center, scale, color) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*center))
        sxf.AddScaleOp().Set(Gf.Vec3f(*scale))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collide(seg.GetPrim())

    pitch = cfg.cell + cfg.wall_t
    outer_x = cfg.nx * pitch + cfg.wall_t
    outer_y = cfg.ny * pitch + cfg.wall_t
    box("bed", (0.0, 0.0, -cfg.bed_t / 2), (outer_x, outer_y, cfg.bed_t), cfg.bed_color)
    for m in range(cfg.nx + 1):
        xb = (m - cfg.nx / 2) * pitch
        box(f"wx_{m}", (xb, 0.0, cfg.wall_h / 2), (cfg.wall_t, outer_y, cfg.wall_h),
            cfg.wall_color)
    for m in range(cfg.ny + 1):
        yb = (m - cfg.ny / 2) * pitch
        box(f"wy_{m}", (0.0, yb, cfg.wall_h / 2), (outer_x, cfg.wall_t, cfg.wall_h),
            cfg.wall_color)
    return root


def _hopper_spawner_cfg(*, inner_r: float, wall_t: float, height: float, bot_t: float,
                        mass: float, color: tuple, contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hopper" not in _SPAWNER_CACHE:

        @configclass
        class HopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hopper)
            inner_r: float = 0.055
            wall_t: float = 0.008
            height: float = 0.090
            bot_t: float = 0.010
            mass: float = 0.20
            color: tuple = (0.30, 0.40, 0.50)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["hopper"] = HopperSpawnerCfg

    return _SPAWNER_CACHE["hopper"](
        inner_r=inner_r, wall_t=wall_t, height=height, bot_t=bot_t, mass=mass,
        color=color, contact_offset=contact_offset,
    )


def _tray_spawner_cfg(*, nx: int, ny: int, cell: float, wall_t: float, wall_h: float,
                      bed_t: float, bed_color: tuple, wall_color: tuple,
                      contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tray" not in _SPAWNER_CACHE:

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            nx: int = 3
            ny: int = 2
            cell: float = 0.064
            wall_t: float = 0.008
            wall_h: float = 0.032
            bed_t: float = 0.055
            bed_color: tuple = (0.45, 0.30, 0.18)
            wall_color: tuple = (0.58, 0.44, 0.28)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["tray"] = TraySpawnerCfg

    return _SPAWNER_CACHE["tray"](
        nx=nx, ny=ny, cell=cell, wall_t=wall_t, wall_h=wall_h, bed_t=bed_t,
        bed_color=bed_color, wall_color=wall_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class SowingTraySceneCfg(BaseCfg):
    """Config for `SowingTrayScene`. Env-local frame: hopper on the left (-x), the raised
    planting tray on the right (+x); tray cell index = j * nx + i (i along tray-local x,
    j along tray-local y)."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    cell_xy_gate: float = tunable(0.028)  # per-axis |offset| of a pellet centre from a cell
    # centre, in the TRAY frame. Honest by construction: max physical on-floor offset =
    # cell/2 - pellet_r = 18 mm < 28 mm (any pellet genuinely resting on a cell floor
    # counts); a pellet perched on a wall top sits at cell/2 + wall_t/2 = 36 mm — rejected.
    floor_z_tol: float = tunable(0.010)  # |z above bed floor - pellet_r| gate for "resting
    # ON the floor" (a pellet riding on top of another sits ~3r = 42 mm — rejected by 2x).
    settle_lin: float = tunable(0.05)   # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(2.5)    # max |ang vel| (rad/s); angular damping kills spin
    extract_z: float = tunable(0.130)   # `extracted` latches when a present pellet centre
    # rises above this env-local z: hopper rim = 0.092, tray wall-perch top = 0.101 — a
    # 29 mm margin over the tallest passive resting pose, so only a genuine lift latches.

    # --- tunable: score weights --------------------------------------------------------------
    w_extract: float = tunable(0.10)  # latched: some pellet genuinely raised clear of the rim
    w_frac: float = tunable(0.70)     # x fraction of present pellets planted-alone

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    min_present: int = tunable(4)      # pellet count sampled uniformly in {min_present..6}
    tray_jitter: float = tunable(0.03)     # uniform +/- xy jitter of the tray at reset
    tray_yaw_deg: float = tunable(180.0)   # uniform +/- yaw of the tray (cell axes rotate)
    hopper_jitter: float = tunable(0.03)   # uniform +/- xy jitter of the hopper at reset
    clump_r: float = tunable(0.020)        # clump ring radius inside the hopper
    clump_jitter: float = tunable(0.003)   # per-pellet xy jitter inside the clump

    # --- info: structure ---------------------------------------------------------------------
    nx: int = info(3)
    ny: int = info(2)
    cell: float = info(0.064)      # cell inner side; radial funnel = cell/2 - pellet_r = 18 mm
    wall_t: float = info(0.008)
    wall_h: float = info(0.032)    # grid walls above the bed floor; rim at bed_z + wall_h
    bed_z: float = info(0.055)     # bed floor height above the ground — planar sweeping can
    # never enter a cell: a pellet on the ground sits 41 mm BELOW the lowest counted pose.
    tray_pos: tuple = info((0.16, 0.0))
    bed_color: tuple = info((0.45, 0.30, 0.18))
    wall_color: tuple = info((0.58, 0.44, 0.28))
    hopper_inner_r: float = info(0.055)  # interior holds the 4-6 pellet clump in two layers
    hopper_wall_t: float = info(0.008)
    hopper_h: float = info(0.090)        # rim at ~0.092 (deep: pellets must be lifted out)
    hopper_bot_t: float = info(0.010)
    hopper_mass: float = info(0.20)
    hopper_pos: tuple = info((-0.24, 0.0))
    hopper_color: tuple = info((0.30, 0.40, 0.50))
    n_pellets: int = info(6)             # rigid bodies; per-episode presence sampled
    pellet_r: float = info(0.014)
    pellet_mass: float = info(0.03)
    pellet_color: tuple = info((0.75, 0.85, 0.20))
    friction: float = info(0.60)
    contact_offset: float = info(0.002)  # small: the 18 mm cell funnel must not be eaten by
    # speculative contact (2+2 mm combined < the centred 18 mm gap).
    parking_pos: tuple = info((1.1, 1.1))  # off-camera ground depot for absent pellets

    # Derived (filled in __post_init__).
    pitch: float = field(default=None, init=False)
    n_cells: int = field(default=None, init=False)
    cell_offsets: tuple = field(default=None, init=False)  # tray-local (x, y) per cell
    occ_z_max: float = field(default=None, init=False)  # airspace ceiling above the bed floor
    hopper_rim_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.pitch = round(self.cell + self.wall_t, 4)
        self.n_cells = self.nx * self.ny
        offs = []
        for j in range(self.ny):
            for i in range(self.nx):
                offs.append((round((i - (self.nx - 1) / 2) * self.pitch, 4),
                             round((j - (self.ny - 1) / 2) * self.pitch, 4)))
        self.cell_offsets = tuple(offs)
        # A second pellet stacked on a floor pellet sits at ~3r = 42 mm — inside the
        # airspace, so it voids the cell; a wall-perch (46 mm) is excluded by xy anyway.
        self.occ_z_max = round(self.wall_h + self.pellet_r + 0.006, 4)
        self.hopper_rim_z = round(self.hopper_h + 0.002, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("sowing_tray")
class SowingTrayScene(BaseScene):
    cfg: SowingTraySceneCfg

    def __init__(self, cfg: SowingTraySceneCfg | None = None) -> None:
        super().__init__(cfg or SowingTraySceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic raised tray, the dynamic hopper cup, six dynamic
        sphere pellets. reset() re-poses everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.friction, dynamic_friction=c.friction, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

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
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=_tray_spawner_cfg(
                    nx=c.nx, ny=c.ny, cell=c.cell, wall_t=c.wall_t, wall_h=c.wall_h,
                    bed_t=c.bed_z, bed_color=c.bed_color, wall_color=c.wall_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.tray_pos[0], c.tray_pos[1], c.bed_z)),
            ),
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=_hopper_spawner_cfg(
                    inner_r=c.hopper_inner_r, wall_t=c.hopper_wall_t, height=c.hopper_h,
                    bot_t=c.hopper_bot_t, mass=c.hopper_mass, color=c.hopper_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hopper_pos[0], c.hopper_pos[1], c.hopper_h / 2 + 0.002)),
            ),
        }
        for i in range(c.n_pellets):
            out[f"pellet_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pellet_" + str(i),
                spawn=sim_utils.SphereCfg(
                    radius=c.pellet_r,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.40),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.pellet_mass),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.pellet_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hopper_pos[0], c.hopper_pos[1],
                         c.hopper_bot_t + c.pellet_r + 0.02 + 0.03 * i)),
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

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate per-episode buffers."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.tray: RigidObject = env.iscene["tray"]
        self.hopper: RigidObject = env.iscene["hopper"]
        self.pellets: list[RigidObject] = [env.iscene[f"pellet_{i}"]
                                           for i in range(c.n_pellets)]
        self.env_origins = env.iscene.env_origins
        self.tray_c = torch.zeros(n, 2, device=dev)     # tray centre (env-local xy)
        self.tray_yaw = torch.zeros(n, device=dev)
        self.hopper_c = torch.zeros(n, 2, device=dev)   # hopper centre (env-local xy)
        self.present = torch.ones(n, c.n_pellets, dtype=torch.bool, device=dev)
        self.extracted = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cell_off = torch.tensor(c.cell_offsets, device=dev)  # (C, 2), tray frame

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose the kinematic tray (jitter + full yaw), the hopper
        (jitter + yaw), sample the pellet count and stack the present pellets as a clump
        inside the hopper (two layers of three around a random phase); park absent pellets
        in the ground depot; clear the latch."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- tray: kinematic pose-only write (jitter + yaw) ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tray_yaw_deg)
        tc = torch.tensor(c.tray_pos, device=dev).expand(m, 2).clone()
        tc += (torch.rand(m, 2, device=dev) * 2 - 1) * c.tray_jitter
        self.tray_c[env_ids] = tc
        self.tray_yaw[env_ids] = yaw
        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = tc
        st[:, 2] = c.bed_z
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.tray.write_root_pose_to_sim(st, env_ids)

        # --- hopper: dynamic, upright, jitter + yaw ---
        hc = torch.tensor(c.hopper_pos, device=dev).expand(m, 2).clone()
        hc += (torch.rand(m, 2, device=dev) * 2 - 1) * c.hopper_jitter
        self.hopper_c[env_ids] = hc
        hyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = hc
        st[:, 2] = c.hopper_h / 2 + 0.002
        st[:, 3] = torch.cos(hyaw / 2)
        st[:, 6] = torch.sin(hyaw / 2)
        st[:, 0:3] += origin
        self.hopper.write_root_state_to_sim(st, env_ids)

        # --- pellet count: k ~ U{min_present..n_pellets}, random subset ---
        k = torch.randint(c.min_present, c.n_pellets + 1, (m,), device=dev)
        rank = torch.rand(m, c.n_pellets, device=dev).argsort(dim=1).argsort(dim=1)
        self.present[env_ids] = rank < k.unsqueeze(1)

        # --- pellets: clumped inside the hopper (layers of 3 on a ring, random phase) ---
        phase = torch.rand(m, device=dev) * 2 * math.pi
        floor_z = 0.002 + c.hopper_bot_t  # hopper cup floor (top of bottom disc)
        for i in range(c.n_pellets):
            layer, slot = divmod(i, 3)
            ang = phase + slot * (2 * math.pi / 3) + layer * (math.pi / 3)
            clump = torch.zeros(m, 3, device=dev)
            clump[:, 0] = hc[:, 0] + c.clump_r * torch.cos(ang)
            clump[:, 1] = hc[:, 1] + c.clump_r * torch.sin(ang)
            clump[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.clump_jitter
            clump[:, 2] = floor_z + c.pellet_r + 0.003 + layer * (2 * c.pellet_r + 0.004)
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.parking_pos[0] + (i % 3) * 0.10
            park[:, 1] = c.parking_pos[1] + (i // 3) * 0.10
            park[:, 2] = c.pellet_r + 0.002
            pres = self.present[env_ids, i].unsqueeze(1)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.where(pres, clump, park)
            st[:, 3] = 1.0
            self.pellets[i].write_root_state_to_sim(st, env_ids)

        self.extracted[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping every physics substep: EXTRACTED when any present pellet
        centre rises above `extract_z` — higher than the hopper rim and any passive
        resting pose in the scene, so only a genuine lift-out (or an airborne carry)
        fires it."""
        z = self._pellet_pos_local()[:, :, 2]
        self.extracted |= ((z > self.cfg.extract_z) & self.present).any(dim=1)

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pellets": [b.data.root_state_w[env_ids].clone() for b in self.pellets],
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "tray_c": self.tray_c[env_ids].clone(),
            "tray_yaw": self.tray_yaw[env_ids].clone(),
            "hopper_c": self.hopper_c[env_ids].clone(),
            "present": self.present[env_ids].clone(),
            "extracted": self.extracted[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for b, st in zip(self.pellets, state["pellets"]):
            b.write_root_state_to_sim(st, env_ids)
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.tray.write_root_pose_to_sim(state["tray"][:, 0:7], env_ids)
        self.tray_c[env_ids] = state["tray_c"]
        self.tray_yaw[env_ids] = state["tray_yaw"]
        self.hopper_c[env_ids] = state["hopper_c"]
        self.present[env_ids] = state["present"]
        self.extracted[env_ids] = state["extracted"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A deep open hopper cup ({2 * (c.hopper_inner_r + c.hopper_wall_t) * 100:.0f} cm "
            f"wide, {c.hopper_h * 100:.0f} cm tall) stands on the floor holding a clump of "
            f"{c.min_present}-{c.n_pellets} yellow-green seed pellets "
            f"({2 * c.pellet_r * 1000:.0f} mm spheres) — count what you see. To its side sits "
            f"a raised wooden planting tray: a {c.bed_z * 100:.0f} cm high bed divided by low "
            f"walls into a {c.nx} x {c.ny} grid of {c.cell * 1000:.0f} mm square cells, at a "
            f"random heading.\n"
            f"Goal: sow the tray — every pellet must end up resting on the floor of its OWN "
            f"cell, exactly one pellet per cell, all settled. Take pellets out of the hopper "
            f"one at a time, carry each over the bed, and drop it into an empty cell. A cell "
            f"holding two or more pellets counts for nothing until the extras are removed; "
            f"pellets pushed along the floor against the tray's base score nothing (the bed "
            f"is raised and walled — nothing at ground level is inside a cell)."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _pellet_pos_local(self) -> torch.Tensor:
        """(N, P, 3) pellet centres in env-local coords."""
        return torch.stack(
            [b.data.root_pos_w - self.env_origins for b in self.pellets], dim=1)

    def settled(self) -> torch.Tensor:
        """(N, P) bool: per-pellet lin + ang velocity below the settle gates."""
        c = self.cfg
        cols = []
        for b in self.pellets:
            lin = b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
            ang = b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
            cols.append(lin & ang)
        return torch.stack(cols, dim=1)

    def cell_centers_local(self) -> torch.Tensor:
        """(N, C, 2) cell centres in env-local xy (tray pose applied)."""
        cy, sy = torch.cos(self.tray_yaw), torch.sin(self.tray_yaw)
        ox, oy = self.cell_off[None, :, 0], self.cell_off[None, :, 1]
        x = self.tray_c[:, None, 0] + ox * cy[:, None] - oy * sy[:, None]
        y = self.tray_c[:, None, 1] + ox * sy[:, None] + oy * cy[:, None]
        return torch.stack([x, y], dim=-1)

    def _membership(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(occupancy, on_floor), both (N, P, C) bool. `occupancy`: present pellet inside
        cell airspace (per-axis xy gate in the TRAY frame, z anywhere between the bed floor
        and just above the wall rim — a pellet stacked on another still occupies).
        `on_floor`: additionally resting ON the cell floor (z gate `floor_z_tol`)."""
        c = self.cfg
        pos = self._pellet_pos_local()
        d = pos[:, :, :2] - self.tray_c[:, None, :]
        cy, sy = torch.cos(self.tray_yaw), torch.sin(self.tray_yaw)
        lx = d[:, :, 0] * cy[:, None] + d[:, :, 1] * sy[:, None]
        ly = -d[:, :, 0] * sy[:, None] + d[:, :, 1] * cy[:, None]
        loc = torch.stack([lx, ly], dim=-1)                       # (N, P, 2) tray frame
        rel = loc[:, :, None, :] - self.cell_off[None, None, :, :]  # (N, P, C, 2)
        xy_ok = rel.abs().amax(dim=-1) <= c.cell_xy_gate
        z_over = pos[:, :, 2] - c.bed_z                            # above the bed floor
        occ_z = (z_over > 0.004) & (z_over <= c.occ_z_max)
        floor_z = (z_over - c.pellet_r).abs() <= c.floor_z_tol
        pres = self.present[:, :, None]
        occupancy = xy_ok & occ_z[:, :, None] & pres
        on_floor = xy_ok & floor_z[:, :, None] & pres
        return occupancy, on_floor

    def cell_occupancy(self) -> torch.Tensor:
        """(N, C) int: present pellets inside each cell's airspace."""
        occupancy, _f = self._membership()
        return occupancy.sum(dim=1)

    def good_cells(self) -> torch.Tensor:
        """(N, C) bool: cell counts iff EXACTLY ONE present pellet occupies its airspace
        and that pellet rests settled on the cell floor — over-occupancy voids the cell."""
        occupancy, on_floor = self._membership()
        good = on_floor & self.settled()[:, :, None]
        return (occupancy.sum(dim=1) == 1) & (good.sum(dim=1) == 1)

    def n_present(self) -> torch.Tensor:
        return self.present.sum(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: as many good cells as present pellets. Cells are disjoint and each
        good cell holds exactly one pellet, so this holds iff EVERY present pellet is the
        sole settled floor occupant of its own cell — the physical dispersal goal."""
        return self.good_cells().sum(dim=1) == self.n_present()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for doing nothing (the clump occupies no cell);
        w_extract once a pellet was genuinely raised clear of the hopper rim (latched);
        + w_frac * fraction of present pellets planted-alone; exactly 1.0 iff success.
        Max non-success = 0.10 + 0.70 * 5/6 ~= 0.68."""
        c = self.cfg
        frac = self.good_cells().sum(dim=1).float() / self.n_present().clamp(min=1).float()
        base = c.w_extract * self.extracted.float() + c.w_frac * frac
        return torch.where(self.success(), torch.ones_like(base), base)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="sowing_tray", robot="null", env_spacing=4.0))
