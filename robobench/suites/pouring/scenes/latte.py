"""LatteScene — coffee cup + milk cup on a table; pour the milk into the coffee (Newton MPM).

Runs on IsaacLab develop's **Newton backend** with the implicit MPM solver: both liquids are
`MPMObject`s (explicit particle lattices seeded inside the cups), the cups are open-cylinder
trimesh colliders (exact meshes, no convex approximation), and the table a collidable box. The
milk cup is a *kinematic* rigid object so a script (or later a robot hand) can lift and tip it —
the MPM solver treats rigid geometry as colliders and follows their motion.

Layout (env-local meters): table top at z=0.04; the coffee cup stands at (0, 0), pre-filled with
brown "coffee" particles; the milk cup stands at `milk_cup_pos` (default (0.14, 0)), pre-filled
with white "milk" particles. Goal: pour the milk into the coffee cup — `transfer_fraction()`
(milk inside the coffee cup) is the success proxy, with `retention_fraction()` (coffee still in
its cup) and `spilled_fraction()` (milk on the table) as guards.

Requires the Newton venv (`env_newton`, see the README) to build; single-env only for now (the
MPM fixed grid spans the whole scene). Heavy imports are deferred so importing this module stays
app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robobench.core import SCENES, BaseCfg, BaseScene, info, tunable
from robobench.suites.pouring.newton_sim import MpmSimCfg

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

TABLE_TOP_Z = 0.04  # table top height [m]; cups stand here, layout numbers assume it


# ----- procedural geometry (numpy only; app-free) -------------------------------------------------
def cup_mesh(
    r_inner: float, height: float, wall: float, bottom: float, segments: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Open-cup trimesh: inner floor ring, inner+outer walls, rim, outer floor, and center caps.
    Local origin at the *outside bottom center*, +z up (same topology as the in-tree MPM pour
    demo's catch bowl, straight-walled)."""
    theta = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    r_outer = r_inner + wall

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(segments, z)])

    vertices = np.vstack(
        [
            ring(r_inner, bottom),  # 0: inner floor
            ring(r_inner, height),  # 1: inner rim
            ring(r_outer, height),  # 2: outer rim
            ring(r_outer, 0.0),  # 3: outer floor
            np.array([[0.0, 0.0, bottom], [0.0, 0.0, 0.0]], dtype=np.float32),  # center caps
        ]
    ).astype(np.float32)
    inner_center, outer_center = 4 * segments, 4 * segments + 1
    indices: list[int] = []
    for i in range(segments):
        j = (i + 1) % segments
        ib_i, ib_j = i, j
        it_i, it_j = i + segments, j + segments
        ot_i, ot_j = i + 2 * segments, j + 2 * segments
        ob_i, ob_j = i + 3 * segments, j + 3 * segments
        indices.extend([ib_i, it_i, ib_j, ib_j, it_i, it_j])  # inner wall
        indices.extend([ob_i, ob_j, ot_i, ot_i, ob_j, ot_j])  # outer wall
        indices.extend([it_i, ot_i, it_j, it_j, ot_i, ot_j])  # rim
        indices.extend([inner_center, ib_i, ib_j, outer_center, ob_j, ob_i])  # floor caps
    return vertices, np.asarray(indices, dtype=np.int32).reshape((-1, 3))


def cylinder_lattice(
    radius: float, z_lo: float, z_hi: float, voxel: float, particles_per_cell: float, density: float, seed: int
) -> tuple[np.ndarray, float, float]:
    """Jittered particle lattice filling a local-space cylinder (axis +z, centered on xy=0).
    Returns (points, particle_radius, particle_mass) — the in-tree pour demo's seeding recipe."""
    lo = np.array([-radius, -radius, z_lo], dtype=np.float32)
    hi = np.array([radius, radius, z_hi], dtype=np.float32)
    resolution = np.maximum(np.ceil(particles_per_cell * (hi - lo) / voxel), 1).astype(int)
    cell = (hi - lo) / resolution
    cell_volume = float(np.prod(cell))
    p_radius = float(cell.max() * 0.45)
    p_mass = float(cell_volume * density)
    axes = [np.arange(int(n) + 1) * c for n, c in zip(resolution, cell)]
    points = np.stack(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    rng = np.random.default_rng(seed)
    points += (rng.random(points.shape) - 0.5) * (0.10 * float(cell.max()))
    points += lo
    keep = points[:, 0] ** 2 + points[:, 1] ** 2 < radius**2
    points = points[keep]
    if points.shape[0] == 0:
        raise RuntimeError("cylinder_lattice produced no particles; shrink voxel_size or grow the fill volume.")
    return points.astype(np.float32, copy=False), p_radius, p_mass


@dataclass
class LatteSceneCfg(BaseCfg):
    """Liquid, cup, and layout dials. Liquid values are the in-tree MPM pour demo's proven fluid
    recipe; both liquids share one material at v1 (only the color differs)."""

    # --- MPM solver / seeding ---
    voxel_size: float = tunable(0.003)  # [TUNE] MPM grid voxel [m]; finer = crisper liquid, slower
    particles_per_cell: float = tunable(2.0)  # [TUNE] lattice density vs grid (2.0 = demo value)
    # --- liquid material (shared by coffee + milk) ---
    liquid_density: float = tunable(1000.0)
    liquid_viscosity: float = tunable(0.1)  # [TUNE] 0.1 = watery (demo); raise toward 5-50 for syrupy
    liquid_damping: float = tunable(0.02)
    liquid_friction: float = tunable(0.0)
    yield_pressure: float = tunable(1.0e15)  # huge -> never yields as a granular (stays liquid)
    tensile_yield_ratio: float = tunable(5.0)
    # --- coffee mug (textured USD asset; origin at the mug CENTER, handle on -y). The vendored
    # mug_x170.usd is the original mug.usd with a 1.7x scale BAKED INTO THE GEOMETRY (this render
    # stack's Fabric delegate drops USD xform scale ops, so runtime scaling silently no-ops).
    # Baked dimensions: straight cylindrical interior r~0.040, interior floor ~9 mm above the
    # base, rim 0.139 above the base, half-depth 0.0704. The dials below match that bake. ---
    mug_usd: str = info("", doc="'' -> the vendored assets/mug/mug_x170.usd (meters, center origin)")
    mug_scale: float = info(1.0, doc="extra runtime scale — WARNING: dropped by the Fabric renderer; bake instead")
    coffee_cup_r: float = tunable(0.039)  # [TUNE] collider/fill/metric radius: baked mug cavity - 1 mm
    coffee_cup_h: float = tunable(0.139)  # rim height above the table (trajectory anchor)
    coffee_floor_z: float = tunable(0.009)  # interior floor height above the table
    # --- milk cup (procedural open cylinder; local origin at outside bottom center) ---
    milk_cup_r: float = tunable(0.030)
    milk_cup_h: float = tunable(0.075)
    cup_wall: float = tunable(0.006)  # [TUNE] >= ~2 voxels or particles tunnel the wall
    cup_bottom: float = tunable(0.007)
    cup_friction: float = tunable(0.05)  # low, like the demo bowl — liquid slides off ceramic
    cup_contact_margin: float = tunable(0.001)
    # --- fills ---
    coffee_depth: float = tunable(0.025)  # [TUNE] ~118 ml at r=0.04 -> ~35k particles at defaults
    milk_depth: float = tunable(0.020)  # [TUNE] ~57 ml at r=0.03 -> ~17k particles at defaults
    # --- layout ---
    milk_cup_pos: tuple[float, float] = info((0.14, 0.0), doc="milk cup center xy [m]; coffee cup is at (0,0)")
    table_size: tuple[float, float, float] = info((0.7, 0.7, TABLE_TOP_Z), doc="table box extents [m]; top at z=0.04")
    table_friction: float = tunable(0.5)
    light_intensity: float = tunable(2500.0)
    # --- rendering ---
    coffee_color: tuple[float, float, float] = info((0.36, 0.22, 0.12), doc="coffee particle display color")
    milk_color: tuple[float, float, float] = info((0.93, 0.90, 0.85), doc="milk particle display color")
    visual_update_frequency: int = info(4, doc="Kit particle visual update period [render frames]")
    visual_width_scale: float = tunable(2.2)  # [TUNE] Kit display width vs physical particle diameter:
    # at 1x the ~1.4 mm particles read as sparse mist; ~2.2x closes the lattice gaps so the surface
    # reads as liquid. Keep scaled width < cup_wall or particles bulge through the cup exterior.

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets" / "mug"
        self.mug_usd = self.mug_usd or str(assets / "mug_x170.usd")


@SCENES.register("latte")
class LatteScene(BaseScene):
    """Coffee cup (brown MPM liquid) + kinematic milk cup (white MPM liquid) on a table. Goal:
    pour the milk into the coffee cup. Handles after bind: `self.coffee` / `self.milk`
    (MPMObjects) and `self.milk_cup` (kinematic RigidObject); `transfer_fraction()` is the
    success proxy."""

    cfg: LatteSceneCfg

    def __init__(self, cfg: LatteSceneCfg | None = None) -> None:
        super().__init__(cfg or LatteSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        from collections.abc import Callable
        from dataclasses import MISSING

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils.configclass import configclass
        from isaaclab_newton.assets.mpm_object import MPMObjectCfg
        from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg

        c = self.cfg

        @configclass
        class CupMeshCfg(sim_utils.MeshCfg):
            """Suite-local arbitrary-trimesh spawner (exact collider, optional rigid body)."""

            func: Callable | str = clone(_spawn_cup_mesh)
            vertices: list[list[float]] = MISSING
            faces: list[list[int]] = MISSING
            mesh_collision_props: sim_utils.NewtonMeshCollisionPropertiesCfg | None = None

        @configclass
        class VisualUsdRefCfg(sim_utils.SpawnerCfg):
            """Reference a USD asset as VISUAL-ONLY set dressing: wrapped under our own Xform so
            `scale` reliably applies (UsdFileCfg's scale is silently skipped when the asset root
            carries its own xform ops, as this mug does), with every collision/rigid-body API on
            the referenced prims force-disabled."""

            func: Callable | str = clone(_spawn_visual_usd_ref)
            usd_path: str = MISSING
            scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

        def cup_spawn(
            r_inner: float,
            height: float,
            kinematic: bool,
            color: tuple | None,
            wall: float | None = None,
            bottom: float | None = None,
            visible: bool = True,
        ) -> CupMeshCfg:
            vertices, faces = cup_mesh(r_inner, height, wall or c.cup_wall, bottom or c.cup_bottom)
            return CupMeshCfg(
                visible=visible,
                vertices=vertices.tolist(),
                faces=faces.tolist(),
                rigid_props=(
                    sim_utils.NewtonRigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True
                    )
                    if kinematic
                    else None
                ),
                collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                    collision_enabled=True, contact_margin=c.cup_contact_margin
                ),
                mesh_collision_props=sim_utils.NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="none"),
                physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=c.cup_friction, dynamic_friction=c.cup_friction
                ),
                physics_material_path="physicsMaterial",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color) if color is not None else None,
                visual_material_path="visualMaterial",
            )

        def liquid(
            cup_r: float, depth: float, color: tuple, cup_xy: tuple[float, float], seed: int, z_lo: float
        ) -> MPMObjectCfg:
            fill_r = cup_r - 2.0 * c.voxel_size / c.particles_per_cell  # stay off the wall
            points, p_radius, p_mass = cylinder_lattice(
                fill_r, z_lo, z_lo + depth, c.voxel_size, c.particles_per_cell, c.liquid_density, seed
            )
            return MPMObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + ("Coffee" if seed == 0 else "Milk"),
                spawn=MPMPointsCfg(
                    positions=points.tolist(),
                    mass=p_mass,
                    radius=p_radius,
                    material=MPMParticleMaterialCfg(
                        density=c.liquid_density,
                        viscosity=c.liquid_viscosity,
                        friction=c.liquid_friction,
                        damping=c.liquid_damping,
                        yield_pressure=c.yield_pressure,
                        tensile_yield_ratio=c.tensile_yield_ratio,
                    ),
                    visual_color=color,
                    visual_update_frequency=c.visual_update_frequency,
                ),
                init_state=MPMObjectCfg.InitialStateCfg(pos=(cup_xy[0], cup_xy[1], TABLE_TOP_Z)),
            )

        mx, my = c.milk_cup_pos
        return {
            "ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.85, 0.85, 0.85)),
            ),
            "table": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z - c.table_size[2] / 2)),
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size,
                    collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                        collision_enabled=True, contact_margin=0.0003
                    ),
                    physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                        static_friction=c.table_friction, dynamic_friction=c.table_friction
                    ),
                    physics_material_path="physicsMaterial",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.45, 0.4)),
                    visual_material_path="visualMaterial",
                ),
            ),
            # Textured mug asset, VISUAL-ONLY (origin at the mug center, half-depth 0.0414 m at
            # scale 1, handle on -y). Its baked convex-decomposition collision is authored for
            # grasping, not containment — the wall-box junctions leak MPM particles and the floor
            # piece protrudes beyond the walls (measured: 37% of the coffee ends up pooled on the
            # protruding slab). Collision and rigid-body APIs are disabled by the spawner; the
            # invisible procedural cup below is the actual collider.
            "coffee_cup": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/CoffeeCup",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z + 0.0704 * c.mug_scale)),
                spawn=VisualUsdRefCfg(usd_path=c.mug_usd, scale=(c.mug_scale, c.mug_scale, c.mug_scale)),
            ),
            # Watertight collider matched to the mug cavity: inner radius = coffee_cup_r, rim =
            # coffee_cup_h, floor top = coffee_floor_z (liquid rests at the mug's visual floor),
            # outer wall 0.045 m < the mug's 0.050 m outer wall, so it stays hidden inside.
            "coffee_cup_collider": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/CoffeeCupCollider",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z)),
                spawn=cup_spawn(
                    c.coffee_cup_r,
                    c.coffee_cup_h,
                    kinematic=False,
                    color=None,
                    wall=0.005,
                    bottom=c.coffee_floor_z,
                    visible=False,
                ),
            ),
            "milk_cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MilkCup",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(mx, my, TABLE_TOP_Z)),
                spawn=cup_spawn(c.milk_cup_r, c.milk_cup_h, kinematic=True, color=(0.72, 0.72, 0.75)),
            ),
            "coffee": liquid(
                c.coffee_cup_r, c.coffee_depth, c.coffee_color, (0.0, 0.0), seed=0, z_lo=c.coffee_floor_z + 0.004
            ),
            "milk": liquid(c.milk_cup_r, c.milk_depth, c.milk_color, (mx, my), seed=1, z_lo=c.cup_bottom + 0.004),
        }

    def sim_cfg(self) -> MpmSimCfg:
        return MpmSimCfg(voxel_size=self.cfg.voxel_size)

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        import torch

        super().bind(env)
        self.coffee = env.iscene["coffee"]
        self.milk = env.iscene["milk"]
        self.milk_cup = env.iscene["milk_cup"]
        # Snapshot the spawn state as the reset target (liquids seeded in their cups, at rest).
        self._default_state = {
            name: (obj.data.nodal_pos_w.torch.clone(), obj.data.nodal_vel_w.torch.clone().zero_())
            for name, obj in (("coffee", self.coffee), ("milk", self.milk))
        }
        # Constructed (not read back) so we only depend on the proven write API: cfg spawn pose,
        # world frame = env origin + local, identity quat in xyzw (develop convention).
        origins = env.iscene.env_origins
        local = torch.tensor([*self.cfg.milk_cup_pos, TABLE_TOP_Z], device=origins.device)
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=origins.device).expand(origins.shape[0], 4)
        self._default_cup_pose = torch.cat([origins + local, quat], dim=-1)
        self._fabric_particle_attrs: list[tuple[Any, Any]] = []

    # ----- Kit particle visuals ---------------------------------------------------------------
    # The Newton backend creates a UsdGeom.Points prim per MPM object for Kit rendering, but its
    # per-frame position sync writes the plain USD layer, which the Fabric scene delegate (active
    # in the headless-rendering/recording experience) ignores — the liquids render frozen at
    # their spawn state, hidden inside the cups. Verified fix: push positions through usdrt
    # (Fabric) ourselves. Display-only; physics is untouched.
    def setup_particle_visuals(self) -> None:
        """Prepare Kit particle rendering: scale display widths by `cfg.visual_width_scale`
        (at 1x the ~1.4 mm points read as sparse mist) and attach the Fabric stage for
        `push_particle_visuals`. Idempotent; silent no-op without the kit visualizer."""
        if self._fabric_particle_attrs:
            return
        try:
            import numpy as np

            import isaaclab.sim as sim_utils
            import usdrt
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import UsdGeom, Vt

            stage = sim_utils.get_current_stage()
            if not stage.GetPrimAtPath("/World/Visuals/MPMParticles").IsValid():
                return
            vis_prims = [
                p
                for p in stage.Traverse()
                if p.GetTypeName() == "Points" and str(p.GetPath()).startswith("/World/Visuals/MPMParticles")
            ]
            scale = float(self.cfg.visual_width_scale)
            rt_stage = get_current_stage(fabric=True)
            for prim in vis_prims:
                scaled = np.array(UsdGeom.Points(prim).GetWidthsAttr().Get(), dtype=np.float32) * scale
                # Write widths on BOTH layers: USD for non-Fabric viewers (interactive GUI), and
                # usdrt for the Fabric scene delegate (headless-rendering capture) — plain USD
                # writes are unreliably picked up once the prim is Fabric-resident.
                UsdGeom.Points(prim).GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(scaled))
                path = str(prim.GetPath())
                obj = self.coffee if "Coffee" in path else self.milk if "Milk" in path else None
                rt_prim = rt_stage.GetPrimAtPath(path)
                if obj is not None and rt_prim:
                    try:
                        w_attr = rt_prim.GetAttribute("widths") or rt_prim.CreateAttribute(
                            "widths", usdrt.Sdf.ValueTypeNames.FloatArray, False
                        )
                        w_attr.Set(usdrt.Vt.FloatArray(scaled.reshape(-1, 1)))  # usdrt arrays are 2-D
                    except Exception as e:  # noqa: BLE001
                        print(f"[latte] fabric widths write skipped ({e}); USD-layer widths still set", flush=True)
                    self._fabric_particle_attrs.append((rt_prim.GetAttribute("points"), obj))
            self._usdrt_vt = usdrt.Vt
        except Exception as e:  # noqa: BLE001 — display sugar must never kill a run
            print(f"[latte] Kit particle visual setup skipped: {e}", flush=True)

    def push_particle_visuals(self) -> None:
        """Write current particle positions into Fabric so the Kit render shows live liquid.
        Call at render cadence (every few physics steps); no-op if setup found no prims."""
        import numpy as np

        for attr, obj in self._fabric_particle_attrs:
            pts = obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float32)
            attr.Set(self._usdrt_vt.Vec3fArray(pts))

    def reset(self, env_ids: torch.Tensor) -> None:
        import torch

        for obj, (pos, vel) in ((self.coffee, self._default_state["coffee"]), (self.milk, self._default_state["milk"])):
            obj.write_nodal_pos_to_sim_index(pos[env_ids].contiguous(), env_ids=env_ids)
            obj.write_nodal_velocity_to_sim_index(vel[env_ids].contiguous(), env_ids=env_ids)
        self.milk_cup.write_root_link_pose_to_sim_index(
            root_pose=self._default_cup_pose[env_ids].contiguous(), env_ids=env_ids
        )
        zero_twist = torch.zeros((len(env_ids), 6), device=self._default_cup_pose.device)
        self.milk_cup.write_root_link_velocity_to_sim_index(root_velocity=zero_twist, env_ids=env_ids)

    # ----- state ----------------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        # Liquids only: the milk cup is kinematic — its pose belongs to whoever scripts it.
        return {
            "coffee_pos": self.coffee.data.nodal_pos_w.torch[env_ids].clone(),
            "coffee_vel": self.coffee.data.nodal_vel_w.torch[env_ids].clone(),
            "milk_pos": self.milk.data.nodal_pos_w.torch[env_ids].clone(),
            "milk_vel": self.milk.data.nodal_vel_w.torch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.coffee.write_nodal_pos_to_sim_index(state["coffee_pos"].contiguous(), env_ids=env_ids)
        self.coffee.write_nodal_velocity_to_sim_index(state["coffee_vel"].contiguous(), env_ids=env_ids)
        self.milk.write_nodal_pos_to_sim_index(state["milk_pos"].contiguous(), env_ids=env_ids)
        self.milk.write_nodal_velocity_to_sim_index(state["milk_vel"].contiguous(), env_ids=env_ids)

    # ----- metrics --------------------------------------------------------------------------------
    def _local(self, obj) -> torch.Tensor:
        """Particle positions with the env origin removed (env-local frame): (num_envs, P, 3)."""
        return obj.data.nodal_pos_w.torch - self.env.iscene.env_origins[:, None, :]

    def _in_coffee_cup(self, p: torch.Tensor) -> torch.Tensor:
        """Boolean mask: particles inside the coffee cup's inner cylinder (env-local positions)."""
        c = self.cfg
        r2 = p[..., 0] ** 2 + p[..., 1] ** 2
        return (r2 < c.coffee_cup_r**2) & (p[..., 2] > TABLE_TOP_Z) & (p[..., 2] < TABLE_TOP_Z + c.coffee_cup_h + 0.02)

    def transfer_fraction(self) -> torch.Tensor:
        """Per-env fraction of MILK particles inside the coffee cup — the success proxy."""
        return self._in_coffee_cup(self._local(self.milk)).float().mean(dim=1)

    def retention_fraction(self) -> torch.Tensor:
        """Per-env fraction of COFFEE particles still inside the coffee cup."""
        return self._in_coffee_cup(self._local(self.coffee)).float().mean(dim=1)

    def spilled_fraction(self) -> torch.Tensor:
        """Per-env fraction of MILK particles resting on/below table level outside every cup:
        below `table_top + 1 cm`, not in the coffee cup, and outside the milk cup's home
        footprint (milk still riding in its own cup — home, lifted, or tipped high — never
        counts; a home-footprint mask suffices because away from home the cup is airborne)."""
        c = self.cfg
        p = self._local(self.milk)
        low = p[..., 2] < TABLE_TOP_Z + 0.01
        home_r = c.milk_cup_r + c.cup_wall + 0.01
        home_d2 = (p[..., 0] - c.milk_cup_pos[0]) ** 2 + (p[..., 1] - c.milk_cup_pos[1]) ** 2
        return (low & ~self._in_coffee_cup(p) & (home_d2 > home_r**2)).float().mean(dim=1)

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A ceramic mug (cavity radius {c.coffee_cup_r:.3f} m, rim {c.coffee_cup_h:.3f} m above the table) stands at"
            f" (0, 0) on a table (top at z={TABLE_TOP_Z}) holding brown coffee (liquid particles,"
            f" ~{c.coffee_depth * 1e3:.0f} mm deep). A smaller steel milk cup at"
            f" ({c.milk_cup_pos[0]}, {c.milk_cup_pos[1]}) holds white milk. The milk cup is kinematic: write its"
            " root pose to move it. Goal: pour the milk into the coffee cup — lift the milk cup, carry it over"
            " the coffee cup, and tip it so the milk streams in, without spilling on the table. Success: >= 70%"
            " of milk particles inside the coffee cup, >= 90% of coffee retained, <= 5% of milk spilled."
        )


# ----- suite-local spawners (module level so configclass `func` can reference them) ---------------
def _spawn_visual_usd_ref(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    """Reference `cfg.usd_path` under a fresh Xform we own (translate/orient via create_prim, an
    explicit scale op appended), then disable every rigid-body/collision API inside the reference
    so the asset is pure set dressing."""
    from isaaclab.sim.utils import create_prim, get_current_stage
    from pxr import Usd, UsdPhysics

    stage = get_current_stage()
    root = create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        scale=tuple(float(s) for s in cfg.scale),
        stage=stage,
    )
    asset_prim = stage.DefinePrim(f"{prim_path}/asset")
    asset_prim.GetReferences().AddReference(cfg.usd_path)
    for prim in Usd.PrimRange(asset_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    return root


# ----- suite-local mesh spawner (module level so configclass `func` can reference it) -------------
def _spawn_cup_mesh(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
):
    """Spawn an arbitrary trimesh as a standard Isaac Lab asset (pattern from the in-tree MPM pour
    demo): Xform root + Mesh prim, then collision / rigid-body / material schemas from the cfg."""
    import numpy as np

    from isaaclab.sim import schemas
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material, create_prim, get_current_stage

    stage = get_current_stage()
    vertices = np.asarray(cfg.vertices, dtype=np.float32)
    faces = np.asarray(cfg.faces, dtype=np.int32)

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    if not getattr(cfg, "visible", True):
        from pxr import UsdGeom

        UsdGeom.Imageable(stage.GetPrimAtPath(prim_path)).MakeInvisible()
    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    create_prim(geom_prim_path, prim_type="Xform", stage=stage)
    create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices,
            "faceVertexIndices": faces.reshape(-1),
            "faceVertexCounts": np.full(faces.shape[0], 3, dtype=np.int32),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)
    if cfg.mesh_collision_props is not None:
        schemas.define_mesh_collision_properties(mesh_prim_path, cfg.mesh_collision_props, stage=stage)
    if cfg.visual_material is not None:
        material_path = cfg.visual_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)
    if cfg.physics_material is not None:
        material_path = cfg.physics_material_path
        if not material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{material_path}"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)
