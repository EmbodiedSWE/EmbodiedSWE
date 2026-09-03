"""LatteScene — THE latte benchmark scene (registered `latte`); pour the milk into the coffee.

ONE scene class, one cfg, one registration — every run (agent benchmark, verified Franka
solution, local demonstration scripts) builds this same scene. It composes, in one place:

  - Newton implicit **MPM liquids**: both liquids are `MPMObject`s (particle lattices seeded
    inside the vessels); the vessels' interiors are exact open-cup trimesh colliders.
  - the COUPLED MJWarp+MPM substrate: robots get real dynamics (gravity, actuator PD, MuJoCo
    rigid contacts) while the liquids follow the post-rigid body poses.
  - DYNAMIC vessels: free rigid bodies (authored mass, real gravity) on concave rigid-proxy
    shells (ring + floor slab + handle bar); vessel-table and vessel-vessel contacts are real.
  - the AUTO-WELD grasp contract: move a gripper's pinch point within `auto_weld_dist` of a
    handle bar with the fingers closed to the bar's width and the vessel welds on at the
    measured pose; open past `auto_weld_release` to let go. No scripted attach calls anywhere.
  - 1.5-WAY LIQUID FEEDBACK: MPM collider impulses are applied back onto the rigid bodies, so
    vessels weigh what they hold and lighten as they pour (`cfg.liquid_feedback`; the offline
    replay renderer builds with it OFF — that build path spins under the kit-visualizer app —
    which is scenery-only and never steps physics).

Layout (env-local meters): table top at z=0.04; the coffee mug at (0, 0) pre-filled with brown
"coffee"; the milk pitcher at `pitcher_pos` (default (0.16, 0)) pre-filled with white "milk".
Goal: pour the milk into the coffee — `transfer_fraction()` is the success proxy, with
`retention_fraction()` and `spilled_fraction()` as guards; metrics track the vessels' ACTUAL
poses, so a dropped vessel scores honestly.

Requires the Newton venv (`env_newton`, see the README) to build; single-env only (the MPM
fixed grid spans the whole scene). Heavy imports are deferred so importing stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robobench.core import SCENES, BaseCfg, BaseScene
from robobench.suites.deformable.newton.mpm_sim import MpmSimCfg

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

TABLE_TOP_Z = 0.04  # table top height [m]; cups stand here, layout numbers assume it

# Concave rigid-proxy geometry for the DYNAMIC vessels, in each vessel's local frame (base
# origin, z up), traced from the assets' VISUAL surfaces: a ring of boxes on the outer wall, a
# box floor slab (base-table contact), and the handle's outer bar (the grasp target; also an
# MPM collider so milk cannot pass the visual handle).
MUG_PROXY = {
    "ring_r_out": 0.058,
    "ring_z": (0.012, 0.0832),
    "slab_r": 0.041,
    "slab_h": 0.010,
    # grip_w: bar box width (wider than the visual 2r so finger pads engage with PD headroom)
    "handle": {"x": -0.0877, "z": 0.0515, "r": 0.008, "half_height": 0.0103, "grip_w": 0.022},
}
PITCHER_PROXY = {
    "ring_r_out": 0.0425,
    "ring_z": (0.008, 0.0927),
    "slab_r": 0.041,
    "slab_h": 0.008,
    "handle": {"x": 0.0643, "z": 0.0634, "r": 0.006, "half_height": 0.0161, "grip_w": 0.018},
}


# ----- procedural geometry (numpy only; app-free) -------------------------------------------------
def cup_mesh(
    r_inner_bottom: float, r_inner_top: float, height: float, wall: float, bottom: float, segments: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Open-cup trimesh (optionally tapered): inner floor ring, inner+outer walls, rim, outer
    floor, and center caps. Local origin at the *outside bottom center*, +z up (same topology as
    the in-tree MPM pour demo's catch bowl)."""
    theta = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack([radius * cos_t, radius * sin_t, np.full(segments, z)])

    vertices = np.vstack(
        [
            ring(r_inner_bottom, bottom),  # 0: inner floor
            ring(r_inner_top, height),  # 1: inner rim
            ring(r_inner_top + wall, height),  # 2: outer rim
            ring(r_inner_bottom + wall, 0.0),  # 3: outer floor
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
    radius: float,
    z_lo: float,
    z_hi: float,
    voxel: float,
    particles_per_cell: float,
    density: float,
    seed: int,
    radius_top: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Jittered particle lattice filling a local-space cylinder — or truncated cone when
    `radius_top` differs (axis +z, centered on xy=0). Returns (points, particle_radius,
    particle_mass) — the in-tree pour demo's seeding recipe."""
    r_top = radius if radius_top is None else radius_top
    r_max = max(radius, r_top)
    lo = np.array([-r_max, -r_max, z_lo], dtype=np.float32)
    hi = np.array([r_max, r_max, z_hi], dtype=np.float32)
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
    t = np.clip((points[:, 2] - z_lo) / max(z_hi - z_lo, 1e-9), 0.0, 1.0)
    r_at_z = radius + (r_top - radius) * t
    keep = points[:, 0] ** 2 + points[:, 1] ** 2 < r_at_z**2
    points = points[keep]
    if points.shape[0] == 0:
        raise RuntimeError("cylinder_lattice produced no particles; shrink voxel_size or grow the fill volume.")
    return points.astype(np.float32, copy=False), p_radius, p_mass


@dataclass
class LatteSceneCfg(BaseCfg):
    """Liquid, cup, and layout dials. Liquid values are the in-tree MPM pour demo's proven fluid
    recipe; both liquids share one material at v1 (only the color differs)."""

    # --- MPM solver / seeding ---
    voxel_size: float = 0.003  # [TUNE] MPM grid voxel [m]; finer = crisper liquid, slower
    particles_per_cell: float = 2.0  # [TUNE] lattice density vs grid (2.0 = demo value; LOWER
    # under-resolves the constitutive model — at 1.6 a deep narrow fill collapsed into a sticky blob)
    # --- liquid material (shared by coffee + milk) ---
    liquid_density: float = 1000.0
    liquid_viscosity: float = 3.0  # [TUNE] creamy (steamed-milk-ish): the compacted MPM liquid
    # avalanches out of deep vessels at ~90 deg when watery (0.1) — viscosity 3 makes the outflow a
    # controllable ooze so a partial pour can actually stop
    liquid_damping: float = 0.02
    liquid_friction: float = 0.0
    yield_pressure: float = 1.0e15  # huge -> never yields as a granular (stays liquid)
    tensile_yield_ratio: float = 1.0  # [TUNE] cohesion; 5.0 clings to deep vessel walls and exits late
    # --- coffee mug (textured USD asset, VISUAL-ONLY; the invisible tapered collider below is
    # the physics). The vendored BlackCeramicMug/mug_black_zup.usd is the original model with the
    # fix-up BAKED INTO THE GEOMETRY (this render stack's Fabric delegate drops USD xform
    # scale/orient fix-ups, so runtime transforms can't be trusted): rotated Y-up -> Z-up, body
    # axis centered on the origin, base at z=0. Baked dimensions: tapered interior r 0.034 (near
    # floor) -> 0.046 (rim), interior floor ~16 mm above the base, rim at 0.0832, handle on -x.
    # The dials below match that bake. ---
    mug_usd: str = ""  # '' -> the vendored assets/BlackCeramicMug/mug_black_zup.usd (base origin)
    mug_scale: float = 1.0  # extra runtime scale — WARNING: dropped by the Fabric renderer; bake instead
    coffee_cup_r: float = 0.045  # [TUNE] collider/metric radius at the RIM (mug cavity - 1 mm)
    coffee_cup_r_floor: float = 0.034  # [TUNE] collider radius at the FLOOR (tapered interior)
    coffee_cup_h: float = 0.0832  # rim height above the table (trajectory anchor)
    coffee_floor_z: float = 0.016  # interior floor height above the table
    # --- milk pitcher (textured USD asset, VISUAL-ONLY; the invisible straight collider below is
    # the physics). assets/Pitcher/pitcher_zup.usd is the original model with the composed
    # transform baked into the geometry: base at z=0, body axis centered, HANDLE toward +x (the
    # grasp side — the plain rim pours toward -x / the mug). Measured: body outer 0.040 -> 0.036,
    # interior ~straight r 0.030 above a thick base (usable floor at z ~0.030), rim at 0.0927,
    # wall ~5 mm, handle bar out to x=0.068 spanning z 0.027..0.081. ---
    pitcher_usd: str = ""  # '' -> the vendored assets/Pitcher/pitcher_zup.usd (base origin)
    pitcher_r: float = 0.030  # [TUNE] collider/fill/metric radius (interior cavity - margin)
    pitcher_h: float = 0.0927  # rim height above the base (trajectory lip anchor)
    pitcher_floor_z: float = 0.030  # interior floor height above the base (thick bottom)
    pitcher_wall: float = 0.005  # collider wall; outer 0.035 hides inside the visual body
    cup_friction: float = 0.05  # low, like the demo bowl — liquid slides off ceramic
    cup_contact_margin: float = 0.001
    # --- fills ---
    coffee_depth: float = 0.048  # [TUNE] SEEDED depth; implicit MPM settles ~x0.53 of seeded,
    # leaving the mug roughly half full (surface ~0.044 of the 0.083 rim; ~66k particles)
    milk_depth: float = 0.042  # [TUNE] SEEDED depth; settles to ~60% of the pitcher (~28k particles)
    # --- layout ---
    pitcher_pos: tuple[float, float] = (0.16, 0.0)  # pitcher center xy [m]; coffee mug is at (0,0)
    table_size: tuple[float, float, float] = (0.7, 0.7, TABLE_TOP_Z)  # table box extents [m]; top at z=0.04
    table_friction: float = 0.5
    light_intensity: float = 2500.0
    # --- dynamic vessels + rigid proxies ---
    mug_mass: float = 0.30  # authored total mass [kg]; inertia computed from geometry
    pitcher_mass: float = 0.25
    proxy_segments: int = 10  # boxes per rigid-proxy ring
    proxy_thickness: float = 0.005  # ring box radial thickness [m]
    # 1.5-way liquid feedback (vessels weigh what they hold). ALWAYS on for physics runs; the
    # OFFLINE replay renderer builds scenery with it off (that build spins under the
    # kit-visualizer app) — scenery builds never step physics, so nothing behavioral differs.
    liquid_feedback: bool = True  # apply MPM collider impulses back onto the vessels
    # --- agent auto-grasp: weld engages on proximity + closure ---
    auto_weld_dist: float = 0.03  # pinch-point-to-bar-center engage radius [m]
    auto_weld_close_margin: float = 0.003  # engage when aperture < bar half-width + this [m]
    auto_weld_release: float = 0.02  # release when aperture opens past this [m] (hysteresis)
    proxy_friction: float = 0.5  # ring + slab (MuJoCo-facing) tabletop friction; the
    # handle bars keep cup_friction (MPM-facing)
    # --- success gates: the four the suite's own verified smoke asserts (it prints them, and
    # passes at transfer 0.304 / kept 0.696 / retention 1.000 / spilled 0.000). The task is
    # "fill the mug WITHOUT emptying the pitcher", so a partial pour is the intended outcome and
    # both sides are gated. NOTE: `describe()` still tells the agent ">= 70% transferred", which
    # this scene's pour design never produces — that text is stale.
    success_transfer_min: float = 0.15  # min fraction of MILK inside the coffee cup
    success_kept_min: float = 0.15  # min fraction of MILK still in the pitcher
    success_retention_min: float = 0.90  # min fraction of COFFEE still in its cup
    success_spilled_max: float = 0.05  # max fraction of MILK spilled on the table
    # --- rendering ---
    coffee_color: tuple[float, float, float] = (0.36, 0.22, 0.12)  # coffee particle display color
    milk_color: tuple[float, float, float] = (0.93, 0.90, 0.85)  # milk particle display color
    visual_update_frequency: int = 4  # Kit particle visual update period [render frames]
    visual_width_scale: float = 2.2  # [TUNE] Kit display width vs physical particle diameter:
    # at 1x the ~1.4 mm particles read as sparse mist; ~2.2x closes the lattice gaps so the surface
    # reads as liquid. Keep scaled width < cup_wall or particles bulge through the cup exterior.

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.mug_usd = self.mug_usd or str(assets / "BlackCeramicMug" / "mug_black_zup.usd")
        self.pitcher_usd = self.pitcher_usd or str(assets / "Pitcher" / "pitcher_zup.usd")


@SCENES.register("latte")
class LatteScene(BaseScene):
    """THE latte benchmark scene — see the module docstring for the full mechanic stack (MPM
    liquids, coupled substrate, dynamic vessels, auto-weld grasping, liquid feedback). Handles
    after bind: `self.coffee` / `self.milk` (MPMObjects) and `self.pitcher` / `self.mug`
    (dynamic RigidObjects); `transfer_fraction()` is the success proxy. The ONLY scene in the
    pouring suite: the benchmark env and every demonstration script build it, so exactly one
    scene cfg is ever in play."""

    # (hand prim, vessel) -> weld label; all four combinations exist in the model.
    AUTO_PAIRS = {
        ("Left", "mug"): "weld_mug",
        ("Right", "pitcher"): "weld_pitcher",
        ("Right", "mug"): "weld_mug_r",
        ("Left", "pitcher"): "weld_pitcher_l",
    }

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
            """Suite-local arbitrary-trimesh spawner: exact collider, optional rigid body, and an
            optional VISUAL-ONLY referenced USD child (its physics APIs force-disabled) — so a
            kinematic rigid cup can carry a pretty asset that rides its pose in the renderer."""

            func: Callable | str = clone(_spawn_cup_mesh)
            vertices: list[list[float]] = MISSING
            faces: list[list[int]] = MISSING
            mesh_collision_props: sim_utils.NewtonMeshCollisionPropertiesCfg | None = None
            visual_usd_ref: str | None = None
            # Hide the collider mesh while keeping the ROOT visible (SpawnerCfg.visible=False
            # would hide the whole subtree, including visual_usd_ref — the framework applies it
            # to the spawned root).
            hide_collider_geometry: bool = False
            # Dynamic-vessel extras: authored total mass (UsdPhysics MassAPI on the root) and
            # the rigid-proxy geometry dict, spawned invisible under <root>/rigidproxy/ and
            # flag-routed by the coupled manager.
            mass: float | None = None
            rigidproxy: dict | None = None
            proxy_physics_material: sim_utils.NewtonMaterialPropertiesCfg | None = None

        def cup_spawn(
            r_inner: float,
            height: float,
            color: tuple | None,
            wall: float,
            bottom: float,
            visible: bool = True,
            r_inner_top: float | None = None,
            visual_usd_ref: str | None = None,
            mass: float | None = None,
            proxy: dict | None = None,
        ) -> CupMeshCfg:
            vertices, faces = cup_mesh(
                r_inner, r_inner_top if r_inner_top is not None else r_inner, height, wall, bottom
            )
            # Free-joint vessel under real gravity, carried by the auto-welds.
            rigid_props = sim_utils.NewtonRigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=False, disable_gravity=False
            )
            return CupMeshCfg(
                hide_collider_geometry=not visible,
                visual_usd_ref=visual_usd_ref,
                mass=mass,
                rigidproxy=proxy,
                proxy_physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                    static_friction=c.proxy_friction, dynamic_friction=c.proxy_friction
                ),
                vertices=vertices.tolist(),
                faces=faces.tolist(),
                rigid_props=rigid_props,
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
            cup_r: float,
            depth: float,
            color: tuple,
            cup_xy: tuple[float, float],
            seed: int,
            z_lo: float,
            cup_r_top: float | None = None,
        ) -> MPMObjectCfg:
            margin = 2.0 * c.voxel_size / c.particles_per_cell  # stay off the wall
            points, p_radius, p_mass = cylinder_lattice(
                cup_r - margin,
                z_lo,
                z_lo + depth,
                c.voxel_size,
                c.particles_per_cell,
                c.liquid_density,
                seed,
                radius_top=None if cup_r_top is None else cup_r_top - margin,
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

        px, py = c.pitcher_pos
        return {
            # The visual ground plane is SUNK to z=-1.05 (folding-suite landmine: a ground-plane
            # collider at exactly z=0 goes haywire in the Newton->MuJoCo conversion under the
            # coupled substrate — phantom kN*m contact forces on the arm joints). The invisible
            # static Floor box below carries the actual z=0 surface for spilled MPM particles, so
            # the MPM-only substrate sees identical physics.
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
                spawn=sim_utils.GroundPlaneCfg(),
            ),
            "floor": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Floor",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
                spawn=sim_utils.CuboidCfg(
                    size=(2.0, 2.0, 0.02),
                    collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                        collision_enabled=True, contact_margin=0.0003
                    ),
                    physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                        static_friction=c.table_friction, dynamic_friction=c.table_friction
                    ),
                    physics_material_path="physicsMaterial",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
                    visual_material_path="visualMaterial",
                ),
            ),
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
            # The coffee mug: ONE dynamic rigid body carrying (a) the invisible watertight
            # tapered collider matched to the visual mug's cavity (its baked convex-decomposition
            # collision leaks MPM particles, so it is never used), (b) the concave rigid-proxy
            # shell (MuJoCo-facing), and (c) the textured mug USD referenced as a visual-only
            # child (physics APIs disabled) that rides the body's pose in the renderer.
            "coffee_cup": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CoffeeCup",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_TOP_Z)),
                spawn=cup_spawn(
                    c.coffee_cup_r_floor,
                    c.coffee_cup_h,
                    color=None,
                    wall=0.005,
                    bottom=c.coffee_floor_z,
                    visible=False,
                    r_inner_top=c.coffee_cup_r,
                    visual_usd_ref=c.mug_usd,
                    mass=c.mug_mass,
                    proxy={**MUG_PROXY, "segments": c.proxy_segments, "thickness": c.proxy_thickness},
                ),
            ),
            # The milk pitcher: same pattern as the mug (handle toward +x, the grasp side).
            "pitcher": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pitcher",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, TABLE_TOP_Z)),
                spawn=cup_spawn(
                    c.pitcher_r,
                    c.pitcher_h,
                    color=None,
                    wall=c.pitcher_wall,
                    bottom=c.pitcher_floor_z,
                    visible=False,
                    visual_usd_ref=c.pitcher_usd,
                    mass=c.pitcher_mass,
                    proxy={**PITCHER_PROXY, "segments": c.proxy_segments, "thickness": c.proxy_thickness},
                ),
            ),
            "coffee": liquid(
                c.coffee_cup_r_floor,
                c.coffee_depth,
                c.coffee_color,
                (0.0, 0.0),
                seed=0,
                z_lo=c.coffee_floor_z + 0.004,
                # collider inner radius at the fill's top (linear taper floor -> rim)
                cup_r_top=c.coffee_cup_r_floor
                + (c.coffee_cup_r - c.coffee_cup_r_floor)
                * (0.004 + c.coffee_depth)
                / (c.coffee_cup_h - c.coffee_floor_z),
            ),
            "milk": liquid(c.pitcher_r, c.milk_depth, c.milk_color, (px, py), seed=1, z_lo=c.pitcher_floor_z + 0.004),
        }

    def sim_cfg(self) -> MpmSimCfg:
        return MpmSimCfg(
            voxel_size=self.cfg.voxel_size,
            coupled=True,
            liquid_feedback=self.cfg.liquid_feedback,
            # Builder-time weld rows (disabled until grasp): (label, body1 suffix, body2 suffix).
            welds=[
                ("weld_mug", "Left/panda_hand", "CoffeeCup"),
                ("weld_pitcher", "Right/panda_hand", "Pitcher"),
                ("weld_mug_r", "Right/panda_hand", "CoffeeCup"),
                ("weld_pitcher_l", "Left/panda_hand", "Pitcher"),
            ],
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        import torch

        super().bind(env)
        self.coffee = env.iscene["coffee"]
        self.milk = env.iscene["milk"]
        self.pitcher = env.iscene["pitcher"]
        self.mug = env.iscene["coffee_cup"]
        # Snapshot the spawn state as the reset target (liquids seeded in their cups, at rest).
        self._default_state = {
            name: (obj.data.nodal_pos_w.torch.clone(), obj.data.nodal_vel_w.torch.clone().zero_())
            for name, obj in (("coffee", self.coffee), ("milk", self.milk))
        }
        # Constructed (not read back) so we only depend on the proven write API: cfg spawn pose,
        # world frame = env origin + local, identity quat in xyzw (develop convention).
        origins = env.iscene.env_origins
        local = torch.tensor([*self.cfg.pitcher_pos, TABLE_TOP_Z], device=origins.device)
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=origins.device).expand(origins.shape[0], 4)
        self._default_pitcher_pose = torch.cat([origins + local, quat], dim=-1)
        mug_local = torch.tensor([0.0, 0.0, TABLE_TOP_Z], device=origins.device)
        self._default_mug_pose = torch.cat([origins + mug_local, quat], dim=-1)
        # Latest commanded vessel poses (world) — metrics are computed relative to them, so
        # lifted / tilted carries keep honest numbers. Kinematic: the scripts own them.
        self.mug_pose_w = self._default_mug_pose.clone()
        self.pitcher_pose_w = self._default_pitcher_pose.clone()
        self._fabric_particle_attrs: list[tuple[Any, Any]] = []
        self._auto_state: dict[str, bool] = {}
        self._auto_ready = False
        self._auto_dead = False

    # ----- auto-weld grasp contract + actual-pose tracking ----------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Track the DYNAMIC vessels' actual poses for the pose-relative metrics, then run the
        auto-weld grasp state machine.

        NOTE: root_link_quat_w of a free trimesh body carries a per-body YAW offset vs the prim
        frame; the cylinder metrics and tilt readouts are yaw-invariant, but never mix these
        poses with prim-frame scripted targets."""
        import torch

        self.mug_pose_w = torch.cat(
            [self.mug.data.root_link_pos_w.torch, self.mug.data.root_link_quat_w.torch], dim=-1
        )
        self.pitcher_pose_w = torch.cat(
            [self.pitcher.data.root_link_pos_w.torch, self.pitcher.data.root_link_quat_w.torch], dim=-1
        )
        if self._auto_dead:
            return
        try:
            if not self._auto_ready:
                self._auto_setup()
            self._auto_tick()
        except Exception as e:  # noqa: BLE001 — a broken grasp mechanic must be loud, not fatal
            print(f"[auto-weld] DISABLED after error: {e!r}", flush=True)
            self._auto_dead = True

    def _auto_setup(self) -> None:
        """Resolve body indices, handle-bar local centers (mean of the bar segments'
        shape_transforms — composed with body_q per tick, so frame conventions cancel), finger
        joints, and per-vessel bar half-widths. Runs once, lazily (the model exists post-build)."""
        import numpy as np
        import torch

        from robobench.suites.deformable.newton.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

        model = Mgr._model
        device = self.env.device
        body_labels = [str(b or "") for b in model.body_label]

        def body_idx(suffix: str) -> int:
            matches = [i for i, b in enumerate(body_labels) if b.endswith(suffix)]
            assert len(matches) == 1, (suffix, matches)
            return matches[0]

        shape_labels = [str(s or "") for s in model.shape_label]
        shape_tf = model.shape_transform.numpy()
        shape_body = model.shape_body.numpy()
        self._auto_vessels: dict[str, tuple] = {}
        for vessel, suffix, proxy in (("mug", "CoffeeCup", MUG_PROXY), ("pitcher", "Pitcher", PITCHER_PROXY)):
            b = body_idx(suffix)
            segs = [i for i, s in enumerate(shape_labels) if "/rigidproxy/handle" in s and int(shape_body[i]) == b]
            assert segs, f"no handle segments found for {vessel}"
            bar_local = torch.tensor(
                np.stack([shape_tf[i][:3] for i in segs]).mean(axis=0), device=device, dtype=torch.float32
            )
            half_width = float(proxy["handle"].get("grip_w", 2.0 * proxy["handle"]["r"])) / 2.0
            self._auto_vessels[vessel] = (b, bar_local, half_width)
        self._auto_hands: dict[str, tuple] = {}
        for name, robot in self.env.robot.robots.items():
            prim = name[:1].upper() + name[1:]
            art = robot.articulation
            self._auto_hands[prim] = (body_idx(f"{prim}/panda_hand"), art, art.find_joints(["panda_finger.*"])[0])
        self._auto_ready = True

    def _auto_tick(self) -> None:
        import warp as wp
        import torch

        import isaaclab.utils.math as math_utils

        from robobench.suites.deformable.newton.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

        body_q = wp.to_torch(Mgr._state_0.body_q)
        held_vessels = {v for (h, v), lbl in self.AUTO_PAIRS.items() if self._auto_state.get(lbl)}
        busy_hands = {h for (h, v), lbl in self.AUTO_PAIRS.items() if self._auto_state.get(lbl)}
        tip_local = torch.tensor([[0.0, 0.0, 0.113]], device=body_q.device)
        for (hand, vessel), label in self.AUTO_PAIRS.items():
            hb, art, fids = self._auto_hands[hand]
            aperture = float(art.data.joint_pos.torch[0, fids].mean())
            if self._auto_state.get(label, False):
                if aperture > self.cfg.auto_weld_release:
                    Mgr.set_weld(label, False)
                    self._auto_state[label] = False
                    print(f"  [auto-weld] {hand} RELEASED the {vessel} (aperture {aperture * 1000:.1f} mm)", flush=True)
                continue
            if hand in busy_hands or vessel in held_vessels:
                continue
            vb, bar_local, half_width = self._auto_vessels[vessel]
            if aperture >= half_width + self.cfg.auto_weld_close_margin:
                continue  # fingers not squeezing — cheap early-out before any pose math
            pinch = body_q[hb, :3] + math_utils.quat_apply(body_q[hb, 3:][None], tip_local)[0]
            bar_w = body_q[vb, :3] + math_utils.quat_apply(body_q[vb, 3:][None], bar_local[None])[0]
            dist = float((pinch - bar_w).norm())
            if dist < self.cfg.auto_weld_dist:
                Mgr.set_weld(label, True)
                self._auto_state[label] = True
                busy_hands.add(hand)
                held_vessels.add(vessel)
                print(
                    f"  [auto-weld] {hand} GRIPPED the {vessel} (dist {dist * 100:.1f} cm,"
                    f" aperture {aperture * 1000:.1f} mm)",
                    flush=True,
                )

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

        # Welds off FIRST: the vessel teleports home while the hands still hold their last pose —
        # an active weld would read that as a violent constraint violation.
        from robobench.suites.deformable.newton.coupled_manager import NewtonCoupledMJWarpMPMManager as Mgr

        for label in self.AUTO_PAIRS.values():
            Mgr.set_weld(label, False)
        self._auto_state = {}
        for obj, (pos, vel) in ((self.coffee, self._default_state["coffee"]), (self.milk, self._default_state["milk"])):
            obj.write_nodal_pos_to_sim_index(pos[env_ids].contiguous(), env_ids=env_ids)
            obj.write_nodal_velocity_to_sim_index(vel[env_ids].contiguous(), env_ids=env_ids)
        self.pitcher.write_root_link_pose_to_sim_index(
            root_pose=self._default_pitcher_pose[env_ids].contiguous(), env_ids=env_ids
        )
        zero_twist = torch.zeros((len(env_ids), 6), device=self._default_pitcher_pose.device)
        self.pitcher.write_root_link_velocity_to_sim_index(root_velocity=zero_twist, env_ids=env_ids)
        self.mug.write_root_link_pose_to_sim_index(
            root_pose=self._default_mug_pose[env_ids].contiguous(), env_ids=env_ids
        )
        self.mug.write_root_link_velocity_to_sim_index(root_velocity=zero_twist, env_ids=env_ids)
        self.mug_pose_w = self._default_mug_pose.clone()
        self.pitcher_pose_w = self._default_pitcher_pose.clone()

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
        """Boolean mask: particles inside the mug's inner cylinder, in the mug's CURRENT frame
        (position AND orientation — honest under a tilted carry)."""
        import isaaclab.utils.math as math_utils

        c = self.cfg
        origins = self.env.iscene.env_origins
        mug = self.mug_pose_w[:, :3] - origins  # env-local mug base center
        d = p - mug[:, None, :]
        quat = self.mug_pose_w[:, 3:].unsqueeze(1).expand(-1, d.shape[1], 4)
        d = math_utils.quat_apply_inverse(quat.reshape(-1, 4), d.reshape(-1, 3)).reshape(d.shape)
        r2 = d[..., 0] ** 2 + d[..., 1] ** 2
        return (r2 < c.coffee_cup_r**2) & (d[..., 2] > 0.0) & (d[..., 2] < c.coffee_cup_h + 0.02)

    def transfer_fraction(self) -> torch.Tensor:
        """Per-env fraction of MILK particles inside the coffee cup — the success proxy."""
        return self._in_coffee_cup(self._local(self.milk)).float().mean(dim=1)

    def retention_fraction(self) -> torch.Tensor:
        """Per-env fraction of COFFEE particles still inside the coffee cup."""
        return self._in_coffee_cup(self._local(self.coffee)).float().mean(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: the mug was filled without emptying the pitcher — the four gates the
        suite's verified smoke asserts (milk transferred, milk kept back, coffee retained,
        nothing spilled), all met at once (scene-level success alias, matching the other
        suites' surface)."""
        c = self.cfg
        return (
            (self.transfer_fraction() >= c.success_transfer_min)
            & (self.milk_in_pitcher_fraction() >= c.success_kept_min)
            & (self.retention_fraction() >= c.success_retention_min)
            & (self.spilled_fraction() <= c.success_spilled_max)
        )

    def spilled_fraction(self) -> torch.Tensor:
        """Per-env fraction of MILK particles resting on/below table level outside every cup:
        below `table_top + 1 cm`, not in the coffee cup, and outside the milk cup's home
        footprint (milk still riding in its own cup — home, lifted, or tipped high — never
        counts; a home-footprint mask suffices because away from home the cup is airborne)."""
        c = self.cfg
        p = self._local(self.milk)
        low = p[..., 2] < TABLE_TOP_Z + 0.01
        home_r = c.pitcher_r + c.pitcher_wall + 0.01
        home_d2 = (p[..., 0] - c.pitcher_pos[0]) ** 2 + (p[..., 1] - c.pitcher_pos[1]) ** 2
        # A lifted mug can't shelter table-level particles, so also excluding the mug's home
        # footprint keeps the metric honest whether or not the mug was carried.
        mug_d2 = p[..., 0] ** 2 + p[..., 1] ** 2
        mug_r = c.coffee_cup_r + 0.02
        return (low & ~self._in_coffee_cup(p) & (home_d2 > home_r**2) & (mug_d2 > mug_r**2)).float().mean(dim=1)

    def milk_in_pitcher_fraction(self) -> torch.Tensor:
        """Per-env fraction of MILK still inside the pitcher (pitcher-frame cylinder) — the
        'did not empty the pitcher' gate for the fill-to-target pour."""
        import isaaclab.utils.math as math_utils

        c = self.cfg
        origins = self.env.iscene.env_origins
        p = self._local(self.milk)
        d = p - (self.pitcher_pose_w[:, :3] - origins)[:, None, :]
        quat = self.pitcher_pose_w[:, 3:].unsqueeze(1).expand(-1, d.shape[1], 4)
        d = math_utils.quat_apply_inverse(quat.reshape(-1, 4), d.reshape(-1, 3)).reshape(d.shape)
        r2 = d[..., 0] ** 2 + d[..., 1] ** 2
        inside = (r2 < c.pitcher_r**2) & (d[..., 2] > 0.0) & (d[..., 2] < c.pitcher_h + 0.02)
        return inside.float().mean(dim=1)

    def mug_surface_z(self) -> float:
        """SETTLED liquid surface height inside the mug, in the MUG frame [m above the mug base]:
        the 0.9-quantile z of in-mug particles (coffee + milk) that are near rest (|v| < 0.25 m/s
        — a falling pour stream inside the mug cylinder would otherwise inflate the estimate).
        -inf when the mug is empty. Env 0 only (the pour scripts are single-env)."""
        import isaaclab.utils.math as math_utils

        origins = self.env.iscene.env_origins
        zs = []
        for obj in (self.coffee, self.milk):
            p = obj.data.nodal_pos_w.torch - origins[:, None, :]
            d = p - (self.mug_pose_w[:, :3] - origins)[:, None, :]
            quat = self.mug_pose_w[:, 3:].unsqueeze(1).expand(-1, d.shape[1], 4)
            d = math_utils.quat_apply_inverse(quat.reshape(-1, 4), d.reshape(-1, 3)).reshape(d.shape)
            r2 = d[..., 0] ** 2 + d[..., 1] ** 2
            settled = obj.data.nodal_vel_w.torch.norm(dim=-1) < 0.25
            m = (
                (r2 < self.cfg.coffee_cup_r**2)
                & (d[..., 2] > 0.0)
                & (d[..., 2] < self.cfg.coffee_cup_h + 0.02)
                & settled
            )
            if bool(m[0].any()):
                zs.append(d[0, m[0], 2])
        if not zs:
            return float("-inf")
        import torch

        return float(torch.quantile(torch.cat(zs), 0.9))

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A black ceramic mug (cavity radius {c.coffee_cup_r:.3f} m at the rim, {c.coffee_cup_h:.3f} m tall) stands at"
            f" (0, 0) on a table (top at z={TABLE_TOP_Z}) holding brown coffee (liquid particles,"
            f" ~{c.coffee_depth * 1e3:.0f} mm deep). A smaller steel milk pitcher at"
            f" ({c.pitcher_pos[0]}, {c.pitcher_pos[1]}) holds white milk. Both vessels are DYNAMIC rigid bodies"
            " resting on the table. GRASPING: move a gripper's pinch point within"
            f" {c.auto_weld_dist * 100:.0f} cm of a handle bar and CLOSE the fingers onto it — the vessel then"
            " attaches rigidly and the arm carries its real mass; OPEN the gripper to release it. Either gripper"
            " can grab either handle. The liquids have real weight: a full vessel is heavier. Goal: pour the"
            " milk into the coffee cup — lift the pitcher, carry it over the coffee cup, and tip it so the milk"
            " streams in, without spilling on the table. Success: >= 70% of milk particles inside the coffee"
            " cup, >= 90% of coffee retained, <= 5% of milk spilled."
        )


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

    if getattr(cfg, "hide_collider_geometry", False):
        # Hide the collider GEOMETRY only (not the root — a visual_usd_ref child must stay shown).
        from pxr import UsdGeom

        UsdGeom.Imageable(stage.GetPrimAtPath(geom_prim_path)).MakeInvisible()
    if getattr(cfg, "visual_usd_ref", None):
        # Visual-only referenced asset riding this body's pose. REMOVE its physics API schemas
        # outright (a local delete-op over the reference): merely disabling them leaves the
        # applied schemas visible to prim-resolution queries, which then see two rigid bodies
        # under this prim and refuse to bind the RigidObject.
        from pxr import Usd, UsdPhysics

        asset_prim = stage.DefinePrim(f"{prim_path}/visual")
        asset_prim.GetReferences().AddReference(cfg.visual_usd_ref)
        for prim in Usd.PrimRange(asset_prim):
            for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.RigidBodyAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)

    proxy_paths: list[str] = []
    proxy_friction_paths: list[str] = []  # ring + slab: tabletop-friction material (NOT the handle)
    if getattr(cfg, "rigidproxy", None):
        # Invisible CONCAVE rigid proxies under <root>/rigidproxy/: extra collision shapes on
        # the SAME rigid body. The coupled manager routes flags by the "/rigidproxy/" label:
        # rigid-only, except the handle bars which keep particle collision too.
        pr = cfg.rigidproxy
        base_path = f"{prim_path}/rigidproxy"
        create_prim(base_path, prim_type="Xform", stage=stage)
        n, t = int(pr["segments"]), float(pr["thickness"])
        r_out, (z_lo, z_hi) = float(pr["ring_r_out"]), pr["ring_z"]
        r_mid = r_out - t / 2.0
        chord = 2.0 * r_mid * math.tan(math.pi / n)  # tangential width closing the polygon
        for i in range(n):
            ang = 2.0 * math.pi * i / n
            path = f"{base_path}/ring_{i:02d}"
            create_prim(
                path,
                prim_type="Cube",
                attributes={"size": 1.0},
                translation=(r_mid * math.cos(ang), r_mid * math.sin(ang), (z_lo + z_hi) / 2.0),
                orientation=(0.0, 0.0, math.sin(ang / 2.0), math.cos(ang / 2.0)),  # xyzw, Rz(ang)
                scale=(t, chord, z_hi - z_lo),
                stage=stage,
            )
            proxy_paths.append(path)
            proxy_friction_paths.append(path)
        # Floor slab: a BOX (inscribed square), NOT a cylinder — cylinder-box CCD contacts
        # ratchet resting bodies across the table.
        slab_path = f"{base_path}/slab"
        slab_side = float(pr["slab_r"]) * math.sqrt(2.0)  # inscribed in the base circle
        create_prim(
            slab_path,
            prim_type="Cube",
            attributes={"size": 1.0},
            translation=(0.0, 0.0, float(pr["slab_h"]) / 2.0),
            scale=(slab_side, slab_side, float(pr["slab_h"])),
            stage=stage,
        )
        proxy_paths.append(slab_path)
        proxy_friction_paths.append(slab_path)
        # Handle bar: a SEGMENTED STACK of boxes — CCD emits one contact point per geom PAIR,
        # so N segments give an N-point planar pinch manifold along the bar.
        # Square cross-section grip_w x grip_w.
        h = pr["handle"]
        bar_w = float(h.get("grip_w", 2.0 * h["r"]))
        bar_len = 2.0 * (float(h["half_height"]) + float(h["r"]))
        n_seg = int(h.get("segments", 4))
        seg_len = bar_len / n_seg
        z_lo_bar = float(h["z"]) - bar_len / 2.0
        from pxr import Gf as _Gf
        from pxr import Sdf as _Sdf

        for k in range(n_seg):
            seg_path = f"{base_path}/handle_{k:02d}"
            create_prim(
                seg_path,
                prim_type="Cube",
                attributes={"size": 1.0},
                translation=(float(h["x"]), 0.0, z_lo_bar + (k + 0.5) * seg_len),
                scale=(bar_w, bar_w, seg_len),
                stage=stage,
            )
            # Stiff contact on the grip bar (mjc:solref, read by the newton importer).
            stage.GetPrimAtPath(seg_path).CreateAttribute(
                "mjc:solref", _Sdf.ValueTypeNames.Float2, custom=True
            ).Set(_Gf.Vec2f(0.004, 1.0))
            proxy_paths.append(seg_path)
        from pxr import UsdGeom

        UsdGeom.Imageable(stage.GetPrimAtPath(base_path)).MakeInvisible()

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if getattr(cfg, "mass", None):
        # authored total mass on the body root (importer computes CoM/inertia scaled to match)
        from pxr import UsdPhysics

        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(prim_path)).GetMassAttr().Set(float(cfg.mass))
    if cfg.collision_props is not None:
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)
        for path in proxy_paths:
            schemas.define_collision_properties(path, cfg.collision_props, stage=stage)
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
        # liquid-facing material also covers the handle bars (MPM colliders); ring + slab get
        # the tabletop material below
        for path in proxy_paths:
            if path not in proxy_friction_paths:
                bind_physics_material(path, material_path, stage=stage)
    if getattr(cfg, "proxy_physics_material", None) is not None and proxy_friction_paths:
        proxy_mat_path = f"{geom_prim_path}/proxyMaterial"
        cfg.proxy_physics_material.func(proxy_mat_path, cfg.proxy_physics_material)
        for path in proxy_friction_paths:
            bind_physics_material(path, proxy_mat_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)
