"""TshirtFoldingScene — a T-shirt lying on a box table, to be folded flat (Newton VBD cloth).

Runs on IsaacLab develop's **Newton backend**: the shirt is a *surface deformable*
(`DeformableObjectCfg` + Newton surface material -> VBD cloth particles), the table a static
collidable box (no rigid body, no DOFs — stays out of the Newton joint state), and the substrate
a coupled MJWarp(rigid) + VBD(cloth) solver declared by `sim_cfg()` (`NewtonSimCfg`).

The shirt USD (`assets/tshirt/tshirt.usd`, meters) has its spawn pose baked into the vertices, so
`init_state` stays at the origin. Layout (env-local meters): table top at z=0.20, sleeve tips
settle near (±0.34, -0.58), bottom hem near y=-0.18, collar near y=-0.83.

Success proxy: the cloth footprint = x-extent · y-extent. Settled unfolded ≈ 0.50 m²; a completed
3-fold run lands ≈ 0.17-0.25 m²; below 0.30 m² counts as folded (all observed failure modes stay
≥ 0.41).

Requires the Newton venv (`env_newton`, see the README) to build; the assembly
suite's isaaclab 2.3.2 venv has no Newton backend. Heavy imports are deferred so importing this
module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robobench.core import SCENES, BaseCfg, BaseScene, info, tunable
from robobench.suites.folding.newton_sim import NewtonSimCfg

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv


@dataclass
class TshirtFoldingSceneCfg(BaseCfg):
    """Cloth/contact dials start from IsaacLab's proven meter-scale cloth task
    (isaaclab_tasks core/lift/franka_soft/franka_cloth_env_cfg.py), then hand-tuned for a
    soft-fabric look that drapes flat and holds folds (see the [TUNE] notes)."""

    # --- cloth material (meter scale) ---
    cloth_density: float = tunable(15.0)  # [TUNE] surface density; 5 springs back after folds, 50 too heavy to lift
    particle_radius: float = tunable(0.008)  # [TUNE] cloth-body contact radius; at 0.005 the
    # resting layer is too thin for a fingertip pinch to gather flat fabric (hem grasp fails)
    tri_ke: float = tunable(5e2)  # [TUNE] triangle stretch stiffness
    tri_ka: float = tunable(5e2)  # [TUNE] triangle area stiffness
    tri_kd: float = tunable(1e-1)  # [TUNE] triangle stretch damping; high value damps out cloth wobble
    edge_ke: float = tunable(0.3)  # [TUNE] bending stiffness; low = soft drape that lies flat and holds folds
    edge_kd: float = tunable(1e-1)  # [TUNE] bending damping
    # --- contacts (Newton model-level) ---
    soft_contact_ke: float = tunable(1e3)  # [TUNE] particle-body contact stiffness
    soft_contact_kd: float = tunable(1e-5)  # [TUNE] particle-body contact damping
    soft_contact_mu: float = tunable(0.5)  # [TUNE] particle-side friction
    shape_ke: float = tunable(1e3)  # [TUNE] per-shape contact stiffness override
    shape_kd: float = tunable(1e-5)  # [TUNE] per-shape contact damping override
    shape_mu: float = tunable(1.5)  # [TUNE] per-shape friction (table + robot)
    robot_friction_boost: float | None = tunable(None)  # [TUNE] extra mu on ROBOT shapes only
    # (tried for grasp slip — net regression: the whole arm becomes sticky and drags the cloth)
    cloth_contact_margin: float = tunable(0.012)  # [TUNE] cloth-body collision margin (>= particle_radius)
    # --- VBD solver ---
    vbd_iterations: int = tunable(20)  # [TUNE] VBD iterations/substep; more = crisper (less rubbery) cloth
    self_contact: bool = tunable(True)  # folding lays cloth on cloth — keep self-contact ON
    self_contact_radius: float = tunable(0.002)
    self_contact_margin: float = tunable(0.002)
    num_substeps: int = tunable(10)  # solver substeps per 1/60 s physics tick
    use_cuda_graph: bool = tunable(True)  # False -> slow but debuggable stepping
    # --- layout (meters; the shirt pose is baked into the USD) ---
    light_intensity: float = tunable(3000.0)
    table_size: tuple[float, float, float] = info((0.8, 0.8, 0.2), doc="box table full extents [m]; top at z=0.2")
    table_pos: tuple[float, float, float] = info((0.0, -0.5, 0.1), doc="box table center [m]")
    shirt_z_offset: float = tunable(0.0)  # [TUNE] extra spawn height over the baked pose (avoid table overlap)
    shirt_usd: str = info("", doc="'' -> the vendored assets/tshirt/tshirt.usd (baked pose, meters)")

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets" / "tshirt"
        self.shirt_usd = self.shirt_usd or str(assets / "tshirt.usd")


@SCENES.register("tshirt")
class TshirtFoldingScene(BaseScene):
    """T-shirt (VBD cloth) + static box table + ground. Goal: fold the shirt flat (sleeves to the
    center line, hem up to the collar) — footprint below ~0.20 m². The cloth handle is
    `self.cloth` after bind; `footprint()` is the success proxy."""

    cfg: TshirtFoldingSceneCfg

    def __init__(self, cfg: TshirtFoldingSceneCfg | None = None) -> None:
        super().__init__(cfg or TshirtFoldingSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.assets.deformable_object import DeformableObjectCfg
        from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
        from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

        c = self.cfg
        return {
            # Ground sunk to z=-1.05 like the in-tree Newton cloth task (franka_soft_env_cfg.py):
            # at z=0 the ground-plane collider goes haywire in the Newton->MuJoCo conversion and
            # grinds huge phantom contact forces into the arm (thousands of N*m on joints 2/4,
            # measured via qfrc_constraint). The box table is the actual work surface.
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
                spawn=sim_utils.GroundPlaneCfg(),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9)),
            ),
            # Static collidable box (pattern from the in-tree cloth task): an AssetBaseCfg with
            # collision but no rigid body, so it does not extend the Newton model's joint state.
            "table": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=c.table_pos),
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
                ),
            ),
            # The shirt: a surface deformable — the Newton backend registers the spawned USD mesh
            # into the ModelBuilder as VBD cloth (builder.add_cloth_mesh + graph coloring).
            "cloth": DeformableObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shirt",
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.shirt_z_offset)),  # pose baked into the USD
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.shirt_usd,
                    deformable_props=NewtonDeformableBodyPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
                    physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                        density=c.cloth_density,
                        particle_radius=c.particle_radius,
                        tri_ke=c.tri_ke,
                        tri_ka=c.tri_ka,
                        tri_kd=c.tri_kd,
                        edge_ke=c.edge_ke,
                        edge_kd=c.edge_kd,
                    ),
                ),
            ),
        }

    def sim_cfg(self) -> NewtonSimCfg:
        c = self.cfg
        return NewtonSimCfg(
            dt=1.0 / 60.0,
            num_substeps=c.num_substeps,
            use_cuda_graph=c.use_cuda_graph,
            vbd={
                "iterations": c.vbd_iterations,
                "particle_enable_self_contact": c.self_contact,
                "particle_self_contact_radius": c.self_contact_radius,
                "particle_self_contact_margin": c.self_contact_margin,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.005,
                "particle_collision_detection_interval": -1,
            },
            model={
                "soft_contact_ke": c.soft_contact_ke,
                "soft_contact_kd": c.soft_contact_kd,
                "soft_contact_mu": c.soft_contact_mu,
                "shape_material_ke": c.shape_ke,
                "shape_material_kd": c.shape_kd,
                "shape_material_mu": c.shape_mu,
            },
            collision={"soft_contact_margin": c.cloth_contact_margin},
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cloth = env.iscene["cloth"]
        # Snapshot the spawn state as the reset target (shirt hovering at its baked pose, at rest).
        self._default_nodal_pos = self.cloth.data.nodal_pos_w.torch.clone()
        self._default_nodal_vel = self.cloth.data.nodal_vel_w.torch.clone().zero_()
        if self.cfg.robot_friction_boost is not None:
            self._boost_robot_friction(env, float(self.cfg.robot_friction_boost))

    def _boost_robot_friction(self, env: BaseEnv, mu: float) -> None:
        """Overwrite friction on the ROBOT's collision shapes only (the model-level `shape_mu`
        already covers every shape, robot included — this dial pushes the robot higher, the way the
        in-tree cloth task boosts the Franka to mu=100). Mechanism copied from isaaclab's Newton
        material-randomization event (envs/mdp/events.py, _RandomizeRigidBodyMaterialNewton)."""
        try:
            robot = env.iscene["robot"]
        except KeyError:
            return  # scene-only binding (NullRobot) — nothing to boost
        import warp as wp
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton.solvers import SolverNotifyFlags

        binding = robot._root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0]
        wp.to_torch(binding)[:] = mu
        NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.cloth.write_nodal_pos_to_sim_index(self._default_nodal_pos[env_ids].contiguous(), env_ids=env_ids)
        self.cloth.write_nodal_velocity_to_sim_index(self._default_nodal_vel[env_ids].contiguous(), env_ids=env_ids)

    # ----- state ----------------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        d = self.cloth.data
        return {
            "nodal_pos": d.nodal_pos_w.torch[env_ids].clone(),
            "nodal_vel": d.nodal_vel_w.torch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cloth.write_nodal_pos_to_sim_index(state["nodal_pos"].contiguous(), env_ids=env_ids)
        self.cloth.write_nodal_velocity_to_sim_index(state["nodal_vel"].contiguous(), env_ids=env_ids)

    # ----- metrics --------------------------------------------------------------------------------
    def footprint(self) -> torch.Tensor:
        """Per-env cloth footprint proxy [m²] = x-extent · y-extent of the particle cloud
        (settled unfolded ≈ 0.50, a completed fold ≈ 0.17-0.25)."""
        p = self.cloth.data.nodal_pos_w.torch  # (num_envs, P, 3); extents are env-origin invariant
        return (p[..., 0].amax(dim=1) - p[..., 0].amin(dim=1)) * (p[..., 1].amax(dim=1) - p[..., 1].amin(dim=1))

    def nodal_pos_local(self) -> torch.Tensor:
        """Nodal positions with the env origin removed (env-local frame, for bounds checks)."""
        p = self.cloth.data.nodal_pos_w.torch
        return p - self.env.iscene.env_origins[:, None, :]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        return (
            "A T-shirt lies flat on a gray box table (top at z=0.20 m): sleeve tips near (±0.34, -0.58),"
            " bottom hem near y=-0.18, collar near y=-0.83 (env-local meters). Goal: fold the shirt into a"
            " compact flat bundle — left sleeve to the center line, right sleeve to the center line, then"
            " the bottom hem up over the collar — so its footprint (x-extent · y-extent of the cloth)"
            " drops below 0.30 m² (settled unfolded is ~0.50 m²; a clean fold reaches ~0.25)."
        )
