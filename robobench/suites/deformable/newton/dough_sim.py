"""DoughSimCfg — the Newton sim substrate for elastoplastic dough (coupled MJWarp + implicit MPM).

The pouring suite's `MpmSimCfg` recipe with `coupled=True` as the default: dough is a task about
a robot WORKING the material, so the coupled substrate is SolverMuJoCo advancing the arm (real
gravity, actuator PD, MuJoCo rigid contacts) at `dt/num_substeps`, then the implicit MPM step
advancing the dough once per tick reading the post-rigid body poses (one-way rigid -> dough).
`coupled=False` is the registered robot-less physics-tuning binding (`deformable.dumpling`) — the
pure MPM manager, where rigid geometry is colliders only and the pin is a scripted kinematic
ghost. Robot bindings live outside the benchmark tree (experiments build their own EnvCfg).

The coupled manager itself is suite-local: `robobench.suites.deformable.newton.coupled_manager`, a verbatim
twin of the pouring suite's (suites must not import each other — eval extraction ships one suite
at a time — so each carries its own copy; keep the twins in sync).
`liquid_feedback` stays available but defaults OFF: a rolling pin does not need to feel the dough
at v1, and one-way keeps the commanded pass heights ground truth.

Requires the Newton venv (`env_newton`, see the README). Heavy imports are deferred to
`to_isaaclab()` so importing this module stays app-free (and survives the assembly suite's
2.3.2 venv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robobench.core import SimCfg

# MJWarp defaults for the coupled substrate. An undersized contact buffer drops overflowing
# pairs SILENTLY (ghost contacts) while an undersized njmax fails loudly with "nefc overflow";
# sized with heavy headroom for an arm + tools + cradles.
_MJWARP_DEFAULTS: dict[str, Any] = {
    "njmax": 1600,
    "nconmax": 800,
    "ls_iterations": 20,
    "cone": "pyramidal",
    "impratio": 1,
    "integrator": "implicitfast",
    "ccd_iterations": 100,
}


@dataclass
class DoughSimCfg(SimCfg):
    """Newton implicit-MPM substrate for dough. `dt`/`gravity`/`render` are inherited from
    `SimCfg`; the MPM defaults are the pouring suite's proven values (the dough scene overrides
    `voxel_size` from its cfg). `mpm` is splatted into `MPMSolverCfg` last, so any solver knob
    works without growing this class."""

    dt: float = 1.0 / 200.0  # MPM stability wants small steps; pouring's proven 200 Hz
    voxel_size: float = 0.0025  # MPM grid voxel [m]; 2 voxels through the 5 mm target sheet
    grid_type: str = "fixed"  # "fixed" grid -> the solver loop is CUDA-graph captured
    grid_padding: int = 64  # fixed-grid padding [cells] around the initial particle bounds
    max_active_cell_count: int = 1 << 17
    max_iterations: int = 100  # rheology iterations; inside a CUDA graph it always runs all of them
    air_drag: float = 0.2
    use_cuda_graph: bool = True  # False -> slow but debuggable stepping
    mpm: dict[str, Any] = field(default_factory=dict)  # extra MPMSolverCfg overrides
    # --- coupled MJWarp+MPM substrate ---
    coupled: bool = True  # the benchmark runs a dynamic arm; False -> pure-MPM tuning substrate
    num_substeps: int = 3  # MuJoCo substeps per MPM tick (rigid dt = dt/num_substeps = 1/600)
    mjwarp: dict[str, Any] = field(default_factory=dict)  # MJWarpSolverCfg overrides (merged over
    # _MJWARP_DEFAULTS inside to_isaaclab — sim_overrides replaces this dict wholesale)
    liquid_feedback: bool = False  # 1.5-way coupling: MPM collider impulses -> rigid body_f
    liquid_force_clamp: float = 10.0  # per-grid-node fluid force clamp [N]
    welds: list = field(default_factory=list)  # builder-time MuJoCo equality welds
    # [(label, body1 suffix, body2 suffix)], created DISABLED; toggled via
    # NewtonCoupledMJWarpMPMManager.set_weld. Coupled substrate only.

    def to_isaaclab(self, device: str) -> Any:
        """Build the isaaclab `SimulationCfg` with the Newton implicit-MPM backend."""
        from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

        # Same trick as the folding/pouring suites: the kitless check matches physics-cfg class
        # NAMES ("NewtonCfg"/"OvPhysxCfg"), and a subclass name forces Kit to launch — which the
        # USD asset spawn path and the Kit particle visualization require. Defined here (not
        # module level) so this module imports without the Newton stack installed.
        @configclass
        class DoughNewtonCfg(NewtonCfg):
            model_cfg: Any = None

        mpm_solver_cfg = MPMSolverCfg(
            **{
                "voxel_size": self.voxel_size,
                "grid_type": self.grid_type,
                "grid_padding": self.grid_padding,
                "max_active_cell_count": self.max_active_cell_count,
                "max_iterations": self.max_iterations,
                "air_drag": self.air_drag,
                "collider_velocity_mode": "backward",
                "project_outside_colliders": True,
                **self.mpm,
            }
        )
        if self.coupled:
            # the suite-local coupled manager — MJWarp rigids + the SAME MPM recipe
            from isaaclab_newton.physics import MJWarpSolverCfg

            from robobench.suites.deformable.newton.coupled_manager import MJWarpMPMSolverCfg

            solver_cfg: Any = MJWarpMPMSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(**{**_MJWARP_DEFAULTS, **self.mjwarp}),
                mpm_solver_cfg=mpm_solver_cfg,
                weld_specs=[tuple(w) for w in self.welds],
                liquid_feedback=self.liquid_feedback,
                liquid_force_clamp=self.liquid_force_clamp,
            )
            num_substeps = self.num_substeps
        else:
            solver_cfg = mpm_solver_cfg
            num_substeps = 1  # the MPM-only manager steps once per tick at dt
        physics = DoughNewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=num_substeps,
            use_cuda_graph=self.use_cuda_graph,
            simplify_meshes=False,  # keep the pin barrel's exact round collider
        )
        return SimulationCfg(
            device=device, dt=self.dt, gravity=self.gravity, physics=physics, render=RenderCfg(**self.render)
        )
