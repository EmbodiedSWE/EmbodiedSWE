"""MpmSimCfg — the Newton sim substrate for particle liquids (implicit MPM).

IsaacLab develop selects its physics backend through `SimulationCfg.physics`; this `SimCfg`
subclass carries the MPM knobs app-free and builds the real config in `to_isaaclab()`. It drops
into the unchanged robobench core the same way the folding suite's `NewtonSimCfg` does:
`EnvCfg.build()` only `dataclasses.replace`s the scene's SimCfg (type-preserving) and calls
`to_isaaclab()` (polymorphic).

Solver shape mirrors the in-tree MPM pour demo (IsaacLab `scripts/demos/mpm/particle_pour.py`):
implicit MPM with a fixed grid so the whole solve is captured in one CUDA graph. Two substrates
share this cfg, selected by `coupled`:

- `coupled=False` (default): the MPM-only manager — rigid geometry is *colliders only*, robots
  are kinematic ghosts (Phase 1 / 2a).
- `coupled=True` (Phase 2b): the suite-local coupled MJWarp+MPM manager
  (`robobench.suites.pouring.coupled_manager`) — SolverMuJoCo advances articulations with real
  gravity/actuators/contacts at `dt/num_substeps`, then the implicit MPM step advances the
  liquids once per tick reading the post-rigid body poses (one-way rigid -> fluid).

Requires the Newton venv (`env_newton`, see the README). Heavy imports are deferred to
`to_isaaclab()` so importing this module stays app-free (and survives the assembly suite's
2.3.2 venv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robobench.core import SimCfg

# MJWarp defaults for the coupled substrate — folding's proven values, contact buffers doubled:
# that suite sized njmax=300/nconmax=150 for ONE arm + table; the latte scene runs TWO Frankas.
# Undersizing fails at runtime with "nefc overflow, increase njmax to N".
_MJWARP_DEFAULTS: dict[str, Any] = {
    "njmax": 600,
    "nconmax": 300,
    "ls_iterations": 20,
    "cone": "pyramidal",
    "impratio": 1,
    "integrator": "implicitfast",
    "ccd_iterations": 100,
}


@dataclass
class MpmSimCfg(SimCfg):
    """Newton implicit-MPM substrate. `dt`/`gravity`/`render` are inherited from `SimCfg`; the
    defaults below are the pour demo's proven values. `mpm` is splatted into `MPMSolverCfg` last,
    so any solver knob works without growing this class."""

    dt: float = 1.0 / 200.0  # MPM stability wants small steps; the pour demo runs 200 Hz
    voxel_size: float = 0.003  # MPM grid voxel [m]; particle spacing follows via particles_per_cell
    grid_type: str = "fixed"  # "fixed" grid -> the solver loop is CUDA-graph captured
    grid_padding: int = 64  # fixed-grid padding [cells] around the initial particle bounds
    max_active_cell_count: int = 1 << 17
    max_iterations: int = 100  # rheology iterations; inside a CUDA graph it always runs all of them
    air_drag: float = 0.2
    use_cuda_graph: bool = True  # False -> slow but debuggable stepping
    mpm: dict[str, Any] = field(default_factory=dict)  # extra MPMSolverCfg overrides
    # --- coupled MJWarp+MPM substrate (Phase 2b) ---
    coupled: bool = False  # True -> MJWarp rigid dynamics + MPM liquids (dynamic robots)
    num_substeps: int = 3  # MuJoCo substeps per MPM tick (rigid dt = dt/num_substeps = 1/600)
    mjwarp: dict[str, Any] = field(default_factory=dict)  # MJWarpSolverCfg overrides (merged over
    # _MJWARP_DEFAULTS inside to_isaaclab — sim_overrides replaces this dict wholesale)
    finger_pads: bool = False  # analytic box pads on the Franka fingertips (force closure)
    welds: list = field(default_factory=list)  # builder-time MuJoCo equality welds
    # [(label, body1 suffix, body2 suffix)], created DISABLED; toggled via
    # NewtonCoupledMJWarpMPMManager.set_weld. Coupled substrate only.

    def to_isaaclab(self, device: str) -> Any:
        """Build the isaaclab `SimulationCfg` with the Newton implicit-MPM backend."""
        from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg

        # Same trick as the folding suite: the kitless check matches physics-cfg class NAMES
        # ("NewtonCfg"/"OvPhysxCfg"), and a subclass name forces Kit to launch — which the USD
        # asset spawn path and the Kit particle visualization require. Defined here (not module
        # level) so this module imports without the Newton stack installed.
        @configclass
        class PouringNewtonCfg(NewtonCfg):
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
            # Phase 2b: the suite-local coupled manager — MJWarp rigids + the SAME MPM recipe.
            from isaaclab_newton.physics import MJWarpSolverCfg

            from robobench.suites.pouring.coupled_manager import MJWarpMPMSolverCfg

            solver_cfg: Any = MJWarpMPMSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(**{**_MJWARP_DEFAULTS, **self.mjwarp}),
                mpm_solver_cfg=mpm_solver_cfg,
                weld_specs=[tuple(w) for w in self.welds],
                finger_pad_boxes=self.finger_pads,
            )
            num_substeps = self.num_substeps
        else:
            solver_cfg = mpm_solver_cfg
            num_substeps = 1  # the MPM-only manager steps once per tick at dt
        physics = PouringNewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=num_substeps,
            use_cuda_graph=self.use_cuda_graph,
            simplify_meshes=False,  # keep the exact cup geometry (thin walls) as colliders
        )
        return SimulationCfg(
            device=device, dt=self.dt, gravity=self.gravity, physics=physics, render=RenderCfg(**self.render)
        )
