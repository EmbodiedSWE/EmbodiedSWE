"""NewtonSimCfg — the Newton sim substrate (MJWarp rigid + VBD cloth, two-way coupled).

IsaacLab develop selects its physics backend through `SimulationCfg.physics`; this `SimCfg`
subclass carries the Newton knobs app-free (plain dicts) and builds the real config in
`to_isaaclab()`. It drops into the unchanged robobench core: `EnvCfg.build()` only
`dataclasses.replace`s the scene's SimCfg (type-preserving) and calls `to_isaaclab()`
(polymorphic). The inherited `physx` dict is ignored — PhysX knobs do not apply here.

Solver shape mirrors the proven in-tree cloth task (isaaclab_tasks `core/lift/franka_soft`):
MJWarp rigid solver + VBD cloth solver, `coupling_mode="two_way"` so the arm feels the cloth
through contact friction (what lets a pinch grip carry cloth).

Requires the Newton venv (`env_newton`, see the README): isaaclab develop +
isaaclab_newton + isaaclab_contrib. Heavy imports are deferred to `to_isaaclab()` so importing
this module stays app-free (and survives the assembly suite's 2.3.2 venv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robobench.core import SimCfg

# Proven MJWarp rigid-solver settings for coupled cloth manipulation (franka_soft_env_cfg.py),
# with contact/constraint buffers sized up for this scene (arm + table + ground: the task's
# 40/20 overflowed — "nefc overflow, increase njmax to 140" — once the gripper worked the cloth).
_MJWARP_DEFAULTS: dict[str, Any] = {
    "njmax": 300,
    "nconmax": 150,
    "ls_iterations": 20,
    "cone": "pyramidal",
    "impratio": 1,
    "integrator": "implicitfast",
    "ccd_iterations": 100,
}


@dataclass
class NewtonSimCfg(SimCfg):
    """Newton-backend substrate. `dt`/`gravity`/`render` are inherited from `SimCfg`; the dict
    fields below are splatted into the matching isaaclab_newton / isaaclab_contrib cfg classes
    (so any knob works without growing this class), keeping this module import-light."""

    num_substeps: int = 10  # solver substeps per physics tick (solver dt = dt / num_substeps)
    use_cuda_graph: bool = True  # capture the substep loop in a CUDA graph (False = slow, easier to debug)
    coupled: bool = True  # False -> pure VBD manager; use for robot-less bindings (MJWarp needs >= 1 joint)
    rigid_solver: str = "mjwarp"  # "mjwarp" | "featherstone"
    coupling_mode: str = "one_way"  # rigid->cloth only; "two_way" adds cloth->rigid reactions
    mjwarp: dict[str, Any] = field(default_factory=dict)  # MJWarpSolverCfg overrides (merged over defaults)
    vbd: dict[str, Any] = field(default_factory=dict)  # VBDSolverCfg kwargs (iterations, self-contact, ...)
    model: dict[str, Any] = field(default_factory=dict)  # NewtonModelCfg kwargs (soft/shape contact params)
    collision: dict[str, Any] = field(default_factory=dict)  # NewtonCollisionPipelineCfg kwargs (margins, ...)

    def to_isaaclab(self, device: str) -> Any:
        """Build the isaaclab `SimulationCfg` with a coupled MJWarp+VBD Newton backend."""
        from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_contrib.deformable.newton_manager_cfg import (
            CoupledFeatherstoneVBDSolverCfg,
            CoupledMJWarpVBDSolverCfg,
            NewtonModelCfg,
            VBDSolverCfg,
        )
        from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

        # Same trick as the in-tree cloth task's `DeformableNewtonCfg` (franka_soft_env_cfg.py):
        # the coupled deformable managers consume an optional `model_cfg` attribute (hasattr), and
        # the kitless check matches class NAMES ("NewtonCfg"/"OvPhysxCfg") — a subclass name forces
        # Kit to launch, which the USD deformable spawn path requires. Defined here (not module
        # level) so this module imports without the Newton stack installed.
        @configclass
        class FoldingNewtonCfg(NewtonCfg):
            model_cfg: Any = None

        if self.coupled and self.rigid_solver == "featherstone":
            solver_cfg: Any = CoupledFeatherstoneVBDSolverCfg(
                rigid_solver_cfg=FeatherstoneSolverCfg(),
                soft_solver_cfg=VBDSolverCfg(integrate_with_external_rigid_solver=True, **self.vbd),
                coupling_mode=self.coupling_mode,
            )
        elif self.coupled:
            solver_cfg = CoupledMJWarpVBDSolverCfg(
                rigid_solver_cfg=MJWarpSolverCfg(**{**_MJWARP_DEFAULTS, **self.mjwarp}),
                soft_solver_cfg=VBDSolverCfg(integrate_with_external_rigid_solver=True, **self.vbd),
                coupling_mode=self.coupling_mode,
            )
        else:
            # Robot-less scene: standalone VBD integrates everything itself (MuJoCo conversion
            # rejects a model with zero joints, so the coupled manager cannot be used here).
            solver_cfg = VBDSolverCfg(**self.vbd)
        physics = FoldingNewtonCfg(
            solver_cfg=solver_cfg,
            model_cfg=NewtonModelCfg(**self.model),
            collision_cfg=NewtonCollisionPipelineCfg(**self.collision) if self.collision else None,
            num_substeps=self.num_substeps,
            use_cuda_graph=self.use_cuda_graph,
        )
        return SimulationCfg(
            device=device, dt=self.dt, gravity=self.gravity, physics=physics, render=RenderCfg(**self.render)
        )
