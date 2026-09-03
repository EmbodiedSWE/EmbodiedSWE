"""RodSimCfg — the Newton sim substrate for rod (cable-joint) scenes, standalone VBD.

IsaacLab develop selects its physics backend through `SimulationCfg.physics`; this `SimCfg`
subclass carries the rod-solver knobs app-free and builds the real config in `to_isaaclab()`.
It drops into the unchanged robobench core the same way the folding suite's `NewtonSimCfg` does:
`EnvCfg.build()` only `dataclasses.replace`s the scene's SimCfg (type-preserving) and calls
`to_isaaclab()` (polymorphic).

Solver shape mirrors the standalone shoe_tying scripts this suite was ported from: SolverVBD
with sticky contact matching + contact-history warm-start (the knot-stability recipe), 12
substeps at 60 fps, collide every substep. The knobs reach the solver through the suite-local
cfg subclasses in `lace_manager.py`.

Robot-less scenes use the standalone VBD manager (cable joints are not MuJoCo-convertible).
Robot bindings set `coupled=True` -> the suite's proxy-coupled manager (`newton/lace_coupled_manager.py`):
SolverMuJoCo owns the arms, SolverVBD the rods, and the gripper bodies are proxied into the rod
solve (newton's `example_franka_cable_ik_pick_place` recipe).

Requires the Newton venv (`env_newton`, newton >= 1.6 — see the README). Heavy imports are
deferred to `to_isaaclab()` so importing this module stays app-free (and survives the assembly
suite's 2.3.2 venv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robobench.core import SimCfg

# Proven MuJoCo rigid-solver settings for the coupled robot entry — the upstream
# franka+cable pick-place example's values (use_mujoco_contacts is forced False by the
# manager: robot contacts come from the Newton collision pipeline, filtered per entry).
_MJC_DEFAULTS: dict[str, Any] = {
    "solver": "newton",
    "integrator": "implicitfast",
    "cone": "elliptic",
    "iterations": 100,
    "ls_iterations": 20,
    "njmax": 256,
    "nconmax": 64,
}


@dataclass
class RodSimCfg(SimCfg):
    """Newton standalone-VBD substrate for rod scenes. `dt`/`gravity`/`render` are inherited from
    `SimCfg`; the dict fields below are splatted into the suite's cfg subclasses (so any solver /
    collision knob works without growing this class), keeping this module import-light."""

    dt: float = 1.0 / 60.0
    num_substeps: int = 12  # solver substeps per physics tick (solver dt = dt / num_substeps)
    use_cuda_graph: bool = False  # per-substep python control cannot run inside a captured graph
    collision_decimation: int = 1  # collide every substep (the standalone scripts' cadence)
    coupled: bool = False  # True -> proxy-coupled MJWarp+VBD manager; use for robot bindings
    vbd: dict[str, Any] = field(default_factory=dict)  # LaceVBDSolverCfg kwargs (iterations, rigid contact knobs, ...)
    collision: dict[str, Any] = field(default_factory=dict)  # LaceCollisionCfg kwargs (contact_matching, margins, ...)
    mjwarp: dict[str, Any] = field(default_factory=dict)  # coupled only: SolverMuJoCo overrides (merged over defaults)
    coupling: dict[str, Any] = field(default_factory=dict)  # coupled only: LaceCoupledSolverCfg proxy_* overrides

    def to_isaaclab(self, device: str) -> Any:
        """Build the isaaclab `SimulationCfg` with a standalone-VBD Newton backend + rod hooks."""
        from isaaclab.sim.simulation_cfg import RenderCfg, SimulationCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_newton.physics import NewtonCfg

        from robobench.suites.deformable.newton.lace_manager import LaceCollisionCfg, LaceVBDSolverCfg

        # Same trick as folding's `FoldingNewtonCfg`: the kitless check matches class NAMES
        # ("NewtonCfg"/"OvPhysxCfg") — a subclass name forces Kit to launch. Defined here (not
        # module level) so this module imports without the Newton stack installed.
        @configclass
        class RodNewtonCfg(NewtonCfg):
            pass

        if self.coupled:
            # Robot binding: SolverMuJoCo (arms) + SolverVBD (rods) under SolverCoupledProxy.
            # The rod entry keeps the SAME AVBD recipe the knot was tuned on (self.vbd).
            from robobench.suites.deformable.newton.lace_coupled_manager import LaceCoupledSolverCfg

            solver_cfg: Any = LaceCoupledSolverCfg(
                mjwarp={**_MJC_DEFAULTS, **self.mjwarp}, vbd=dict(self.vbd), **self.coupling
            )
        else:
            solver_cfg = LaceVBDSolverCfg(**self.vbd)
        physics = RodNewtonCfg(
            solver_cfg=solver_cfg,
            collision_cfg=LaceCollisionCfg(**self.collision),
            num_substeps=self.num_substeps,
            collision_decimation=self.collision_decimation,
            use_cuda_graph=self.use_cuda_graph,
        )
        return SimulationCfg(
            device=device, dt=self.dt, gravity=self.gravity, physics=physics, render=RenderCfg(**self.render)
        )
