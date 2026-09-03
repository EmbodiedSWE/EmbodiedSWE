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

Robot-less scenes use the standalone VBD manager (cable joints are not MuJoCo-convertible); a
future Franka binding needs a coupled MJWarp+VBD-rod manager (pouring's `coupled_manager`
precedent; upstream reference: newton's `example_mujoco_franka_vbd_cable_admm_solver`).

Requires the Newton venv (`env_newton`, newton >= 1.6 — see the README). Heavy imports are
deferred to `to_isaaclab()` so importing this module stays app-free (and survives the assembly
suite's 2.3.2 venv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robobench.core import SimCfg


@dataclass
class RodSimCfg(SimCfg):
    """Newton standalone-VBD substrate for rod scenes. `dt`/`gravity`/`render` are inherited from
    `SimCfg`; the dict fields below are splatted into the suite's cfg subclasses (so any solver /
    collision knob works without growing this class), keeping this module import-light."""

    dt: float = 1.0 / 60.0
    num_substeps: int = 12  # solver substeps per physics tick (solver dt = dt / num_substeps)
    use_cuda_graph: bool = False  # per-substep python control cannot run inside a captured graph
    collision_decimation: int = 1  # collide every substep (the standalone scripts' cadence)
    vbd: dict[str, Any] = field(default_factory=dict)  # LaceVBDSolverCfg kwargs (iterations, rigid contact knobs, ...)
    collision: dict[str, Any] = field(default_factory=dict)  # LaceCollisionCfg kwargs (contact_matching, margins, ...)

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

        physics = RodNewtonCfg(
            solver_cfg=LaceVBDSolverCfg(**self.vbd),
            collision_cfg=LaceCollisionCfg(**self.collision),
            num_substeps=self.num_substeps,
            collision_decimation=self.collision_decimation,
            use_cuda_graph=self.use_cuda_graph,
        )
        return SimulationCfg(
            device=device, dt=self.dt, gravity=self.gravity, physics=physics, render=RenderCfg(**self.render)
        )
