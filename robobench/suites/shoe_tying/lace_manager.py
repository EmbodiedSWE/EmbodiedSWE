"""NewtonLaceVBDManager — standalone VBD manager specialized for rod scenes.

Adds per-substep control callbacks (`_substep_controls` — kinematic handle writes and the pin
spring forces must run every solver substep, since `clear_forces()` runs per step), invokes the
rod builder hook on the stage-fallback build path, and prepares the builder for finalize
(`rigid_gap = 0`, `builder.color()` for rod-only scenes). The cfg subclasses below carry the
rod contact recipe into `SolverVBD` / `CollisionPipeline`.

Runs ONLY under the Newton venv with the app up (imports isaaclab_newton at module level);
import it lazily from `RodSimCfg.to_isaaclab` / the smoke.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from isaaclab.physics import PhysicsManager
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.physics.newton_manager import NewtonManager


class NewtonLaceVBDManager(NewtonVBDManager):
    """VBD manager with per-substep control interleaving for kinematic rod handles."""

    # Callables (state_0, state_1, solver_dt) invoked before EVERY solver substep. Write handle
    # body_q into BOTH states (the loop double-buffers); add forces onto state_0.body_f.
    _substep_controls: list[Callable[[Any, Any, float], None]] = []

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Stage-fallback build + lace injection: this path (taken when no asset queued cloner
        replication) never invokes the per-world builder hooks, so run the lace hook here
        (single unpartitioned world, identity origin)."""
        super().instantiate_builder_from_stage()
        from robobench.suites.shoe_tying.scenes import shoe_knot

        for scene in shoe_knot._ACTIVE:
            if not scene.lace_bodies_w:  # cloner didn't run the hook — flat single-world build
                shoe_knot._add_knot_world(cls._builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    @classmethod
    def _prepare_builder_for_finalize(cls, builder) -> None:
        super()._prepare_builder_for_finalize(builder)
        if not builder.body_count:
            return  # nothing hook-injected — leave the builder alone
        builder.rigid_gap = 0.0
        # VBD requires ModelBuilder.color() before finalize(); the parent only colors when its
        # deformable registry is non-empty, so rod-only scenes color here.
        builder.color()

    @classmethod
    def _simulate_physics_only(cls) -> None:
        # skip the VBD parent's cloth-only rebuild_bvh (breaks on particle-less models)
        super(NewtonVBDManager, cls)._simulate_physics_only()

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        controls = cls._substep_controls
        if not controls:
            super()._run_solver_substeps(contacts)
            return

        # Mirror of NewtonManager._run_solver_substeps (double-buffered branch) with the
        # control callbacks interleaved, per-substep order: control -> collide -> solve.
        collide = cls._needs_collision_pipeline and contacts is not None
        cfg = PhysicsManager._cfg
        need_copy_on_last = (cfg is not None and cfg.use_cuda_graph) and cls._num_substeps % 2 == 1

        for i in range(cls._num_substeps):
            for ctl in controls:
                ctl(cls._state_0, cls._state_1, cls._solver_dt)
            if collide:
                cls._collision_pipeline.collide(cls._state_0, contacts)
            cls._step_solver(cls._state_0, cls._state_1, cls._control, contacts, cls._solver_dt)
            if need_copy_on_last and i == cls._num_substeps - 1:
                cls._state_0.assign(cls._state_1)
            else:
                NewtonManager._state_0, NewtonManager._state_1 = cls._state_1, cls._state_0
            cls._state_0.clear_forces()

    @classmethod
    def _solver_specific_clear(cls) -> None:
        super()._solver_specific_clear()
        NewtonLaceVBDManager._substep_controls = []


@configclass
class LaceVBDSolverCfg(VBDSolverCfg):
    """`VBDSolverCfg` + the AVBD rigid-contact knobs the rod knots were tuned on; `class_type`
    routes model build + stepping to the lace manager."""

    class_type: type[NewtonManager] | str = NewtonLaceVBDManager

    rigid_body_contact_buffer_size: int = 512
    """Per-body body-body contact list capacity (rod self-contact in a cinched knot is dense)."""

    rigid_contact_history: bool = True
    """Warm-start AVBD contact duals across steps (pairs with sticky contact matching)."""

    rigid_avbd_contact_alpha: float = 0.0
    """Body-body contact alpha override."""


@configclass
class LaceCollisionCfg(NewtonCollisionPipelineCfg):
    """Collision pipeline cfg + frame-to-frame contact matching."""

    contact_matching: str = "sticky"
    """"sticky" replays previous-frame contact geometry on matched contacts."""
