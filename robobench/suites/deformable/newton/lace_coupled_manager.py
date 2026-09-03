"""NewtonLaceCoupledManager — MJWarp arms + VBD rods on ONE Newton model, proxy-coupled.

The robot-in-the-loop substrate for the knot scene (`deformable.knot.aloha.*`): the laces are Newton
rods (cable joints — not MuJoCo-convertible), so the model is PARTITIONED between two sub-solvers
through newton's :class:`SolverCoupledProxy` (the sanctioned recipe — upstream reference at the
pinned newton: ``examples/multiphysics/example_franka_cable_ik_pick_place.py``):

  - a ``SolverMuJoCo`` entry owns the robot bodies/joints (actuator PD, MuJoCo-internal rigid
    dynamics, ``use_mujoco_contacts=False`` — robot contacts come from the manager's Newton
    :class:`CollisionPipeline`, filtered per entry);
  - a ``SolverVBD`` entry owns the hook-injected rod bodies + cable joints (the suite's proven
    AVBD contact recipe, same knobs as `lace_manager.LaceVBDSolverCfg`);
  - the gripper bodies (matched by label keyword) are exposed to the VBD entry as virtual
    PROXIES with MuJoCo-effective mass, with their own proxy collision pipeline — that is what
    lets the fingers pinch a 2.4 mm rod purely through contact friction and lets the rod's
    reaction load the wrist (lagged two-way coupling).

Cross-entry contacts from the main pipeline are dropped by the coupled solver's per-entry
filtering (finger-lace contact is exclusively the proxy pipeline's job); static shapes (table
twin, shoe trimesh, bridge capsules, ground) stay in the parent shape namespace and are seen by
both entries. Partition indices come from the active `ShoeKnotScene`'s hook records
(`lace_bodies_w` / `lace_joints_w`); everything else is the robot's.

`_step_solver` mirrors the upstream example's substep: ``solver.step`` then a full ``eval_ik``
so `state.joint_q` tracks body poses for ALL joints (articulation views read the robot's, and
the masked FK-before-step round-trips instead of stomping rod poses).

Follows the pouring/dough `newton/lace_coupled_manager.py` precedent (a suite-local manager overriding
`_build_solver`; here the coupled solver IS the canonical ``NewtonManager._solver`` since
:class:`SolverCoupledProxy` implements the standard solver interface). Runs ONLY under the
Newton venv with the app up (imports isaaclab_newton at module level); import it lazily from
`RodSimCfg.to_isaaclab`.
"""

from __future__ import annotations

import inspect

from newton import CollisionPipeline, Model, ModelBuilder, eval_ik
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg


def _filter_kwargs(cls: type, kwargs: dict) -> dict:
    """The suite/contrib pattern: keep only keys the solver's ``__init__`` accepts."""
    valid = set(inspect.signature(cls.__init__).parameters) - {"self", "model"}
    return {k: v for k, v in kwargs.items() if k in valid}


class NewtonLaceCoupledManager(NewtonManager):
    """Coupled MJWarp (robot) + VBD (rods) manager; ``_solver`` is the SolverCoupledProxy."""

    # ----- builder hooks --------------------------------------------------------------------------
    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register BOTH sub-solvers' custom attributes (idempotent; runs from create_builder,
        i.e. before the USD stage parse, so `mjc:`/`newton:vbd:` authored attrs survive)."""
        if not builder.has_custom_attribute("mujoco:gravcomp"):
            SolverMuJoCo.register_custom_attributes(builder)
        if not builder.has_custom_attribute("vbd:joint_is_hard"):
            SolverVBD.register_custom_attributes(builder)

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Stage build + lace injection guard (parity with `NewtonLaceVBDManager`): if the
        per-world hooks did not run (no replication queued), inject the single flat world."""
        super().instantiate_builder_from_stage()
        from robobench.suites.deformable.scenes import shoe_knot

        for scene in shoe_knot._ACTIVE:
            if not scene.lace_bodies_w:
                shoe_knot._add_knot_world(cls._builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    @classmethod
    def _prepare_builder_for_finalize(cls, builder) -> None:
        super()._prepare_builder_for_finalize(builder)
        if not builder.body_count:
            return
        builder.rigid_gap = 0.0  # belt-and-braces; every hook shape sets gap=0.0 explicitly
        # VBD requires ModelBuilder.color() before finalize() (the upstream franka+cable
        # example colors its mixed robot+rod builder the same way).
        builder.color()

    # ----- solver construction --------------------------------------------------------------------
    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: LaceCoupledSolverCfg) -> None:
        """Partition the model (rods -> VBD, everything else -> MuJoCo) and build the proxy-
        coupled solver. Gripper proxy bodies are matched by `proxy_body_keywords` on the body
        label (the robot's USD prim path)."""
        from robobench.suites.deformable.scenes import shoe_knot

        assert shoe_knot._ACTIVE, "no active ShoeKnotScene — the lace hook never registered"
        scene = shoe_knot._ACTIVE[0]
        assert scene.lace_bodies_w, "builder hook never ran — no laces were injected into the model"

        lace_bodies = sorted(b for lists in scene.lace_bodies_w.values() for lace in lists for b in lace)
        lace_joints = sorted(j for lists in scene.lace_joints_w.values() for lace in lists for j in lace)
        robot_bodies = sorted(set(range(model.body_count)) - set(lace_bodies))
        robot_joints = sorted(set(range(model.joint_count)) - set(lace_joints))
        if not robot_joints:
            raise RuntimeError(
                "LaceCoupledSolverCfg needs an articulated robot in the scene (MuJoCo rejects a "
                "jointless entry) — use the standalone LaceVBDSolverCfg for robot-less bindings."
            )

        labels = list(model.body_label)
        keywords = tuple(solver_cfg.proxy_body_keywords)
        proxy_bodies = [b for b in robot_bodies if labels[b] and any(k in labels[b] for k in keywords)]
        if not proxy_bodies:
            raise RuntimeError(f"no proxy (gripper) bodies matched {keywords} in {len(robot_bodies)} robot bodies")
        print(
            f"[lace-coupled] mjc: {len(robot_bodies)} bodies / {len(robot_joints)} joints | "
            f"vbd: {len(lace_bodies)} rod bodies / {len(lace_joints)} cable joints | "
            f"proxies: {[labels[b].rsplit('/', 1)[-1] for b in proxy_bodies]}",
            flush=True,
        )

        mj_kwargs = _filter_kwargs(SolverMuJoCo, dict(solver_cfg.mjwarp))
        vbd_kwargs = _filter_kwargs(SolverVBD, dict(solver_cfg.vbd))
        NewtonManager._solver = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, **mj_kwargs),
                    bodies=robot_bodies,
                    joints=robot_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                    bodies=lace_bodies,
                    joints=lace_joints,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=proxy_bodies,
                        mass_scale=solver_cfg.proxy_mass_scale,
                        mode=solver_cfg.proxy_mode,
                        # "latest" contact matching when the rod entry warm-starts AVBD
                        # contacts (SolverVBD(rigid_contact_history=True) rejects
                        # match-index-less buffers): matched contacts keep their k/lambda
                        # warm-start while the GEOMETRY stays fresh. Never "sticky" here —
                        # sticky replays the previous frame's contact geometry, which pins
                        # finger-lace contact points in place under a MOVING gripper and
                        # kills the friction drag a carry needs.
                        collision_pipeline=lambda view: CollisionPipeline(
                            view,
                            broad_phase="explicit",
                            **({"contact_matching": "latest"} if vbd_kwargs.get("rigid_contact_history") else {}),
                        ),
                        collide_interval=solver_cfg.proxy_collide_interval,
                    )
                ],
                iterations=solver_cfg.proxy_iterations,
            ),
        )
        NewtonManager._use_single_state = False  # the coupled step is input/output-state
        NewtonManager._needs_collision_pipeline = True  # rods + robot-static contacts

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Base pipeline + buffers, then preallocate the per-entry filtered contact buffers."""
        super()._initialize_contacts()
        if cls._solver is not None and cls._contacts is not None:
            cls._solver.prepare_contacts(cls._contacts)

    # ----- stepping -------------------------------------------------------------------------------
    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt: float) -> None:
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        # Keep joint_q consistent with the stepped body poses for ALL joints (the upstream
        # example's cadence): articulation views read the robot's joint coordinates, and the
        # manager's masked FK-before-step then round-trips instead of stomping rod poses.
        eval_ik(cls._model, state_1, state_1.joint_q, state_1.joint_qd)

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        return False  # the proxy solver re-plans per step; run eager (RodSimCfg default anyway)


@configclass
class LaceCoupledSolverCfg(NewtonSolverCfg):
    """Coupled MJWarp + VBD-rod solver configuration; `class_type` routes model build + stepping
    to :class:`NewtonLaceCoupledManager`. The dicts are splatted into the sub-solver
    constructors (signature-filtered), so any solver knob works without growing this class."""

    class_type: type[NewtonManager] | str = NewtonLaceCoupledManager

    solver_type: str = "coupledmjwarprod"

    mjwarp: dict = {}
    """SolverMuJoCo kwargs for the robot entry (solver/integrator/cone/iterations/njmax/...).
    ``use_mujoco_contacts`` is forced False — contacts come from the manager's pipeline."""

    vbd: dict = {}
    """SolverVBD kwargs for the rod entry (iterations + the AVBD rigid-contact recipe)."""

    proxy_body_keywords: tuple = ("carriage_", "gripper_")
    """Body-label substrings selecting the gripper bodies exposed to the rod solve (the WXAI
    finger links; both arms match). Extend for other embodiments."""

    proxy_mass_scale: float = 1.0
    """Scale on the MuJoCo-effective mass/inertia assigned to the VBD proxy bodies."""

    proxy_mode: str = "lagged"
    """Proxy transfer mode: "lagged" (sync begin poses + end velocities) or "staggered"."""

    proxy_iterations: int = 1
    """Proxy relaxation passes per coupled substep."""

    proxy_collide_interval: int = 1
    """Proxy collision-pipeline refresh interval [coupled substeps]."""
