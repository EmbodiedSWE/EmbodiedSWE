"""NewtonCoupledMJWarpMPMManager — MJWarp rigid dynamics + implicit-MPM liquids on ONE Newton model.

The Phase 2b substrate for the pouring suite: :class:`SolverMuJoCo` advances the articulations
(real gravity, actuator PD, MuJoCo-internal rigid contacts) at ``dt / num_substeps``, then
:class:`SolverImplicitMPM` advances the liquids ONCE per physics tick at the full ``dt``, reading
the post-rigid ``state.body_q`` — Newton's ``examples/mpm/example_mpm_anymal.py`` recipe hoisted
into the IsaacLab Newton-manager contract (``_step_solver``'s docstring blesses multi-solver
batching; we override ``_run_solver_substeps`` because the MPM cadence is per-tick, not
per-substep).

Coupling is ONE-WAY (rigid -> fluid): after construction the MPM collider set is re-registered
with ``body_mass = zeros`` — the override Newton documents for treating all bodies as kinematic
colliders — so every rigid body (dynamic Franka links included) is an infinite-mass collider for
the liquid. This also keeps the per-step compliant-collider rigidity assembly off; without it the
liquid would press against "recoiling" bodies whose recoil MuJoCo never receives. Fluid -> rigid
impulses remain available via ``collect_collider_impulses`` for a future two-way phase.

KINEMATIC-flagged rigid bodies (the scripted vessels) are GHOSTED from the rigid solver: their
shapes drop ``COLLIDE_SHAPES`` before finalize while keeping ``COLLIDE_PARTICLES`` (still MPM
colliders). The barista choreography overlaps the pitcher lip with the mug mouth by design — as
MuJoCo geoms the two convexified hulls would grind constraint rows against each other (both
immovable) for nothing. Unlike ``NewtonMPMManager`` their masses are NOT zeroed: they carry
auto-added free joints and MuJoCo requires positive inertia on jointed bodies; immovability comes
from the KINEMATIC flag's 1e10 armature, and scripted `write_root_link_pose` writes reach MuJoCo
through the per-step ``joint_q -> qpos`` push.

Phase 2c-a extends this with DYNAMIC vessels: bodies carrying ``/rigidproxy/`` shapes collide in
MuJoCo through those concave proxies ONLY (the interior trimesh turns MPM-only instead of being
convexified into a mouth-filling hull), and hand<->vessel MuJoCo equality WELDS — created
disabled at build from ``MJWarpMPMSolverCfg.weld_specs`` — engage at the measured grasp pose via
:meth:`NewtonCoupledMJWarpMPMManager.set_weld`, so the arm carries the real vessel mass.

Runs ONLY under the Newton venv with the app up (imports isaaclab_newton at module level); import
it lazily from `MpmSimCfg.to_isaaclab`.
"""

from __future__ import annotations

import inspect

import warp as wp
from newton import BodyFlags, EqType, Model, ModelBuilder, ShapeFlags, eval_fk
from newton.solvers import SolverImplicitMPM, SolverMuJoCo, SolverNotifyFlags
from warp.fem import TemporaryStore

from isaaclab.physics import PhysicsManager
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.mpm_manager import _make_solver_config
from isaaclab_newton.physics.mpm_manager_cfg import MPMSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg


class NewtonCoupledMJWarpMPMManager(NewtonMJWarpManager):
    """MJWarp + implicit-MPM coupled manager (one-way rigid -> fluid).

    Inherits from :class:`NewtonMJWarpManager` so the canonical ``NewtonManager._solver`` slot
    holds the :class:`SolverMuJoCo` instance — model-change notifications, contact-sensor
    ``update_contacts`` and the convergence debug logging all keep working. The MPM half lives on
    :attr:`_mpm_solver` and is stepped by our ``_run_solver_substeps`` override.
    """

    _mpm_solver: SolverImplicitMPM | None = None
    _project_outside_colliders: bool = False

    # ----- builder hooks --------------------------------------------------------------------------
    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the MPM per-particle custom attributes (idempotent — the hook runs once per
        source builder in the cloner and again in start_simulation)."""
        if not builder.has_custom_attribute("mpm:young_modulus"):
            SolverImplicitMPM.register_custom_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Shape-flag routing + builder-time welds. Three rules, in order:

        1. KINEMATIC-body ghosting (Phase 2b, unchanged): clear COLLIDE_SHAPES on their shapes
           (COLLIDE_PARTICLES stays — they remain MPM colliders). Masses are intentionally KEPT,
           unlike NewtonMPMManager: MuJoCo needs positive inertia on their auto-added free
           joints, and the MPM side is neutralized wholesale via the
           setup_collider(body_mass=zeros) override in _build_solver.
        2. Rigid-proxy routing (Phase 2c-a), by `builder.shape_label` prim path: shapes under a
           ``/rigidproxy/`` scope are RIGID-only (drop COLLIDE_PARTICLES) — except ``handle``
           capsules, which keep BOTH flags (grasp target + the milk must not pass the visual
           handle). Every OTHER shape on a body that carries proxies (the concave interior
           trimesh MuJoCo would convexify into a mouth-filling hull) turns MPM-only.
        3. Weld rows (Phase 2c-a): for each ``(label, body1_suffix, body2_suffix)`` in the solver
           cfg's ``weld_specs``, add a DISABLED MuJoCo equality weld — activated at grasp time
           via :meth:`set_weld`, which writes the measured relative pose first.
        """
        kinematic = int(BodyFlags.KINEMATIC)
        no_rigid_collision = ~int(ShapeFlags.COLLIDE_SHAPES)
        no_particle_collision = ~int(ShapeFlags.COLLIDE_PARTICLES)
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx >= 0 and int(builder.body_flags[body_idx]) & kinematic:
                builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & no_rigid_collision

        proxy_bodies: set[int] = set()
        for shape_idx, label in enumerate(builder.shape_label):
            if label and "/rigidproxy/" in label:
                body_idx = builder.shape_body[shape_idx]
                if body_idx >= 0:
                    proxy_bodies.add(int(body_idx))
                if not label.rsplit("/", 1)[-1].startswith("handle"):
                    builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & no_particle_collision
        if proxy_bodies:
            for shape_idx, label in enumerate(builder.shape_label):
                if int(builder.shape_body[shape_idx]) in proxy_bodies and not (label and "/rigidproxy/" in label):
                    builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & no_rigid_collision

        solver_cfg = PhysicsManager._cfg.solver_cfg if PhysicsManager._cfg is not None else None
        weld_labels = []
        for label, suffix1, suffix2 in getattr(solver_cfg, "weld_specs", None) or []:
            builder.add_equality_constraint(
                EqType.WELD,
                body1=cls._find_body(builder, suffix1),
                body2=cls._find_body(builder, suffix2),
                label=label,
                enabled=False,
            )
            weld_labels.append(label)
        if proxy_bodies or weld_labels:
            n_proxy = sum(1 for lb in builder.shape_label if lb and "/rigidproxy/" in lb)
            n_mpm_only = sum(
                1
                for i, lb in enumerate(builder.shape_label)
                if int(builder.shape_body[i]) in proxy_bodies and not (lb and "/rigidproxy/" in lb)
            )
            print(
                f"[coupled] rigid proxies: {n_proxy} shapes on {len(proxy_bodies)} dynamic bodies"
                f" (interior shapes -> MPM-only: {n_mpm_only}) | welds (disabled): {weld_labels}",
                flush=True,
            )

    @staticmethod
    def _find_body(builder: ModelBuilder, suffix: str) -> int:
        matches = [i for i, key in enumerate(builder.body_label) if key and str(key).endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"weld body suffix {suffix!r} matched {len(matches)} bodies: {matches}")
        return matches[0]

    # ----- solver construction --------------------------------------------------------------------
    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpMPMSolverCfg) -> None:
        """Build SolverMuJoCo (canonical ``_solver``) + SolverImplicitMPM over the same model."""
        rigid_cfg = solver_cfg.rigid_solver_cfg
        if not rigid_cfg.use_mujoco_contacts:
            raise ValueError(
                "MJWarpMPMSolverCfg: use_mujoco_contacts=False (Newton CollisionPipeline contacts)"
                " is not supported by the coupled manager — MPM rasterizes its own colliders and"
                " the pipeline would double-drive rigid contacts."
            )
        if PhysicsManager._cfg is not None and PhysicsManager._cfg.collision_cfg is not None:
            # Same cross-validation as NewtonMJWarpManager._build_solver (which this overrides):
            # without it a user-supplied collision pipeline cfg would be silently dead config.
            raise ValueError("MJWarpMPMSolverCfg: NewtonCfg.collision_cfg cannot be set — MuJoCo collides internally.")
        # Rigid half — same signature-filtered construction as NewtonMJWarpManager._build_solver.
        ignored = {"class_type", "solver_type", "ls_parallel"}
        valid = set(inspect.signature(SolverMuJoCo.__init__).parameters) - {"self", "model"} - ignored
        kwargs = {k: v for k, v in rigid_cfg.to_dict().items() if k in valid}
        NewtonManager._solver = SolverMuJoCo(model, **kwargs)

        # MPM half — same construction as NewtonMPMManager._build_solver.
        mpm_cfg = solver_cfg.mpm_solver_cfg
        cls._mpm_solver = SolverImplicitMPM(model, _make_solver_config(mpm_cfg), temporary_store=TemporaryStore())
        cls._project_outside_colliders = mpm_cfg.project_outside_colliders
        # One-way coupling: re-register every collider as kinematic/infinite-mass (Newton's
        # documented override), and seed collider poses (-> body_q_prev for the "backward"
        # velocity mode) from the FK'd state rather than the identity-posed model arrays.
        cls._mpm_solver.setup_collider(
            body_mass=wp.zeros_like(model.body_mass),
            body_q=NewtonManager._state_0.body_q,
        )

        NewtonManager._use_single_state = True  # both sub-solvers step in place on state_0
        NewtonManager._needs_collision_pipeline = False  # MuJoCo-internal contacts + MPM colliders
        # Nothing else refreshes body_q for the MPM collider read after resets / kinematic vessel
        # writes, so the pre-step masked eval_fk must run (same rationale as NewtonMPMManager).
        NewtonManager._needs_fk_before_step = True

    # ----- stepping -------------------------------------------------------------------------------
    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """``num_substeps`` MuJoCo substeps at ``_solver_dt``, then ONE implicit-MPM step at the
        full tick dt reading the post-rigid ``state_0.body_q`` (the anymal-example cadence: the
        implicit MPM solve is unconditionally stable and much more expensive than MJWarp)."""
        del contacts  # MuJoCo collides internally; SolverImplicitMPM ignores the contacts arg
        for _ in range(cls._num_substeps):
            cls._solver.step(cls._state_0, cls._state_0, cls._control, None, cls._solver_dt)
            cls._state_0.clear_forces()
        mpm_dt = cls._solver_dt * cls._num_substeps
        cls._mpm_solver.step(cls._state_0, cls._state_0, None, None, mpm_dt)
        if cls._project_outside_colliders:
            cls._mpm_solver.project_outside(cls._state_0, cls._state_0, mpm_dt)

    @classmethod
    def step(cls) -> None:
        """Fan model-change notifications out to the MPM half before the base step drains them
        into the canonical ``_solver`` (SolverMuJoCo) only. SolverImplicitMPM no-ops on all but
        particle-material changes, so blind forwarding is safe (and idempotent if the base step
        early-returns while not playing)."""
        if cls._model_changes and cls._mpm_solver is not None:
            with wp.ScopedDevice(PhysicsManager._device):
                for change in cls._model_changes:
                    cls._mpm_solver.notify_model_changed(change)
        super().step()

    # ----- welds (Phase 2c-a) ---------------------------------------------------------------------
    @classmethod
    def set_weld(cls, label: str, active: bool) -> None:
        """Toggle a builder-time equality weld by label. On ACTIVATION the current relative pose
        ``inv(X_body1) * X_body2`` is measured from ``state_0.body_q`` and written as the weld
        target first, so engaging never snaps — call it with the hands at the grasp pose, between
        steps (the writes land in Newton model arrays that the solver re-reads OUTSIDE the CUDA
        graph on the queued CONSTRAINT_PROPERTIES notification)."""
        import torch

        import isaaclab.utils.math as math_utils

        model = cls._model
        labels = list(model.mujoco.equality_constraint_label)
        if label not in labels:
            raise ValueError(f"unknown weld label {label!r}; builder welds: {labels}")
        eq_idx = labels.index(label)
        device = model.device
        if active:
            body_q = wp.to_torch(cls._state_0.body_q)  # (nbody, 7) [pos, quat-xyzw], warp layout
            b1 = int(model.mujoco.equality_constraint_body1.numpy()[eq_idx])
            b2 = int(model.mujoco.equality_constraint_body2.numpy()[eq_idx])
            p, q = math_utils.subtract_frame_transforms(
                body_q[b1, :3][None], body_q[b1, 3:][None], body_q[b2, :3][None], body_q[b2, 3:][None]
            )
            rel = torch.cat([p, q], dim=-1)[0].tolist()
            staged = wp.array(
                [wp.transform(wp.vec3(*rel[:3]), wp.quat(*rel[3:]))], dtype=wp.transform, device=device
            )
            wp.copy(model.mujoco.equality_constraint_relpose, staged, dest_offset=eq_idx, count=1)
        staged_en = wp.array([bool(active)], dtype=wp.bool, device=device)
        wp.copy(model.mujoco.equality_constraint_enabled, staged_en, dest_offset=eq_idx, count=1)
        cls.add_model_change(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

    @classmethod
    def resync_collider_history(cls) -> None:
        """Re-seed the MPM collider pose history from the CURRENT state — call after any
        TELEPORTING write (an episodic `env.reset()` that moves arms/vessels). In the suite's
        `collider_velocity_mode="backward"` the solver finite-differences `body_q` against the
        previous step's snapshot, so an unsynced teleport reads as a (jump/dt) collider velocity
        that violently kicks the liquid for one tick. Do NOT call per scripted per-tick pose
        write — the finite difference IS the vessel velocity the liquid must see."""
        if cls._mpm_solver is not None and cls._state_0 is not None:
            # Reset writes land in joint_q immediately, but body_q only refreshes at the NEXT
            # step's masked FK — run a full FK first so the snapshot holds the POST-teleport poses.
            eval_fk(cls._model, cls._state_0.joint_q, cls._state_0.joint_qd, cls._state_0, None)
            cls._mpm_solver._last_step_data.save_collider_current_position(cls._state_0.body_q)

    # ----- capture / teardown ---------------------------------------------------------------------
    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Graph-capturable only with a fixed MPM grid (sparse/dense grids reallocate)."""
        return cls._mpm_solver is not None and cls._mpm_solver.grid_type == "fixed"

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._mpm_solver = None
        cls._project_outside_colliders = False


@configclass
class MJWarpMPMSolverCfg(NewtonSolverCfg):
    """Coupled MJWarp + implicit-MPM solver configuration.

    Selects :class:`NewtonCoupledMJWarpMPMManager`; carries one sub-cfg per solver half. The
    rigid half must keep ``use_mujoco_contacts=True`` (MuJoCo-internal collision)."""

    class_type: type[NewtonManager] | str = NewtonCoupledMJWarpMPMManager

    solver_type: str = "coupledmjwarpmpm"

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp sub-solver configuration (articulations + rigid contacts)."""

    mpm_solver_cfg: MPMSolverCfg = MPMSolverCfg()
    """Implicit-MPM sub-solver configuration (particle liquids)."""

    weld_specs: list = []
    """Builder-time MuJoCo equality welds ``[(label, body1 suffix, body2 suffix)]`` (Phase 2c-a).

    Suffixes match against ``builder.body_label`` prim paths (must match exactly one body each).
    Rows are created DISABLED; toggle at runtime with
    :meth:`NewtonCoupledMJWarpMPMManager.set_weld`."""
