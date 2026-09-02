"""Task-space (operational-space) torque controllers for a redundant arm.

Two controllers sharing all machinery, differing in one line — how the task PD error becomes a force:
  - `TaskSpaceImpedanceController` ("task_impedance"): f = Kp·e − Kd·ė       (Jacobian-transpose impedance)
  - `OperationalSpaceController`   ("osc"):            f = Λ (Kp·e − Kd·ė)   (inertia-shaped)
Both then apply τ = Jᵀ f + nullspace posture, clamped. `Λ = (J M⁻¹ Jᵀ)⁻¹` and the nullspace projector
are computed each step from the live `J`, `M` (not tuned); both forms need `Λ` for the nullspace.

Robot-general: the driven arm joints and the end-effector frame are config (defaults: all joints / the
articulation's tip), so it drives any fixed-base arm — a robot overrides them when it has extra DOFs
(e.g. a gripper). Stateless unless `ema_factor < 1` (action smoothing keeps a prev-action buffer,
cleared by `reset()` and round-tripped by `get_state`/`set_state`). `command_type` is "effort" (arm
actuators must be torque-mode). Assumes a fixed base (the Jacobian drops the root, so EE col = `idx−1`).
Heavy math is imported in-method so registration stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robobench.core import CONTROLLERS, BaseController, BaseControllerCfg, info, tunable

if TYPE_CHECKING:
    import torch


@dataclass
class TaskSpaceControllerCfg(BaseControllerCfg):
    """Shared config. Robot-specific structure is `info`; gains / action scaling are `tunable`."""

    ee_body: str = info("")  # end-effector frame (a body name); "" -> the articulation's last body
    arm_joint_names: tuple[str, ...] | None = info(None)  # driven joints; None -> all of the robot's joints
    task_prop_gains: tuple[float, ...] = tunable((100.0, 100.0, 100.0, 30.0, 30.0, 30.0))  # task stiffness [xyz, rpy]
    task_deriv_gains: tuple[float, ...] = info(())  # () -> critical damping (2√Kp)
    pos_scale: float = tunable(0.02)  # action unit -> position step (m)
    rot_scale: float = tunable(0.097)  # action unit -> rotation step (rad)
    unidirectional_rot: bool = info(False)  # clamp the yaw action to one sign (tighten-only)
    ema_factor: float = tunable(1.0)  # action smoothing: 1 = off (stateless); <1 = low-pass (stateful)
    nullspace_dof_pos: tuple[float, ...] = info(())  # posture target; () -> the arm's default joint pose
    kp_null: float = tunable(10.0)  # nullspace posture stiffness
    kd_null: float = tunable(6.3246)  # nullspace posture damping
    torque_limit: float = tunable(100.0)  # per-joint torque clamp (N·m)


class _TaskSpaceController(BaseController):
    """Shared base: action → EE pose target → task force → `Jᵀ` torque + nullspace posture. Subclasses
    implement only `_task_force`. Not registered."""

    def __init__(self, cfg: TaskSpaceControllerCfg | None = None) -> None:
        super().__init__(cfg or TaskSpaceControllerCfg(), command_type="effort")

    def _resolve_joints(self, robot: Any) -> Any:
        names = self.cfg.arm_joint_names
        return robot.articulation.find_joints(list(names))[0] if names else list(range(robot.articulation.num_joints))

    @property
    def action_dim(self) -> int:
        return 6

    def bind(self, robot: Any) -> None:
        """Resolve the arm joints + EE body and pre-build the gain / posture tensors on the sim device."""
        import torch

        super().bind(robot)  # sets self._robot, self.joint_ids, self._sink (effort), self.limits
        c = self.cfg
        art = robot.articulation
        dev = robot.env.device
        self._ee_idx = art.body_names.index(c.ee_body or art.body_names[-1])
        self._jac_ee_idx = self._ee_idx - 1  # fixed-base: the Jacobian drops the root body
        kp = torch.tensor(c.task_prop_gains, device=dev)
        self._kp = kp  # (6,)
        self._kd = torch.tensor(c.task_deriv_gains, device=dev) if c.task_deriv_gains else 2.0 * kp.sqrt()
        self._q_default = (
            torch.tensor(c.nullspace_dof_pos, device=dev)
            if c.nullspace_dof_pos
            else art.data.default_joint_pos[0, self.joint_ids].clone()
        )
        self._n_arm = len(self.joint_ids)
        # prev-action buffer iff smoothing is on (else stateless)
        self._prev_action = torch.zeros(robot.env.num_envs, self.action_dim, device=dev) if c.ema_factor < 1.0 else None
        # EE pose target: latched from the action at the control rate, held while the torque tracks it.
        self._target_pos: torch.Tensor | None = None
        self._target_quat: torch.Tensor | None = None

    def reset(self, env_ids: Any = None) -> None:
        """Clear the action-smoothing buffer (no-op when smoothing is off)."""
        if self._prev_action is not None:
            if env_ids is None:
                self._prev_action.zero_()
            else:
                self._prev_action[env_ids] = 0.0  # assignment, not [idx].zero_() (which hits a copy)

    def get_state(self, env_ids: Any = None) -> dict[str, Any]:
        """The smoothing buffer (empty when smoothing is off -> stateless)."""
        if self._prev_action is None:
            return {}
        buf = self._prev_action if env_ids is None else self._prev_action[env_ids]
        return {"prev_action": buf.clone()}

    def set_state(self, state: dict[str, Any], env_ids: Any = None) -> None:
        if self._prev_action is not None and "prev_action" in state:
            if env_ids is None:
                self._prev_action.copy_(state["prev_action"])
            else:
                self._prev_action[env_ids] = state["prev_action"]

    def _task_force(self, pose_error, ee_vel, lambda_task):
        """Map task error + velocity (and Λ) to a 6-D task force. Overridden per form."""
        raise NotImplementedError

    # ----- the two rates, decoupled ------------------------------------------------------------
    # A torque law is state feedback: it must recompute every physics step from the live state, even
    # while the agent acts more slowly. So we split the work and override `apply`:
    #   - `_latch_target` (action -> EE pose target) runs at the CONTROL rate (`control_period`).
    #   - `_compute_torque` (target + live state -> τ) runs EVERY physics substep.
    # `compute` chains both (one-shot — for direct calls / tests / `control_period == 1`).

    def _latch_target(self, action: "torch.Tensor") -> None:
        """Latch the EE pose target (current pose + scaled delta) from `action`, held until the next
        latch. EMA smoothing lives here so it runs at the control rate, not per physics step."""
        import torch
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul

        art = self._robot.articulation
        c = self.cfg
        if self._prev_action is not None:  # optional action smoothing (stateful only when on)
            action = c.ema_factor * action + (1.0 - c.ema_factor) * self._prev_action
            self._prev_action.copy_(action)

        ee_pos = art.data.body_pos_w[:, self._ee_idx]
        ee_quat = art.data.body_quat_w[:, self._ee_idx]
        self._target_pos = ee_pos + action[:, 0:3] * c.pos_scale
        rot_action = action[:, 3:6].clone()
        if c.unidirectional_rot:
            rot_action[:, 2] = -(rot_action[:, 2] + 1.0) * 0.5  # [-1,1] -> [-1,0]: tighten-only
        rot_action = rot_action * c.rot_scale
        angle = rot_action.norm(dim=-1)
        axis = rot_action / angle.clamp_min(1e-6).unsqueeze(-1)
        self._target_quat = quat_mul(quat_from_angle_axis(angle, axis), ee_quat)

    def _compute_torque(self) -> "torch.Tensor":
        """Joint torque toward the latched target from the LIVE state — runs every physics step."""
        import torch
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        art = self._robot.articulation
        c = self.cfg
        jids = self.joint_ids

        # current end-effector pose / velocity (world)
        ee_pos = art.data.body_pos_w[:, self._ee_idx]
        ee_quat = art.data.body_quat_w[:, self._ee_idx]
        ee_vel = torch.cat((art.data.body_lin_vel_w[:, self._ee_idx], art.data.body_ang_vel_w[:, self._ee_idx]), dim=-1)

        # task-space pose error to the latched target (pos + axis-angle, shortest path vs the LIVE pose)
        target_quat = torch.where((self._target_quat * ee_quat).sum(-1, keepdim=True) >= 0, self._target_quat, -self._target_quat)
        quat_error = quat_mul(target_quat, quat_conjugate(ee_quat))
        pose_error = torch.cat((self._target_pos - ee_pos, axis_angle_from_quat(quat_error)), dim=-1)  # (n, 6)

        # Jacobian, mass matrix, op-space inertia Λ (all from live state)
        jac = art.root_physx_view.get_jacobians()[:, self._jac_ee_idx, 0:6, :][:, :, jids]  # (n, 6, n_arm)
        jac_T = jac.transpose(1, 2)
        mass = art.root_physx_view.get_generalized_mass_matrices()[:, jids][:, :, jids]  # (n, n_arm, n_arm)
        mass_inv = torch.inverse(mass)
        lambda_task = torch.inverse(jac @ mass_inv @ jac_T)  # Λ = (J M⁻¹ Jᵀ)⁻¹

        # task torque (the two forms differ in `_task_force`) + dynamically-consistent nullspace posture
        tau = (jac_T @ self._task_force(pose_error, ee_vel, lambda_task).unsqueeze(-1)).squeeze(-1)
        dof_pos, dof_vel = art.data.joint_pos[:, jids], art.data.joint_vel[:, jids]
        to_default = (self._q_default - dof_pos + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]
        u_null = (mass @ (c.kp_null * to_default - c.kd_null * dof_vel).unsqueeze(-1)).squeeze(-1)
        eye = torch.eye(self._n_arm, device=tau.device).unsqueeze(0)
        tau_null = ((eye - jac_T @ (lambda_task @ jac @ mass_inv)) @ u_null.unsqueeze(-1)).squeeze(-1)

        return torch.clamp(tau + tau_null, -c.torque_limit, c.torque_limit)

    def compute(self, action: "torch.Tensor") -> "torch.Tensor":
        """One-shot latch-then-torque (pure, for direct calls / tests). The per-step path is `apply`,
        which latches only on the control subdivision but recomputes the torque every physics step."""
        self._latch_target(action)
        return self._compute_torque()

    def apply(self, action: "torch.Tensor", substep: int = 0) -> None:
        """Decouple the two rates: latch a fresh target only on this controller's control subdivision
        (`control_period`), but recompute + write the torque toward the latched target EVERY physics
        step. With `control_period == 1` this latches every step = the plain `compute` path."""
        if substep % self._control_period == 0 or self._target_pos is None:
            self._latch_target(action)
        self._sink(self._compute_torque(), self.joint_ids)


@CONTROLLERS.register("task_impedance")
class TaskSpaceImpedanceController(_TaskSpaceController):
    """Jacobian-transpose task-space impedance: gains applied directly as the wrench, no inertia
    shaping. The form Isaac's Factory tasks use."""

    def _task_force(self, pose_error, ee_vel, lambda_task):
        return self._kp * pose_error - self._kd * ee_vel


@CONTROLLERS.register("osc")
class OperationalSpaceController(_TaskSpaceController):
    """Operational-space control: the PD wrench is shaped by the op-space inertia Λ, decoupling the
    task axes to unit-mass behaviour."""

    def _task_force(self, pose_error, ee_vel, lambda_task):
        return (lambda_task @ (self._kp * pose_error - self._kd * ee_vel).unsqueeze(-1)).squeeze(-1)
