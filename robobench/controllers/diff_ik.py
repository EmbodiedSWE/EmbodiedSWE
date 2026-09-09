"""DiffIKController — batched differential IK for a fixed-base arm (joint position targets).

The classic Jacobian-inverse control mode: the action is 6 end-effector pose deltas (the SAME action
convention + scales as `task_space`, so solvers port across modes), a damped-least-squares step maps
the pose error to a joint step, and the result is written as joint POSITION targets — the robot's
actuator PD does the tracking, so the gains live in the `RobotCfg` (Isaac's IK examples pair this
with the Franka HIGH_PD gains, stiffness 400 / damping 80), not here. Wraps Isaac's
`DifferentialIKController` (`isaaclab.controllers`) in relative-pose mode: fully batched tensor math
on the sim device — the throughput swap `pink_ik`'s per-env QP loop points at (this solves ONE chain
per action; multi-frame tasks stay with pink).

The command shaping is the recipe PROVEN in the joint-space bulb port (verified seated on two
seeds) — a stiff position-PD arm needs its command stream tamed or phase-boundary target jumps
become PD torque spikes:
  - DLS damping `ik_lambda = 0.05` (the bulb/dough/tshirt/latte joint experiments' value; Isaac's
    default 0.01 is livelier but less robust near singularities),
  - a PERSISTENT rate-limited command integrator (`max_dq` per control tick): the emitted target
    walks from the previous command, never teleports,
  - a lead clamp (`lead_max`) of the command over the LIVE joints, bounding the stored PD torque
    (position debt) the chain can accumulate,
  - joint-limit clamping.
The integrator makes the controller stateful: `reset()` re-seeds it from the live joints (the bulb
port's "detorque" — zeroing stored position debt — is exactly a `reset()`), and it round-trips via
`get_state`/`set_state`.

World-frame throughout: PhysX Jacobians are world-frame, so the target latches from the LIVE world
EE pose + world-frame delta and the error is solved in world — consistent for a fixed base at ANY
base pose (Isaac's own example converts poses to the base frame, equivalent only while the base
rotation is identity). Heavy imports live in `bind`/`compute` so registration stays app-free.

VELOCITY SEMANTICS (measured 2026-08-24, syringe-cell hold probes): because the target re-latches
from the LIVE pose each tick, this is resolved-rate control — the action is an EE velocity, and
position is its open-loop integral. Zero action means "zero velocity", NOT "hold this pose": a
disturbance (e.g. scene contact) displaces the arm and the relatch ratifies the new pose with zero
error, permanently (0.30 rad elbow shove held forever with zero command lead; `pink_ik`, whose
action is an ABSOLUTE pose, pulled back). Faithful to the DROID/pi teleop stack, which is exactly
this law — but solvers wanting position feedback must carry their own absolute target (the
binding smoke's virtual-carrier reach) or use `pink_ik`. A cfg switch anchoring the latch to the
command integrator's FK (`q_ref`, the bulb_ik recipe) is the future option if "zero action =
hold" is ever needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robobench.core import CONTROLLERS, BaseController, BaseControllerCfg

if TYPE_CHECKING:
    import torch


@dataclass
class DiffIKControllerCfg(BaseControllerCfg):
    """Config for `DiffIKController`: the chain + EE frame (robot-supplied structure) and the solver /
    command-shaping knobs. Action scales default to the `task_space` values so the two mode families
    share an action vocabulary; the shaping defaults are the joint-space bulb port's proven set."""

    ee_body: str = ""  # end-effector frame (a body name); "" -> the articulation's last body
    arm_joint_names: tuple[str, ...] | None = None  # driven joints; None -> all of the robot's joints
    ik_method: str = "dls"  # "pinv" | "svd" | "trans" | "dls" (Isaac's solver menu)
    ik_lambda: float = 0.05  # DLS damping (bulb_ik/dough/tshirt/latte proven; Isaac default 0.01)
    ik_params: dict[str, float] | None = None  # explicit override; None -> dls uses ik_lambda,
    # other methods use Isaac's defaults
    pos_scale: float = 0.02  # action unit -> position step (m)
    rot_scale: float = 0.097  # action unit -> rotation step (rad)
    max_dq: float = 0.06  # commanded-target step per control tick (rad); 0 = no rate limit.
    # 0.06 = 3 rad/s at the Franka's 50 Hz latch (bulb_ik ran 0.06 @ 60 Hz)
    lead_max: float = 0.20  # max commanded lead over the live joints (rad); 0 = no clamp —
    # bounds the PD torque a target jump can wind into the chain
    ema_factor: float = 1.0  # action smoothing: 1 = off; <1 = low-pass on the action stream
    clamp_joint_limits: bool = True  # clamp targets to the joint position limits


@CONTROLLERS.register("diff_ik")
class DiffIKController(BaseController):
    """6-D EE pose delta -> joint position targets for the cfg's chain, via one damped-least-squares
    IK step per control tick (the actuator PD holds the target in between), the emitted command
    rate-limited/lead-clamped through a persistent integrator. Fixed `command_type="position"`."""

    def __init__(self, cfg: DiffIKControllerCfg | None = None) -> None:
        super().__init__(cfg or DiffIKControllerCfg(), command_type="position")

    def _resolve_joints(self, robot: Any) -> Any:
        names = self.cfg.arm_joint_names
        return robot.articulation.find_joints(list(names))[0] if names else list(range(robot.articulation.num_joints))

    @property
    def action_dim(self) -> int:
        return 6

    def bind(self, robot: Any) -> None:
        """Resolve the chain + EE body (base template), then build one batched Isaac IK solver."""
        import torch
        from isaaclab.controllers import DifferentialIKController as _IsaacDiffIK
        from isaaclab.controllers import DifferentialIKControllerCfg as _IsaacCfg

        super().bind(robot)  # sets self._robot, self.joint_ids, self._sink (position), self.limits
        c = self.cfg
        art = robot.articulation
        dev = robot.env.device
        self._ee_idx = art.body_names.index(c.ee_body or art.body_names[-1])
        self._jac_ee_idx = self._ee_idx - 1  # fixed-base: the Jacobian drops the root body
        params = c.ik_params if c.ik_params is not None else ({"lambda_val": c.ik_lambda} if c.ik_method == "dls" else None)
        self._ik = _IsaacDiffIK(
            _IsaacCfg(command_type="pose", use_relative_mode=True, ik_method=c.ik_method, ik_params=params),
            num_envs=robot.env.num_envs,
            device=dev,
        )
        n, n_arm = robot.env.num_envs, len(self.joint_ids)
        # The persistent command integrator (rate limit / lead clamp walk from the PREVIOUS command).
        # Seeded lazily from the live joints per env (`_q_ref_valid`), so reset() = re-seed = detorque.
        self._shaped = c.max_dq > 0.0 or c.lead_max > 0.0
        self._q_ref = torch.zeros(n, n_arm, device=dev)
        self._q_ref_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        # prev-action buffer iff smoothing is on (else absent) — the task_space contract.
        self._prev_action = torch.zeros(n, self.action_dim, device=dev) if c.ema_factor < 1.0 else None

    def reset(self, env_ids: Any = None) -> None:
        """Invalidate the command integrator (next compute re-seeds it from the live joints — this
        zeroes any stored position debt, the bulb port's "detorque") and clear the smoothing buffer."""
        if env_ids is None:
            self._q_ref_valid.zero_()
            if self._prev_action is not None:
                self._prev_action.zero_()
        else:
            self._q_ref_valid[env_ids] = False
            if self._prev_action is not None:
                self._prev_action[env_ids] = 0.0  # assignment, not [idx].zero_() (which hits a copy)

    def get_state(self, env_ids: Any = None) -> dict[str, Any]:
        """The command integrator (+ smoothing buffer when on) — round-trips so replay matches."""
        sel = slice(None) if env_ids is None else env_ids
        state = {"q_ref": self._q_ref[sel].clone(), "q_ref_valid": self._q_ref_valid[sel].clone()}
        if self._prev_action is not None:
            state["prev_action"] = self._prev_action[sel].clone()
        return state

    def set_state(self, state: dict[str, Any], env_ids: Any = None) -> None:
        sel = slice(None) if env_ids is None else env_ids
        if "q_ref" in state:
            self._q_ref[sel] = state["q_ref"]
            self._q_ref_valid[sel] = state["q_ref_valid"]
        if self._prev_action is not None and "prev_action" in state:
            self._prev_action[sel] = state["prev_action"]

    def compute(self, action: "torch.Tensor") -> "torch.Tensor":
        """`action` (n, 6): scaled EE pose delta (world axes; rot = axis-angle, premultiplied — the
        `task_space` convention, which Isaac's `apply_delta_pose` shares). One DLS step from the LIVE
        state -> shaped (n, |chain|) joint position targets."""
        import torch

        art = self.robot.articulation
        c = self.cfg
        jids = self.joint_ids
        if self._prev_action is not None:  # optional action smoothing (stateful only when on)
            action = c.ema_factor * action + (1.0 - c.ema_factor) * self._prev_action
            self._prev_action.copy_(action)

        ee_pos = art.data.body_pos_w[:, self._ee_idx]
        ee_quat = art.data.body_quat_w[:, self._ee_idx]
        dpose = torch.cat((action[:, 0:3] * c.pos_scale, action[:, 3:6] * c.rot_scale), dim=-1)
        self._ik.set_command(dpose, ee_pos=ee_pos, ee_quat=ee_quat)  # latch target = live pose + delta

        jac = art.root_physx_view.get_jacobians()[:, self._jac_ee_idx, 0:6, :][:, :, jids]  # (n, 6, n_arm)
        q = art.data.joint_pos[:, jids]
        q_des = self._ik.compute(ee_pos, ee_quat, jac, q)

        if self._shaped:  # rate limit + lead clamp through the persistent integrator (see module doc)
            q_ref = torch.where(self._q_ref_valid.unsqueeze(-1), self._q_ref, q)  # lazy per-env seed
            if c.max_dq > 0.0:
                q_des = q_ref + (q_des - q_ref).clamp(-c.max_dq, c.max_dq)
            if c.lead_max > 0.0:
                q_des = torch.min(torch.max(q_des, q - c.lead_max), q + c.lead_max)
        if c.clamp_joint_limits:
            q_des = torch.clamp(q_des, self.limits["pos"][..., 0], self.limits["pos"][..., 1])
        if self._shaped:
            self._q_ref.copy_(q_des)
            self._q_ref_valid.fill_(True)
        return q_des
