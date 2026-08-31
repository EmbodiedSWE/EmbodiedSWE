"""LocoPolicyController — legs driven by a FROZEN locomotion policy, commanded by base velocity.

The loco half of loco-manipulation. The action is not a joint target and not an end-effector pose:
it is a **base command** `[vx, vy, wz, hip_height]` — where the robot should go, not how to move its
legs — and a frozen neural policy turns that into position targets for the 12 leg DOFs, closing the
balance loop at the control rate. Everything above the pelvis is somebody else's controller; this is
a leaf, and `composite` is what makes the pair one robot (its docstring has named this case from the
start: "loco-manip = arms by IK **+** legs by a frozen policy").

    action = [vx, vy, wz, hip_height]      (m/s, m/s, rad/s, m — in the robot's own base frame)
      -> policy([command(4) | observation(79)])  -> 12 raw leg residuals
      -> target = 0.25 * residual + default_joint_pos[legs]

THE POLICY. The shipped default is NVIDIA's **AGILE** G1 checkpoint, vendored at
`robots/assets/g1/policies/agile_locomotion.pt` (see `robots/assets/fetch_g1_locomotion.py` for
provenance and the measured 4+79 -> 12 signature). It is TorchScript, stateless and deterministic —
the only memory in the loop is the `last_action` term, which this controller owns and feeds back.
Nothing here is G1-specific except the defaults the robot passes in: `policy_path`, the joint
regexes and the observation spec are all cfg, so another embodiment's checkpoint — or a HOMIE /
self-trained one — is a config swap, not a code change.

THE OBSERVATION IS A CONTRACT WITH THE CHECKPOINT. All 79 numbers, in this order, are Isaac Lab's
`AgileTeacherPolicyObservationsCfg` (`isaaclab_tasks/manager_based/locomanipulation/pick_place/
configs/agile_locomotion_observation_cfg.py`), reproduced here straight off `articulation.data` so
robobench owes nothing to isaaclab's manager stack:

    base_lin_vel        3    root_lin_vel_b          (pelvis frame)
    base_ang_vel        3    root_ang_vel_b
    projected_gravity   3    projected_gravity_b
    joint_pos          29    joint_pos - default_joint_pos      over the 29 body joints
    joint_vel          29    (joint_vel - default_joint_vel) * 0.1
    last_action        12    the PREVIOUS tick's RAW policy output (pre-scale, pre-offset)

Two ordering facts that will silently wreck the gait if you get them wrong — a mismatched
observation does not raise anywhere, the robot just twitches and falls over:

  - The 29 body joints are **articulation DOF order**, not regex order. Isaac resolves observation
    joints through `SceneEntityCfg` with `preserve_order=False`, i.e. sorted by joint index; robobench's
    `find_joints` defaults to the same. Every joint but the 14 hand DOFs is in there.
  - `last_action` is the policy's **raw** output (`AgileBasedLowerBodyAction._raw_actions`), before
    `output_scale` and before the default-pose offset. Feeding back the scaled target instead is a
    plausible-looking bug that just degrades the gait.

`bind` asserts the resolved joint counts and the policy's input width against each other, so a
re-vendored checkpoint or a re-authored USD fails loudly at build instead of at the first step.

MEASURED TRACKING (G1 + the vendored AGILE checkpoint, 4 s per trial from a settled stand on flat
ground, hip_height 0.72; achieved rates in the robot's own start frame). The command is a REQUEST,
not a guarantee — a solver must close its loop on the measured base pose, never dead-reckon:

    cmd (vx, vy, wz)   achieved (vx, vy, wz)        note
    (0.20, 0,    0)    (+0.155, +0.005, +0.013)     78 % of command
    (0.40, 0,    0)    (+0.368, -0.007, -0.003)     92 %
    (0.60, 0,    0)    (+0.543, -0.016, -0.019)     91 %
    (-0.30, 0,   0)    (-0.256, -0.011, +0.009)     85 %, backward walking works
    (0, 0.20,    0)    (+0.004, +0.151, +0.008)     76 %, strafing works
    (0, 0.40,    0)    (+0.009, +0.377, -0.018)     94 %
    (0, 0,    0.20)    (+0.000, +0.003, +0.019)     10 %  <-- see below
    (0, 0,    0.35)    (+0.001, +0.006, +0.035)     10 %  <--
    (0, 0,    0.50)    (+0.003, +0.008, +0.041)      8 %  <--
    (0, 0,   -0.50)    (-0.000, -0.010, -0.072)     14 %  <--
    (0.30, 0, 0.30)    (+0.194, +0.161, +0.362)     121 % on yaw, walking an arc

  - Translation tracks well in every direction and the axes barely cross-couple, so `vx`/`vy` are
    the reliable way to move the base. Under-tracking grows as the command shrinks (a 0.2 command
    is the weakest), so prefer a moderate command over a timid one.
  - **Turn-in-place barely turns.** From a standing start, a pure `wz` yields under a sixth of what
    is asked. Add forward speed and yaw tracks fully — the last row turns faster than commanded.
    This is a WALKING policy: it steers while stepping and has almost no pivot gait. Plan turns as
    arcs, or sidestep with `vy` instead of turning at all.
  - Pelvis height held 0.722-0.728 m against a 0.72 command through every trial, including transit.

Heavy imports (torch / isaaclab) are deferred so importing this module — and registering the
controller — stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robobench.core import CONTROLLERS, BaseController, BaseControllerCfg

if TYPE_CHECKING:
    import torch


@dataclass
class LocoPolicyControllerCfg(BaseControllerCfg):
    """Config for `LocoPolicyController`. `policy_path` + the two joint sets + `obs_dim` are the
    structural wiring the robot supplies (only it knows its kinematics and which checkpoint matches);
    the scales and clips are feedback-law dials an agent may retune."""

    policy_path: str = ""  # TorchScript checkpoint; required
    #: The DOFs the policy drives (regexes). For the G1: hips + knees + ankles = 12.
    joint_names: tuple[str, ...] = ()
    #: The DOFs that appear in the observation (regexes), resolved in ARTICULATION order. For the
    #: G1: every body joint = all 29 (arms, waist, legs) — i.e. everything but the hands.
    obs_joint_names: tuple[str, ...] = ()
    #: Raw policy output -> joint-target residual, added to the legs' default pose. NVIDIA's value.
    output_scale: float = 0.25
    #: Per-component clip on the incoming base command, `(lo, hi)` for vx / vy / wz / hip_height.
    #: The linear/angular bands are the scales Isaac Lab's own VR teleop pipeline drives this
    #: checkpoint at (`LocomotionRootCmdRetargeterConfig`: movement 0.5, rotation 0.35, initial hip
    #: height 0.72), widened a little — evidence of the range it was flown at, not a proof of its
    #: training bounds. Clipping here is what keeps a solver's overshoot from asking for a gait the
    #: policy never learned; widen it if you want to explore, but expect falls.
    command_clip: tuple[tuple[float, float], ...] = (
        (-0.6, 0.6),  # vx  forward / back (m/s)
        (-0.4, 0.4),  # vy  strafe (m/s)
        (-0.6, 0.6),  # wz  turn (rad/s)
        (0.50, 0.80),  # hip_height (m)
    )
    #: Sanity bounds asserted at bind: (n_drive, n_obs_joints, policy_in, policy_out).
    expect: tuple[int, int, int, int] = (12, 29, 83, 12)


@CONTROLLERS.register("loco_policy")
class LocoPolicyController(BaseController):
    """Base-velocity command -> leg joint position targets, through a frozen TorchScript policy.
    `action_dim` is 4 regardless of how many DOFs the policy drives — the action is a *command*, not
    a joint vector. Stateful: it carries the previous raw output as part of the observation, so
    `reset` / `get_state` / `set_state` are real, not no-ops."""

    cfg: LocoPolicyControllerCfg

    def __init__(self, cfg: LocoPolicyControllerCfg) -> None:
        # Position: the legs are G1_29DOF_CFG's DCMotor actuators, which run the PD in PhysX. The
        # policy was trained against exactly those gains, so the robot must not retune them.
        super().__init__(cfg, command_type="position")
        self._policy: Any = None
        self._obs_joint_ids: Any = None
        self._offset: Any = None  # default joint pos of the driven legs, (num_envs, n_drive)
        self._last_action: Any = None  # (num_envs, n_drive) RAW policy output, the obs feedback term
        self._clip_lo: Any = None
        self._clip_hi: Any = None

    # ----- bind ---------------------------------------------------------------------------------
    def _resolve_joints(self, robot: Any) -> Any:
        if not self.cfg.policy_path:
            raise ValueError("LocoPolicyControllerCfg.policy_path is required (the frozen checkpoint)")
        return robot.articulation.find_joints(self.cfg.joint_names)[0]

    def bind(self, robot: Any) -> None:
        """Resolve both joint sets, load the checkpoint onto the sim device, capture the legs'
        default pose as the output offset, and assert every width against the policy."""
        import torch

        super().bind(robot)  # -> joint_ids, sink, limits, control_period
        art = robot.articulation
        device = robot.env.device
        n_drive, n_obs_j, p_in, p_out = self.cfg.expect

        self._obs_joint_ids = art.find_joints(self.cfg.obs_joint_names)[0]  # ARTICULATION order
        if len(self.joint_ids) != n_drive or len(self._obs_joint_ids) != n_obs_j:
            raise ValueError(
                f"{type(self).__name__}: joint resolution does not match the checkpoint — drove "
                f"{len(self.joint_ids)} (expected {n_drive}), observed {len(self._obs_joint_ids)} "
                f"(expected {n_obs_j}). The articulation is {art.num_joints} DOF: {art.joint_names}"
            )

        self._policy = torch.jit.load(self.cfg.policy_path, map_location=device)
        self._policy.eval()
        # Width check by construction, not by trust: one forward pass at the shape we will use.
        probe = torch.zeros(1, self.action_dim + self._obs_dim, device=device)
        with torch.no_grad():
            out = self._policy(probe)
        if probe.shape[1] != p_in or tuple(out.shape) != (1, p_out):
            raise ValueError(
                f"{self.cfg.policy_path}: expected {p_in} -> {p_out}, got {probe.shape[1]} -> "
                f"{tuple(out.shape)[1]}. Re-vendor with robots/assets/fetch_g1_locomotion.py"
            )

        self._offset = art.data.default_joint_pos[:, self.joint_ids].clone()
        self._last_action = torch.zeros(robot.env.num_envs, len(self.joint_ids), device=device)
        clip = torch.tensor(self.cfg.command_clip, dtype=torch.float32, device=device)  # (4, 2)
        self._clip_lo, self._clip_hi = clip[:, 0], clip[:, 1]

    @property
    def _obs_dim(self) -> int:
        """3 + 3 + 3 root terms, then joint pos + vel over the observed joints, then last_action."""
        return 9 + 2 * len(self._obs_joint_ids) + len(self.joint_ids)

    @property
    def action_dim(self) -> int:
        return 4  # [vx, vy, wz, hip_height] — a command, not a joint vector

    # ----- the loop -----------------------------------------------------------------------------
    def _observe(self) -> torch.Tensor:
        """The 79-vector, rebuilt from `articulation.data` every tick (see the module docstring for
        the term order — it is a contract with the checkpoint, not a free choice)."""
        import torch

        d = self.robot.articulation.data
        jids = self._obs_joint_ids
        return torch.cat(
            (
                d.root_lin_vel_b,
                d.root_ang_vel_b,
                d.projected_gravity_b,
                d.joint_pos[:, jids] - d.default_joint_pos[:, jids],
                (d.joint_vel[:, jids] - d.default_joint_vel[:, jids]) * 0.1,
                self._last_action,
            ),
            dim=-1,
        )

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """`(num_envs, 4)` base command -> `(num_envs, 12)` leg position targets. Also advances the
        `last_action` feedback, which is why this is not a pure function of `action` alone — the
        policy's own previous output is part of its next input."""
        import torch

        cmd = torch.clamp(action, min=self._clip_lo, max=self._clip_hi)
        with torch.no_grad():
            raw = self._policy(torch.cat((cmd, self._observe()), dim=-1))
        self._last_action = raw  # RAW, pre-scale/offset — what the checkpoint's obs expects
        return raw * self.cfg.output_scale + self._offset

    # ----- state --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the action history so the first tick after a reset sees a standing robot with no
        stale gait phase behind it."""
        if self._last_action is None:
            return
        if env_ids is None:
            self._last_action.zero_()
        else:
            self._last_action[env_ids] = 0.0

    def get_state(self, env_ids: torch.Tensor | None = None) -> dict[str, Any]:
        if self._last_action is None:
            return {}
        sel = self._last_action if env_ids is None else self._last_action[env_ids]
        return {"last_action": sel.clone()}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor | None = None) -> None:
        if not state or self._last_action is None:
            return
        if env_ids is None:
            self._last_action.copy_(state["last_action"])
        else:
            self._last_action[env_ids] = state["last_action"]
