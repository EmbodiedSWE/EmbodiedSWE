"""PinkIKController — multi-task IK over any articulation (wraps Isaac's PinkIKController).

Embodiment-agnostic: a per-step **QP** over a Pinocchio model driving any number of weighted
`LocalFrameTask`s (one per end-effector pose) + an optional `NullSpacePostureTask` (redundancy toward
home), with hard joint limits — over whatever `frames` / `joint_names` the cfg names (a humanoid's two
wrists, a single arm, a mobile manipulator, ...).

Action = the target pose of each frame, `[pos(3), quat(4, wxyz)]`, in the *env* frame. `compute`
transforms them into the base-link frame, solves one QP per env, and returns joint **position**
targets for the controlled chain (joints outside it, e.g. hands, are a separate controller).

Wraps `isaaclab.controllers.pink_ik.PinkIKController`, one per env. isaaclab/pink imports are deferred
into `bind`/`compute` (they pull `pxr`), so importing this module stays app-free.

PARALLELISM — per-env CPU loop, NOT batched: `compute` batches only the pose prep; the solve is a
`for env: ...` loop (one Pinocchio model + daqp QP each), so cost is ~linear in `num_envs`. A batched
controller (`diff_ik` single-chain GPU, or `pyroki` JAX multi-task) is the future swap for throughput
via this same slot — pink trades batching for the multi-task QP + hard limits.

CRITICAL — **`import pinocchio` BEFORE `AppLauncher`**: otherwise in-app pinocchio calls fail with
"No Python class registered for C++ class std::vector<std::string>" (its eigenpy STL converters must
register in a clean process first). Any script building a `pink_ik` env imports pinocchio at the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robobench.core import CONTROLLERS, BaseController, BaseControllerCfg, info, tunable

if TYPE_CHECKING:
    import torch


@dataclass
class FrameTaskCfg:
    """One end-effector frame the IK tracks. `link` is the **URDF** frame name; the costs weight
    position vs orientation in the QP."""

    link: str
    position_cost: float = 8.0  # [cost]/[m]
    orientation_cost: float = 2.0  # [cost]/[rad]
    lm_damping: float = 10.0  # Levenberg-Marquardt damping (smooths step jumps)
    gain: float = 0.5  # fraction of error closed per step


@dataclass
class PinkIKControllerCfg(BaseControllerCfg):
    """Config for the Pink IK wrapper: the `frames` to track + the `joint_names` chain it drives.
    Frame/base names are **URDF** names; joint names are **USD/Isaac**. `info` = robot-supplied
    structure; `tunable` = solver knobs.

    PLACEMENT: the reusable schema lives here (one Pink IK serves any robot); a *robot* fills the
    structural values (URDF / frames / chain / base) — only it knows its kinematics (see
    `G1Robot.build_controller`)."""

    urdf_path: str = info()  # kinematics URDF (Pinocchio model); meshes not needed for IK
    base_link: str = info()  # USD body name of the base
    base_link_frame: str = info()  # URDF frame name of the base
    frames: tuple[FrameTaskCfg, ...] = info()  # the end-effector frames (one per target pose)
    joint_names: tuple[str, ...] = info()  # the IK-driven joint chain (USD names / regex)
    nullspace_joints: tuple[str, ...] = info(())  # joints to regularize toward home ('' -> no null-space)
    nullspace_cost: float = tunable(0.5)
    nullspace_gain: float = tunable(0.3)
    nullspace_lm_damping: float = tunable(1.0)
    mesh_path: str | None = info(None)
    show_ik_warnings: bool = tunable(False)


@CONTROLLERS.register("pink_ik")
class PinkIKController(BaseController):
    """Multi-task Pink IK over the cfg's joint chain + frame set (any robot). Fixed
    `command_type="position"` (IK emits joint position targets). `action_dim` = 7 per frame. One Isaac
    `PinkIKController` per env."""

    def __init__(self, cfg: PinkIKControllerCfg) -> None:
        super().__init__(cfg, command_type="position")  # IK always outputs position targets

    def _resolve_joints(self, robot: Any) -> Any:
        return robot.articulation.find_joints(list(self.cfg.joint_names))[0]

    @property
    def action_dim(self) -> int:
        return len(self.cfg.frames) * 7  # [pos3, quat4] per frame; hands are a separate controller

    def bind(self, robot: Any) -> None:
        """Resolve joints + sink (base template), then build one Isaac PinkIKController per env."""
        super().bind(robot)  # -> self.joint_ids (the chain), self._sink (position), self.limits

        from isaaclab.controllers.pink_ik import PinkIKController as _IsaacPinkIK
        from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
        from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
        from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg as _IsaacCfg

        c = self.cfg
        art = robot.articulation
        all_names = list(art.data.joint_names)  # USD joint order
        controlled_names = [all_names[i] for i in self.joint_ids]  # same order as joint_ids

        def make_cfg():  # fresh task objects per env (targets are set per-env on these)
            tasks: list[Any] = [
                LocalFrameTask(
                    f.link,
                    base_link_frame_name=c.base_link_frame,
                    position_cost=f.position_cost,
                    orientation_cost=f.orientation_cost,
                    lm_damping=f.lm_damping,
                    gain=f.gain,
                )
                for f in c.frames
            ]
            if c.nullspace_joints:
                tasks.append(
                    NullSpacePostureTask(
                        cost=c.nullspace_cost,
                        lm_damping=c.nullspace_lm_damping,
                        controlled_frames=[f.link for f in c.frames],
                        controlled_joints=list(c.nullspace_joints),
                        gain=c.nullspace_gain,
                    )
                )
            return _IsaacCfg(
                articulation_name="robot",
                base_link_name=c.base_link,
                num_hand_joints=0,  # hands are a separate composite leaf, not part of this action
                urdf_path=c.urdf_path,
                mesh_path=c.mesh_path,
                show_ik_warnings=c.show_ik_warnings,
                fail_on_joint_limit_violation=False,
                variable_input_tasks=tasks,
                fixed_input_tasks=[],
                joint_names=controlled_names,
                all_joint_names=all_names,
            )

        device = robot.env.device
        self._controllers = [_IsaacPinkIK(make_cfg(), art.cfg, device, self.joint_ids) for _ in range(robot.env.num_envs)]
        self._base_idx = art.data.body_names.index(c.base_link)
        # IK integration step = the CONTROL period, not the sim dt: the solver fires once per
        # control_period physics steps (its target is held in between), so it integrates over that span.
        self._dt = robot.env.dt * self._control_period
        self._n_frames = len(c.frames)

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """`action` (n, 7*frames): per-frame `[pos3, quat4]` in the env frame. Transform each to the
        base-link frame, set the per-env frame-task targets, solve one QP per env, return (n, |chain|)
        joint position targets."""
        import torch
        from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
        from isaaclab.utils import math as mu

        art = self.robot.articulation
        n = action.shape[0]

        # Base-link pose in the env frame, then its inverse (world->base).
        base_w = art.data.body_link_state_w[:, self._base_idx, :7]  # (n, 7) pos+quat in world
        base_pos = base_w[:, :3] - self.robot.env.iscene.env_origins
        base_inv = mu.pose_inv(mu.make_pose(base_pos, mu.matrix_from_quat(base_w[:, 3:7])))  # (n, 4, 4)

        # Each frame's commanded pose (env frame) -> base-link frame (batched over envs).
        pos_b, rot_b = [], []
        for fi in range(self._n_frames):
            s = fi * 7
            pose_w = mu.make_pose(action[:, s : s + 3], mu.matrix_from_quat(action[:, s + 3 : s + 7]))
            p, R = mu.unmake_pose(mu.pose_in_A_to_pose_in_B(pose_w, base_inv))
            pos_b.append(p)
            rot_b.append(R)

        # Per env: set the frame-task targets, then solve. THIS LOOP IS THE SERIAL PART (one QP per
        # env, ~linear in num_envs — see the module docstring's PARALLELISM note); the prep above is
        # batched. A batched controller (diff_ik / pyroki) is the future swap for throughput.
        cur_all = art.data.joint_pos.cpu().numpy()  # (n, num_joints)
        out = []
        for env_i in range(n):
            ctrl = self._controllers[env_i]
            fi = 0
            for task in ctrl.cfg.variable_input_tasks:
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                    target.translation = pos_b[fi][env_i].cpu().numpy()
                    target.rotation = rot_b[fi][env_i].cpu().numpy()
                    task.set_target(target)
                    fi += 1
            out.append(ctrl.compute(cur_all[env_i], self._dt))  # (|chain|,) torch on device
        return torch.stack(out)
