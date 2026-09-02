"""Shared mid-level controllers, reusable across embodiments (the robot's "System 1").

Importing this package registers every controller into `robobench.core.CONTROLLERS`. Each is a
`BaseController` that maps an agent action → joint position targets for the DOFs it owns; a robot
selects one per `control_mode` and wires it with its own joints / ee-frame / gains. Import-light
where possible (heavy solvers deferred), so registration is app-free.

Roster (built incrementally — see CLAUDE.md):
  joint · diff_ik · osc · pink_ik            # base controllers
  composite (DOF-group split) · policy (frozen checkpoint) · residual (learned correction)  # wrappers

Built so far: `joint` (pass-through), `composite` (DOF-group split), the task-space torque pair
(`osc` / `task_impedance`), both IK solvers — `pink_ik` (per-env multi-task QP, humanoids) and
`diff_ik` (batched single-chain DLS, arms) — and `loco_policy` (a frozen locomotion checkpoint
driven by a base-velocity command; the legs leaf of loco-manipulation).
"""

from .composite import CompositeController
from .diff_ik import DiffIKController, DiffIKControllerCfg
from .joint import JointController, JointControllerCfg
from .loco_policy import LocoPolicyController, LocoPolicyControllerCfg
from .pink_ik import FrameTaskCfg, PinkIKController, PinkIKControllerCfg
from .task_space import (
    OperationalSpaceController,
    TaskSpaceControllerCfg,
    TaskSpaceImpedanceController,
)

__all__ = [
    "JointController",
    "JointControllerCfg",
    "CompositeController",
    "DiffIKController",
    "DiffIKControllerCfg",
    "TaskSpaceControllerCfg",
    "TaskSpaceImpedanceController",
    "OperationalSpaceController",
    "PinkIKController",
    "PinkIKControllerCfg",
    "FrameTaskCfg",
    "LocoPolicyController",
    "LocoPolicyControllerCfg",
]
