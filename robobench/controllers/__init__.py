"""Shared mid-level controllers, reusable across embodiments (the robot's "System 1").

Importing this package registers every controller into `robobench.core.CONTROLLERS`. Each is a
`BaseController` that maps an agent action → joint position targets for the DOFs it owns; a robot
selects one per `control_mode` and wires it with its own joints / ee-frame / gains. Import-light
where possible (heavy solvers deferred), so registration is app-free.

Roster (built incrementally — see CLAUDE.md):
  joint · diff_ik · osc · pink_ik            # base controllers (IK solver next)
  composite (DOF-group split) · policy (frozen checkpoint) · residual (learned correction)  # wrappers

Built so far: `joint` (pass-through) and `composite` (DOF-group split). The IK solvers (`pink_ik`
for the G1 upper body, `diff_ik` for arms) land next, alongside their first robot.
"""

from .composite import CompositeController
from .joint import JointController, JointControllerCfg
from .pink_ik import FrameTaskCfg, PinkIKController, PinkIKControllerCfg

__all__ = [
    "JointController",
    "JointControllerCfg",
    "CompositeController",
    "PinkIKController",
    "PinkIKControllerCfg",
    "FrameTaskCfg",
]
