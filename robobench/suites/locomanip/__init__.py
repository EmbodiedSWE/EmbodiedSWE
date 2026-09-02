"""locomanip suite — tasks a humanoid can only do by WALKING to them.

Every other robobench suite bolts its robot down: an arm on a plinth, or a humanoid with its pelvis
welded to the world. That is a deliberate simplification, and it caps what a task can ask for — an
object out of arm's reach is simply unsolvable. This suite drops the weld. Its scenes are laid out
over metres rather than centimetres, so the robot's own base motion is part of the solution and the
locomotion controller is as load-bearing as the arm controller.

Two scenes today. `wheel_carry`: pick a steering wheel off one packing table, carry it about three
metres to another, and drop it in the basket standing there — deliberately a close descendant of
`assembly.wheel_pick_place` (same assets, same measured reach geometry at both ends, one basket
moved out of range), so what the walk adds is isolated from everything else. `box_to_bin`: pick a
cardboard box off a shelf board and leave it in a sorting bin on a low table around a ~100 deg turn
— a measured port of IsaacLab-Arena's `galileo_g1_locomanip_pick_and_place`, adding what the first
scene's straight corridor does not: a heading change, a shelf pick, and a bimanual-sized object.

Needs a robot whose `control_mode` drives its legs (today: the G1's `loco_*` modes, see
`robots/g1.py` and `controllers/loco_policy.py`). A fixed-base binding will build and then fail to
reach the second station, which is the point.
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
