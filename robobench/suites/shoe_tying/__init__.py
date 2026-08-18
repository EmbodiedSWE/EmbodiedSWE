"""shoe_tying suite — rod-based shoelace manipulation on IsaacLab's **Newton** physics backend.

The fourth robobench suite, and the first that runs Newton's *rods* (capsule-chain bodies joined
by cable joints, solved by VBD/AVBD): TIE a half knot from two initially separate shoelaces
rooted at a sneaker's top eyelets, by moving their free ends through the classic four beats
(cross -> under -> cross again -> pull apart + seat) — self-holding, slack and pin-free, seated
on the tongue. Ported from the proven standalone-newton `shoe_tying_knot.py` at
`~/research/newton/outputs/shoe_tying`.

Rods do not exist in IsaacLab's asset layer, so the scene injects them straight into the Newton
`ModelBuilder` through `NewtonManager._per_world_builder_hooks` (the in-tree MPM asset's
mechanism), while rendering goes through USD so Kit's **RTX** viewport draws it (textured visual
shoe + per-segment lace capsules synced from `body_q`). The suite carries its own sim substrate
(`newton_sim.py`) and manager specialization (`lace_manager.py`: per-substep kinematic handle
driving and the rod contact recipe).

Runs ONLY under the Newton venv (`env_newton`, newton >= 1.6 — see the README's Newton env
setup). Importing this package (registration) stays app-free and works in either venv.
"""

from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
