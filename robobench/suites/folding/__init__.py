"""folding suite — deformable-object folding tasks on IsaacLab's **Newton** physics backend.

The second robobench suite, and the first that runs on the Newton backend (IsaacLab develop) —
the only backend with cloth (surface deformables): a T-shirt lying on a box table, to be folded
flat by a Franka. Holds its `scenes/` (a scene *is* a task — it carries its own goal), the shirt
`assets/`, concrete `configs/`, and `scripts/` (the fold smoke); it reuses
`robobench.core` + `robobench.robots`.

Runs ONLY under the Newton venv (`env_newton`, set up per the README's Newton-env steps) — the
assembly suite's isaaclab 2.3.2 venv predates the Newton backend. Importing this
package (registration) stays app-free and works in either venv.
"""

from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
