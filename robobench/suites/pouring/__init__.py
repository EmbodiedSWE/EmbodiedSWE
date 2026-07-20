"""pouring suite — liquid-manipulation tasks on IsaacLab's **Newton** physics backend (implicit MPM).

The third robobench suite, and the first that runs particle fluids: Newton's implicit MPM solver
advances the liquid while rigid geometry (table, cups) acts as colliders. First scene: `latte` —
a cup of coffee (brown particles) and a cup of milk (white particles); the goal is to pour the
milk into the coffee without spilling. Holds its `scenes/` (a scene *is* a task — it carries its
own goal), concrete `configs/`, and `scripts/` (the pour smoke); it reuses `robobench.core` +
`robobench.robots`.

Runs ONLY under the Newton venv (`env_newton`, set up per the README's Newton-env steps) — the
assembly suite's isaaclab 2.3.2 venv predates the Newton backend. Importing this package
(registration) stays app-free and works in either venv.
"""

from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
