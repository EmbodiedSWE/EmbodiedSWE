"""assembly suite — everyday assembly tasks built on real threaded/fastened contact.

The first robobench suite: screwing a light bulb into its socket, threading a nut
onto a bolt, driving an allen bolt, assembling an IKEA-style table (four threaded
legs), seating a GPU / populating a PC motherboard, and fastening an SO101 robot
arm together. Each scene under `scenes/` *is* a task — it carries its own goal
(`describe()`); `assets/` holds the parts, `configs/` the registered presets.
Reuses `robobench.core` + `robobench.robots`.
"""

from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
