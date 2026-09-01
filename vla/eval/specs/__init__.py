"""Named eval-sim setups — bake-free by construction (see ../DESIGN.md).

One module per scene family; importing this package registers everything.
`load_sim` imports it automatically, so a spec is usable by name the moment
its module is listed here. An entry = a robobench preset + the control law
spelled out; never a bake path.
"""

from . import bulb  # noqa: F401
from . import pc_ram  # noqa: F401
