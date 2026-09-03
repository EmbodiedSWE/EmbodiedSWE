"""deformable suite — deformable-object manipulation on IsaacLab's **Newton** physics backend.

Four scenes, one per material class; each scene *is* a task (it carries its own goal):

  - `tshirt`   VBD cloth — a T-shirt lying on a box table, to be folded flat by a Franka.
  - `latte`    implicit-MPM liquids on the coupled MJWarp+MPM substrate — two Frankas, one
               carrying the mug, the other pouring milk from a pitcher into the coffee.
  - `dumpling` elastoplastic implicit-MPM dough — roll the dough ball out into a wrapper with
               the rolling pin.
  - `knot`     Newton rods (capsule chains on cable joints, VBD/AVBD) — tie a self-holding half
               knot from a sneaker's two shoelaces.

These four used to be the `folding`, `pouring`, `dough` and `shoe_tying` suites. They are one
suite now (the paper's catalog groups them as "Deformable"); the old preset names are kept as
aliases in `configs/envs.py`, so `folding.tshirt.franka.joint`, `pouring.latte.bimanual_franka.joint`,
`dough.dumpling` and `shoe_tying.knot` still resolve.

Layout:
  newton/    the per-material Newton substrates — `cloth_sim.py` (NewtonSimCfg), `mpm_sim.py`
             (MpmSimCfg), `dough_sim.py` (DoughSimCfg), `rod_sim.py` (RodSimCfg) — plus the
             coupled MJWarp+MPM manager shared by latte and dumpling (`coupled_manager.py`) and
             the rod manager specialization (`lace_manager.py`) and the proxy-coupled
             MJWarp+VBD-rod manager for the knot scene's robot bindings (`lace_coupled_manager.py`).
  scenes/    the four scenes; `configs/` the registered env presets; `smokes/` the runnable
             capability checks; `assets/` shirt, mug, pitcher and shoe; `docs/` the dumpling and
             shoe-knot physics notes.

Runs ONLY under the Newton venv (`env_newton`, newton >= 1.6 — see the README's Newton-env
steps); the PhysX suites' isaaclab 2.3.2 venv predates the Newton backend. Importing this package
(registration) stays app-free and works in either venv. Single-env only for the MPM scenes: the
solver uses one fixed grid spanning the scene, so keep `num_envs=1` when building.
"""

from . import configs  # noqa: F401  (registers the suite's named env configs into robobench.core.ENVS)
from . import scenes  # noqa: F401  (registers the suite's scenes into robobench.core.SCENES)
