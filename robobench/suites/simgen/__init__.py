"""simgen suite — adopts the generated sim_gen task scenes into the robobench registries.

The sim_gen construction campaign left 47 finished task packages under ``sim_gen/tasks/<seed>_iNN/``,
each registering its scene and a null-robot env on import. Nothing imported them during
``robobench.discover()``, so they were invisible to the standard pipeline (``build_env.py``,
``verify_solution.py --preset``). This shim is the missing suite package: discover() imports every
package under ``robobench/suites/``, so it needs no core changes.

For every task scene that imports cleanly it also registers the FRANKA binding
(``simgen.<scene>.franka.osc``) used by the agent evals: the scenes ship with ``robot="null"``
only (their own smokes teleport), and the eval requirement is a real arm.
"""
from __future__ import annotations

import importlib
import pkgutil


def _load() -> None:
    try:
        import sim_gen.tasks as tasks_pkg
    except ImportError as exc:
        print(f"[simgen suite] sim_gen.tasks not importable ({exc}); suite empty")
        return
    from robobench.core import EnvCfg, register_env
    from robobench.core.registries import ENVS, SCENES

    before = set(SCENES.list())
    for m in pkgutil.iter_modules(tasks_pkg.__path__):
        if not m.ispkg:
            continue
        try:
            importlib.import_module(f"sim_gen.tasks.{m.name}.scene")
        except Exception as exc:  # noqa: BLE001 — one broken task must not hide the rest
            print(f"[simgen suite] skipped task '{m.name}': {exc}")

    for name in sorted(set(SCENES.list()) - before):
        if f"simgen.{name}.franka.osc" in ENVS.list():
            continue
        try:
            register_env("simgen", lambda n=name: EnvCfg(
                scene=n, robot="franka", control_mode="osc", env_spacing=4.0))
        except KeyError as exc:
            print(f"[simgen suite] franka binding for '{name}' not registered: {exc}")


_load()
