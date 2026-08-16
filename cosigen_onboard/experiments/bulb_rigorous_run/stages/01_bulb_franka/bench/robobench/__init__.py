"""robobench — a long-horizon robot benchmark for code-generating agents (Isaac Lab / PhysX).

A collection of robot task-family **suites** that probe whether an agent can few-shot solve
multi-step problems. Each suite (the first is `robobench.suites.assembly` — everyday assembly
tasks on real threaded contact) supplies its own scenes / assets and composes the shared `robobench.core` machinery
+ `robobench.robots` embodiments into an open, GPU-batched env. There is no task layer — each scene
*is* one task (goal + optional grader). See ../CLAUDE.md.

Import order: create `AppLauncher` FIRST; only then import anything that touches `isaaclab.sim`.
`robobench.core.registries` and the base interfaces are import-light (safe without the app).
"""

__version__ = "0.1.0"


def discover() -> None:
    """Populate the core registries by importing everything that registers into them: every robot,
    every controller, and every suite (each suite's `__init__` registers its scenes + env configs).
    Call once before resolving anything by name — `ENVS.get(...)`, `EnvCfg.build()`, a smoke/harness —
    so `SCENES` / `ROBOTS` / `CONTROLLERS` / `ENVS` are complete. App-free (registration is
    import-light) and idempotent (imports are cached). A suite that fails to import is warned and
    skipped, so one broken suite doesn't hide the rest.
    """
    import importlib
    import pkgutil

    importlib.import_module("robobench.robots")
    importlib.import_module("robobench.controllers")
    suites = importlib.import_module("robobench.suites")
    for m in pkgutil.iter_modules(suites.__path__):
        if not m.ispkg:
            continue
        try:
            importlib.import_module(f"robobench.suites.{m.name}")
        except Exception as e:  # noqa: BLE001 — a broken suite shouldn't hide the others
            print(f"[robobench.discover] skipped suite '{m.name}': {e}")
