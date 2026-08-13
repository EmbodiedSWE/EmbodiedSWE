"""simgen suite — adopts the generated sim_gen task scenes into the robobench registries.

Construction campaigns leave finished task packages under ``sim_gen/tasks*/<seed>_iNN/`` — one
corpus directory per campaign (``tasks/``, ``tasks_v4/``, ``tasks_v6/`` …; the orchestrator's
``SIM_GEN_TASKS_DIR``) — each registering its scene and a null-robot env on import. Nothing
imported them during ``robobench.discover()``, so they were invisible to the standard pipeline
(``build_env.py``, ``verify_solution.py --preset``). This shim is the missing suite package:
discover() imports every package under ``robobench/suites/``, so it needs no core changes.
EVERY corpus is loaded, newest last, so a task name appearing in two campaigns resolves to the
newest package.

For every task scene that imports cleanly it also registers the FRANKA binding
(``simgen.<scene>.franka.osc``) used by the agent evals: the scenes ship with ``robot="null"``
only (their own smokes teleport), and the eval requirement is a real arm.
"""
from __future__ import annotations

import importlib
from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parents[3] / "sim_gen"


def _corpora() -> list[str]:
    """Corpus package names, oldest first: `tasks`, then `tasks_<campaign>` sorted."""
    named = sorted(p.name for p in SIM_GEN_ROOT.glob("tasks_*") if p.is_dir())
    return (["tasks"] if (SIM_GEN_ROOT / "tasks").is_dir() else []) + named


def _load() -> None:
    from robobench.core import EnvCfg, register_env
    from robobench.core.registries import ENVS, SCENES

    corpora = _corpora()
    if not corpora:
        print(f"[simgen suite] no task corpus under {SIM_GEN_ROOT}; suite empty")
        return

    before = set(SCENES.list())
    for corpus in corpora:
        try:
            pkg = importlib.import_module(f"sim_gen.{corpus}")
        except ImportError as exc:
            print(f"[simgen suite] sim_gen.{corpus} not importable ({exc}); skipped")
            continue
        # Any folder holding a scene.py is a task, whether or not it carries an __init__.py:
        # a generated package without one still imports as a namespace package, and requiring
        # the file silently hid 10 of the 12 tasks in `tasks/` from the whole pipeline (their
        # presets simply "did not exist"). Discovery keys on the artifact that defines a task.
        for task in sorted({p.name for root in pkg.__path__
                            for p in Path(root).iterdir() if (p / "scene.py").is_file()}):
            try:
                importlib.import_module(f"sim_gen.{corpus}.{task}.scene")
            except Exception as exc:  # noqa: BLE001 — one broken task must not hide the rest
                print(f"[simgen suite] skipped task '{corpus}/{task}': {exc}")

    for name in sorted(set(SCENES.list()) - before):
        if f"simgen.{name}.franka.osc" in ENVS.list():
            continue
        try:
            register_env("simgen", lambda n=name: EnvCfg(
                scene=n, robot="franka", control_mode="osc", env_spacing=4.0))
        except KeyError as exc:
            print(f"[simgen suite] franka binding for '{name}' not registered: {exc}")


_load()
