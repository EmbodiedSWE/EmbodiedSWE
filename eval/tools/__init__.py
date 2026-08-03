"""The tool library: pre-created python tools the agent can import and call.

One file per tool. Each file contains BOTH its declaration (a module-level `TOOL = ToolSpec(...)`)
and its implementation, so granting a tool is adding a name to a condition's `tools:` list and
nothing else — there is no per-tool code in any driver, launcher, or harness module.

## The execution model these tools are built for

The eval harness gives the agent a container with `/bench` (robobench, read-only), a writable
`/workspace`, and Isaac in `/opt/venv`. The agent writes its own python scripts and runs them.
Each script is a FRESH PROCESS that boots its own simulator and exits.

Two consequences shape every tool here:

  1. A tool is a plain python module the agent imports (`from checkpoint_tree import
     CheckpointTree`) — not a hosted service, not an agent-framework tool class. Selected tools
     are installed into the world as an importable directory, so `import <module>` just works.
  2. Nothing survives in memory between the agent's scripts. A tool that keeps state must put it
     on disk under `/workspace`. This is why the checkpoint tree here persists to disk while the
     retired server-side one could hold nodes in RAM.

## Keeping a tool file importable in BOTH places

A tool file is imported twice, in two different shapes:

  * on the LAUNCHER as `tools.checkpoint_tree`, so `discover()` can read its declaration;
  * in the CONTAINER as bare `checkpoint_tree`, because it is installed as a standalone file on
    the agent's PYTHONPATH with no parent package.

So a tool file must import nothing from this package (`from . import ToolSpec` raises
"attempted relative import with no known parent package" in the second shape), and nothing
outside the standard library at module level — the launcher has no Isaac, torch or numpy. Hence:
declare `TOOL` as a plain dict, which `discover()` turns into a ToolSpec, and import
`torch`/`numpy`/`isaaclab` inside the functions that need them.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ToolSpec:
    """One grantable tool, declared by the file that implements it."""

    name: str
    # One line for a human reading a condition file or a receipt.
    description: str
    # What the agent imports. Defaults to the tool's own module name; every selected tool is
    # installed under one importable directory, so this is the `import <module>` name.
    module: str = ""
    # The symbols the companion skill teaches, e.g. ("CheckpointTree",). Documentation only —
    # nothing enforces it — but it keeps a skill and its tool from drifting apart.
    exports: tuple[str, ...] = ()
    # Companion guidance: a skills/ entry teaching WHEN to reach for it. A tool without its skill
    # still installs; the agent is simply not taught to use it.
    skill: str | None = None
    # Markdown section appended to the agent's task text, relative to eval/prompts/. SELECTED,
    # never generated: a condition's prompt is its listed files concatenated.
    prompt_doc: str | None = None
    # Features this tool needs left ON (design note §5): a granted tool keeps the features it
    # runs on, so the combination is refused at load rather than failing mid-run.
    requires_features: tuple[str, ...] = ()
    # Extra files to install beside the module, relative to eval/tools/.
    payload: tuple[str, ...] = field(default_factory=tuple)

    def module_name(self) -> str:
        return self.module or self.name

    def source(self) -> Path:
        """The file to install for the agent."""
        return TOOLS_DIR / f"{self.module_name()}.py"


def discover() -> dict[str, ToolSpec]:
    """Every tool in this library, by name. A file that fails to import raises — a silently
    missing tool would mean a condition granting something that never arrives."""
    found: dict[str, ToolSpec] = {}
    for mod in pkgutil.iter_modules([str(TOOLS_DIR)]):
        if mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{mod.name}")
        declared = getattr(module, "TOOL", None)
        if declared is None:
            raise SystemExit(f"eval/tools/{mod.name}.py defines no TOOL declaration")
        if not isinstance(declared, dict):
            raise SystemExit(f"eval/tools/{mod.name}.py: TOOL must be a dict, got "
                             f"{type(declared).__name__}")
        try:
            spec = ToolSpec(**declared)
        except TypeError as exc:
            raise SystemExit(f"eval/tools/{mod.name}.py: bad TOOL declaration — {exc}") from exc
        # Default the module name to the file it was declared in, so a tool never has to repeat
        # its own filename.
        if not spec.module:
            spec = ToolSpec(**{**declared, "module": mod.name})
        if spec.name in found:
            raise SystemExit(f"two tools claim the name '{spec.name}'")
        if not spec.source().is_file():
            raise SystemExit(f"tool '{spec.name}' has no module file at {spec.source()}")
        found[spec.name] = spec
    return found
