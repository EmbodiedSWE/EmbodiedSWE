"""Assemble the per-run /task folder: instructions + task + rules + skills + tools.

Called by the run launcher, NOT the builder: one built world can host many prompt conditions.
Selection is by presence — the folder contains exactly the files this run's condition grants.

A condition's prompt is its listed files CONCATENATED, never branched (design note §1): where
text must differ between arms, two files exist and each config picks one. So this module is a
read-and-join loop with no conditionals about content, and `envbuild/condition.py` is the only
thing that reads a config.

    /task/
    ├── instructions.md    the contract (explains this folder; the CLI prompt)
    ├── task.md            describe() harvest + goal extension + build snippet
    ├── rules/<name>.md    the selected rules the agent MUST follow
    ├── skills/<name>.md   the selected skills the agent MAY follow
    ├── tools.md           the granted tools' doc sections (written by envbuild/tools.py)
    ├── tool_router.md     the granted tools' workflow + eligibility gates (in the prompt)
    └── skills/<pkg>/      each granted tool's companion skill package
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import tools as tool_install
from .condition import PROMPTS_DIR, RULES_DIR, SKILLS_DIR, Condition, check_facts, list_rules, list_skills

TASKS_DIR = PROMPTS_DIR / "tasks"

_TRANSFER_NOTE = (
    "`/workspace` contains your files from the previous stage of this "
    "experiment. Reuse whatever helps — start with `experiences/` if present."
)

_BUDGET_NOTE = (
    "This session has a hard wall-clock budget of {minutes:g} minutes — at the "
    "deadline the container is stopped, keeping only your files. Budget your "
    "time; submit progress before it runs out."
)


def render_task_dir(
    task_dir: Path,
    *,
    scene: str,
    preset: str,
    describe_text: str,
    facts: dict,
    condition: Condition,
    carryover: bool = False,
    budget_min: float | None = None,
) -> Path:
    """Write the complete /task folder for one run under `condition`.

    Everything except task.md (the auto-generated scene+robot description) is hand-authored
    library content, included only when the condition selects it — an unselected rule means the
    restriction goes undisclosed.
    """
    check_facts(condition, facts)  # fail before touching disk
    task_dir.mkdir(parents=True, exist_ok=True)

    # task.md — auto-generated world + goal, optionally extended per scene
    parts = ["# The task", "", describe_text.strip(), ""]
    extra = TASKS_DIR / f"{scene}.md"
    if extra.exists():
        parts += [extra.read_text().strip(), ""]
    parts += [
        "## Building the environment",
        "",
        "```python",
        "import robobench; robobench.discover()",
        "from robobench.core.registries import ENVS",
        f"env = ENVS.get('{preset}')().build(num_envs=1)",
        "```",
        "",
    ]
    (task_dir / "task.md").write_text("\n".join(parts))

    for sub, names, src in (
        ("rules", condition.rules, RULES_DIR),
        ("skills", condition.skills, SKILLS_DIR),
    ):
        if names:
            d = task_dir / sub
            d.mkdir(exist_ok=True)
            for n in names:
                shutil.copyfile(src / f"{n}.md", d / f"{n}.md")

    # Granted tools: their doc sections (tools.md) and companion skill packages.
    tool_install.install(condition, task_dir)

    # The condition, machine-readable, next to the text it produced. A driver mounts /task
    # anyway, so it can read which tools to register and which names to keep in the program
    # namespace from here — no env var, and no way for the prompt and the toolset to disagree.
    (task_dir / "condition.json").write_text(json.dumps(condition.as_record(), indent=2) + "\n")

    # tool_router.md — the concise workflow + eligibility gates for the granted tools. It
    # replaces the full tools.md in the initial prompt (the agent opens tools.md on demand when
    # the router sends it to a tool), and agent-entry.sh also rides it on the Claude SYSTEM
    # prompt every leg so compaction cannot drop it. Reconstructed 2026-09-06 from the rendered
    # /task of the 2026-09-05 campaign pod (the renderer that wrote it was lost with the laptop);
    # the body is the same for every grant set that campaign used (all five tools).
    router = ""
    if condition.tools:
        granted = ", ".join(f"`{t}`" for t in condition.tools)
        router = (PROMPTS_DIR / "tool_router.md").read_text().format(granted=granted).strip() + "\n"
        (task_dir / "tool_router.md").write_text(router)

    # instructions.md — the contract verbatim (it explains the /task folder semantics
    # generically), plus the notes that depend on how this run is driven rather than on the
    # condition. The router is appended IN the prompt: the CLI's first message is this file,
    # and a tool the agent has to discover by listing /task is a tool half-granted
    # (2026-07-31: agents found tools.md only by exploring).
    contract = (PROMPTS_DIR / "_contract.md").read_text().strip() + "\n"
    if router:
        contract += "\n" + router
    if budget_min:
        contract += "\n" + _BUDGET_NOTE.format(minutes=budget_min) + "\n"
    if carryover:
        contract += "\n" + _TRANSFER_NOTE + "\n"
    (task_dir / "instructions.md").write_text(contract)
    return task_dir
