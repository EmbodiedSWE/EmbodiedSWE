"""Install the python tools a condition grants — the tools/ counterpart of prompts.py.

Granting a tool means three things happen, all by iterating the condition's list:

  1. its module is COPIED into the run's `tools/` directory, which the runner mounts and puts on
     the agent's PYTHONPATH, so `from checkpoint_tree import CheckpointTree` works;
  2. its prompt section is appended to the task text, so the agent knows the tool exists;
  3. its companion skill is installed, so the agent knows when to reach for it.

Withholding a tool is the absence of all three — and because a tool is a FILE, a withheld tool is
not importable at all. There is no namespace filtering, no method-level gate and no prompt
stripping anywhere in the harness: the ablation is that the module is not there.

No per-tool logic lives here. Every branch point is data on the ToolSpec in the tool's own file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .condition import EVAL_DIR, PROMPTS_DIR, SKILLS_DIR, Condition


def prompt_sections(cond: Condition) -> str:
    """The granted tools' doc sections, concatenated in the order the config lists them.

    SELECTED, never branched (design note §1): each section is a file, so the task text grows a
    section when a tool is granted instead of code removing one when it is not.
    """
    parts: list[str] = []
    for spec in cond.tool_specs():
        if not spec.prompt_doc:
            continue
        path = PROMPTS_DIR / spec.prompt_doc
        if not path.exists():
            raise SystemExit(f"tool '{spec.name}' names a missing prompt_doc: {path}")
        parts.append(path.read_text().strip())
    if not parts:
        return ""
    # Where the modules are is a property of THIS run's install, not of any tool, so it is stated
    # once here rather than repeated in every tool's doc.
    preamble = (
        "== Tools available to you ==\n"
        "The tools below are python modules already installed at `/task/tools`, which is on your "
        "PYTHONPATH — import them by name from your own scripts (no install step, no setup)."
    )
    return "\n\n".join([preamble, *parts]) + "\n"


def install(cond: Condition, dest: Path) -> list[str]:
    """Install the granted tools under `dest` (a run's task dir). Returns what was installed.

    Layout:
        <dest>/tools/<module>.py    importable by the agent (runner puts this dir on PYTHONPATH)
        <dest>/skills/<pkg>/        the companion skill packages
        <dest>/tools.md             the concatenated prompt sections
    """
    installed: list[str] = []
    specs = cond.tool_specs()
    if not specs:
        return installed

    tools_dest = dest / "tools"
    skills_dest = dest / "skills"
    for spec in specs:
        tools_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.source(), tools_dest / f"{spec.module_name()}.py")
        installed.append(f"tools/{spec.module_name()}.py")
        for rel in spec.payload:
            src = EVAL_DIR / "tools" / rel
            if not src.exists():
                raise SystemExit(f"tool '{spec.name}' names missing payload: {src}")
            target = tools_dest / Path(rel).name
            shutil.copy2(src, target) if src.is_file() else shutil.copytree(src, target, dirs_exist_ok=True)
            installed.append(f"tools/{Path(rel).name}")
        if spec.skill:
            src = SKILLS_DIR / spec.skill
            if not src.is_dir():
                raise SystemExit(f"tool '{spec.name}' names a missing skill package: {src}")
            skills_dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, skills_dest / spec.skill, dirs_exist_ok=True)
            installed.append(f"skills/{spec.skill}")

    sections = prompt_sections(cond)
    if sections:
        (dest / "tools.md").write_text(sections)
        installed.append("tools.md")
    return installed
