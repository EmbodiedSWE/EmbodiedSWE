"""Assemble the per-run /task folder: instructions + task + rules + hints.

Called by the run launcher, NOT the builder: one built world can host many
prompt conditions. Selection is by presence — the folder contains exactly the
files this run's condition grants. Hints and rules both live as single-file
libraries under eval/prompts/; the harness validates a selection against the
world's receipt so the prompt can never LIE about the world (withholding a
disclosure is allowed — that's an experimental variant; contradicting is not).

    /task/
    ├── instructions.md   the contract (explains this folder; the CLI prompt)
    ├── task.md           describe() harvest + goal extension + build snippet
    ├── rules/<name>.md   the selected rule disclosures
    └── hints/<name>.md   the selected hints
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
HINTS_DIR = PROMPTS_DIR / "hints"
RULES_DIR = PROMPTS_DIR / "rules"
TASKS_DIR = PROMPTS_DIR / "tasks"

# selection may withhold, never lie: each entry names the receipt fact a
# hint/rule presumes, and the value that fact must have for it to be true
HINTS_NEED = {"save_snapshot": ("set_states", True)}
RULES_NEED = {
    "no_set_states": ("set_states", False),
    "frozen_controller": ("control_mode_frozen", True),
}

_TRANSFER_NOTE = (
    "`/workspace` contains your files from the previous stage of this "
    "experiment. Reuse whatever helps — start with `HANDOFF.md` if present."
)


def list_hints() -> list[str]:
    return sorted(p.stem for p in HINTS_DIR.glob("*.md"))


def list_rules() -> list[str]:
    return sorted(p.stem for p in RULES_DIR.glob("*.md"))


def check_condition(hints: list[str], rules: list[str], facts: dict) -> None:
    """Fail fast on unknown names and prompt-vs-world contradictions.

    `facts` is the world receipt's fact set (e.g. {"set_states": True,
    "control_mode_frozen": True}); a selected hint/rule whose presumed fact
    doesn't hold refuses to render."""
    for names, library, needs, kind in (
        (hints, list_hints(), HINTS_NEED, "hint"),
        (rules, list_rules(), RULES_NEED, "rule"),
    ):
        for n in names:
            if n not in library:
                raise SystemExit(
                    f"unknown {kind} '{n}' — library: {', '.join(library) or '(empty)'}")
            if n in needs:
                fact, wanted = needs[n]
                if facts.get(fact) != wanted:
                    raise SystemExit(
                        f"{kind} '{n}' presumes {fact}={wanted}, but this world has "
                        f"{fact}={facts.get(fact)} — the prompt may not contradict the world")


def render_task_dir(
    task_dir: Path,
    *,
    scene: str,
    preset: str,
    describe_text: str,
    facts: dict,
    hints: list[str] | tuple[str, ...] = (),
    rules: list[str] | tuple[str, ...] = (),
    carryover: bool = False,
) -> Path:
    """Write the complete /task folder for one run.

    Everything except task.md (the auto-generated scene+robot description)
    is hand-authored library content, included only when selected — an
    unselected rule means the restriction goes undisclosed."""
    hints = list(hints)
    rules = list(rules)
    check_condition(hints, rules, facts)
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

    for sub, names, src in (("rules", rules, RULES_DIR), ("hints", hints, HINTS_DIR)):
        if names:
            d = task_dir / sub
            d.mkdir(exist_ok=True)
            for n in names:
                shutil.copyfile(src / f"{n}.md", d / f"{n}.md")

    # instructions.md — the contract verbatim (it explains the /task folder
    # semantics generically), plus the transfer note when a workspace carries over
    contract = (PROMPTS_DIR / "_contract.md").read_text().strip() + "\n"
    if carryover:
        contract += "\n" + _TRANSFER_NOTE + "\n"
    (task_dir / "instructions.md").write_text(contract)
    return task_dir
