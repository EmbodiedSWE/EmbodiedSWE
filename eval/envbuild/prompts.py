"""Assemble the per-run /task folder: instructions + task + rules + hints.

Called by the run launcher, NOT the builder: one built world can host many
prompt conditions. Selection is by presence — the folder contains exactly the
files this run's condition grants. Hints and rules both live as single-file
libraries under eval/prompts/; the harness validates a selection against the
world's receipt so the prompt can never LIE about the world (withholding a
disclosure is allowed — that's an experimental arm; contradicting is not).

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

# hints that teach an ability a world-level arm can remove — refused when the
# receipt says the ability is absent
HINTS_NEED_SET_STATES = {"save_snapshot"}

# rules that state a world fact — refused when the receipt says otherwise
RULES_NEED_NO_SET_STATES = {"no_set_states"}

_TRANSFER_NOTE = (
    "`/workspace` contains your files from the previous stage of this "
    "experiment. Reuse whatever helps."
)


def list_hints() -> list[str]:
    return sorted(p.stem for p in HINTS_DIR.glob("*.md"))


def list_rules() -> list[str]:
    return sorted(p.stem for p in RULES_DIR.glob("*.md"))


def check_condition(hints: list[str], rules: list[str], set_states: bool) -> None:
    """Fail fast on unknown names and prompt-vs-world contradictions."""
    lib_h, lib_r = list_hints(), list_rules()
    for h in hints:
        if h not in lib_h:
            raise SystemExit(f"unknown hint '{h}' — library: {', '.join(lib_h) or '(empty)'}")
        if h in HINTS_NEED_SET_STATES and not set_states:
            raise SystemExit(
                f"hint '{h}' teaches set_states, but this world has it disabled "
                "— the prompt may not contradict the world")
    for r in rules:
        if r not in lib_r:
            raise SystemExit(f"unknown rule '{r}' — library: {', '.join(lib_r) or '(empty)'}")
        if r in RULES_NEED_NO_SET_STATES and set_states:
            raise SystemExit(
                f"rule '{r}' claims set_states is disabled, but this world has "
                "it enabled — the prompt may not contradict the world")


def render_task_dir(
    task_dir: Path,
    *,
    scene: str,
    preset: str,
    describe_text: str,
    set_states: bool,
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
    check_condition(hints, rules, set_states)
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
