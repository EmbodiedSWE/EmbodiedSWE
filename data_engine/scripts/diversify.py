"""Launch a diversify authoring session in a data_gen campaign cell.

    python data_engine/scripts/diversify.py <gen_root> --level scene \\
        [--scene scene_0] [--strategy strategy_0] [--name scene_1] \\
        [--extra notes.md ...] [--agent claude] [--model claude-opus-5] [--dry]

Scaffolds the target cell, assembles the session instructions into the cell's
.agent/instructions.md (general contract + campaign facts + any --extra prompt
files), and hands the terminal to the agent CLI inside the cell:

    <agent> --model <model> "Read .agent/instructions.md and begin."

Any agent CLI with that calling shape plugs in via --agent; claude with Opus is
the testbed. --dry assembles everything and prints the instructions path
instead of launching. The session's required artifact is .agent/SUMMARY.md
(the contract says so); the instructions file doubles as the session record.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.cells import create_cell  # noqa: E402
def assemble(gen_root: Path, target: Path, level: str, base: str, name: str,
             extras: list[str]) -> Path:
    contract = (ROOT / "agent" / "prompts" / "_contract.md").read_text()
    template = (ROOT / "agent" / "prompts" / f"{level}.md").read_text()
    template = template.replace("{base}", base).replace("{name}", name)
    facts = [
        "\n\n" + template.strip(),
        "\n## This session\n",
        f"- campaign: {gen_root}",
        f"- your cell: {target}  (work here; your cwd)",
        "\n### gen.yaml\n```yaml",
        (gen_root / "gen.yaml").read_text().rstrip(),
        "```",
    ]
    metas = []
    for m in sorted(gen_root.glob("scenes/*/meta.json")) + sorted(gen_root.glob("scenes/*/strategies/*/meta.json")):
        rel = m.parent.relative_to(gen_root)
        metas.append(f"- {rel}: {m.read_text().strip()}")
    if metas:
        facts += ["\n### current yields (baselines)\n"] + metas
    for x in extras:
        p = Path(x)
        facts += [f"\n\n## Additional instructions — {p.name}\n", p.read_text()]
    out = target / ".agent" / "instructions.md"
    out.write_text(contract + "\n".join(facts) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="launch a diversify authoring session")
    ap.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
    ap.add_argument("--level", required=True, choices=("scene", "strategy", "phase"))
    ap.add_argument("--scene", default="scene_0", help="context scene (and the copy source for --level scene)")
    ap.add_argument("--strategy", default="strategy_0", help="context strategy (phase level) / reference (strategy level)")
    ap.add_argument("--name", default=None, help="new folder name (default: next free index)")
    ap.add_argument("--extra", action="append", default=[], help="extra prompt file(s) appended to the instructions")
    ap.add_argument("--agent", default="claude", help="agent CLI to launch")
    ap.add_argument("--model", default="claude-opus-5", help="model handed to the agent CLI")
    ap.add_argument("--dry", action="store_true", help="assemble only; print the instructions path")
    args = ap.parse_args()

    gen_root = Path(args.gen_root).resolve()
    if not (gen_root / "gen.yaml").is_file():
        raise SystemExit(f"not a campaign (no gen.yaml): {gen_root}")

    target, base, name = create_cell(gen_root, args.level, args.scene, args.strategy, args.name)
    instructions = assemble(gen_root, target, args.level, base, name, args.extra)
    print(f"cell: {target}\ninstructions: {instructions}")
    if args.dry:
        return
    os.chdir(target)
    os.execvp(args.agent, [args.agent, "--model", args.model,
                           "Read .agent/instructions.md and begin."])


if __name__ == "__main__":
    main()
