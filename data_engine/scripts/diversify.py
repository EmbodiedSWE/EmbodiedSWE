"""Launch a diversify authoring session on a data_gen campaign.

    python data_engine/scripts/diversify.py <gen_root> --level scene|strategy|phase \\
        [--scene scene_0] [--strategy strategy_0] [--extra notes.md ...] \\
        [--agent claude] [--model claude-opus-5] [--dry]

The launcher creates NO cell — it only assembles the session instructions
(general contract + level template + campaign facts + any --extra prompt files)
into the campaign's session ledger, `<gen_root>/.agent/<ts>_<level>/instructions.md`,
and hands the terminal to the agent CLI at the campaign root:

    <agent> --model <model> "Read <instructions> and begin."

The agent's first authoring act is creating its first cell with
agent/tools/create_cell.py — one cell per variant it proposes. Any agent CLI
with that calling shape plugs in via --agent; claude with Opus is the testbed.
--dry assembles everything and prints the instructions path instead of launching.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assemble(gen_root: Path, level: str, scene: str, strategy: str,
             extras: list[str]) -> Path:
    contract = (ROOT / "agent" / "prompts" / "_contract.md").read_text()
    template = (ROOT / "agent" / "prompts" / f"{level}.md").read_text()
    template = template.replace("{base}", strategy if level == "phase" else scene)
    facts = [
        "\n\n" + template.strip(),
        "\n## This session\n",
        f"- campaign: {gen_root}  (your cwd; work only in here)",
        f"- level: {level}" + ("" if level == "scene" else f" — context: {scene}"
                               + (f"/{strategy}" if level == "phase" else "")),
        "- create each cell yourself, one per proposed variant:\n"
        f"  `python {ROOT}/agent/tools/create_cell.py {gen_root} --level {level} …`",
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

    session = gen_root / ".agent" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{level}"
    session.mkdir(parents=True)
    out = session / "instructions.md"
    out.write_text(contract + "\n".join(facts) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="launch a diversify authoring session")
    ap.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
    ap.add_argument("--level", required=True, choices=("scene", "strategy", "phase"))
    ap.add_argument("--scene", default="scene_0", help="context scene for the session")
    ap.add_argument("--strategy", default="strategy_0", help="context strategy (phase level)")
    ap.add_argument("--extra", action="append", default=[], help="extra prompt file(s) appended to the instructions")
    ap.add_argument("--agent", default="claude", help="agent CLI to launch")
    ap.add_argument("--model", default="claude-opus-5", help="model handed to the agent CLI")
    ap.add_argument("--dry", action="store_true", help="assemble only; print the instructions path")
    args = ap.parse_args()

    gen_root = Path(args.gen_root).resolve()
    if not (gen_root / "gen.yaml").is_file():
        raise SystemExit(f"not a campaign (no gen.yaml): {gen_root}")

    instructions = assemble(gen_root, args.level, args.scene, args.strategy, args.extra)
    print(f"session: {instructions.parent}\ninstructions: {instructions}")
    if args.dry:
        return
    os.chdir(gen_root)
    os.execvp(args.agent, [args.agent, "--model", args.model,
                           f"Read {instructions} and begin."])


if __name__ == "__main__":
    main()
