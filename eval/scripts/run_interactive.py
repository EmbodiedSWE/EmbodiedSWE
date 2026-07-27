#!/usr/bin/env python3
"""Open an INTERACTIVE agent session inside the sandbox — the human-in-the-loop mode.

    python eval/scripts/run_interactive.py experiments/<exp> \\
        [--config eval/prompts/configs/<cond>.yaml] [--model <id>] [--gpu 0] [--run NAME]

Same world, same mounts, same /task assembly as run_agent.py — but instead of
a headless one-shot, the agent CLI runs interactively inside a tmux session in
the container. The container runs detached; you attach when you want, watch or
type feedback, detach, and it keeps its session:

    attach:  docker exec -it <container> runuser -u agent -- tmux attach -t agent
    detach:  Ctrl-b d
    stop:    docker stop <container>       (workspace stays on the host)

Caveat: this is collaboration, not measurement — interactive sessions pause at
the end of each turn until a human replies, so budgets, turn counts, and
wall-clock are NOT comparable with run_agent results. Use it for coached
solving, debugging a world, or demos.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envbuild import prompts  # noqa: E402

DEFAULT_IMAGE = "rb-l1-agent:2.1.216"
CRED_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="built experiment dir (from build_env.py)")
    ap.add_argument("--config", default=str(prompts.PROMPTS_DIR / "configs" / "interactive.yaml"),
                    help="condition yaml (default: %(default)s)")
    ap.add_argument("--model", default=None, help="model override for the agent CLI")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--run", default=None, help="run name (default: interactive_<timestamp>)")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    if not isinstance(cfg, dict):
        sys.exit(f"--config must be a yaml mapping, got {type(cfg).__name__}")
    hints = list(cfg.get("hints") or [])
    rules = list(cfg.get("rules") or [])

    exp = Path(args.exp).resolve()
    stages = sorted((exp / "stages").iterdir())
    if len(stages) != 1:
        sys.exit(f"run_interactive runs one stage; this experiment has {len(stages)}")
    stage = stages[0]
    receipt = json.loads((exp / "resolved.json").read_text())
    record = next(r for r in receipt["stages"] if r["dir"] == stage.name)
    facts = {"set_states": receipt.get("set_states", True),
             "control_mode_frozen": record.get("control_mode_frozen", False)}
    prompts.check_condition(hints, rules, facts)

    if not any(os.environ.get(v) for v in CRED_VARS):
        sys.exit(f"no credential in env — set one of {CRED_VARS}")

    run_name = args.run or f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = exp / "runs" / run_name
    if run_dir.exists():
        sys.exit(f"refusing to overwrite existing {run_dir}")

    task_dir = prompts.render_task_dir(
        run_dir / "task",
        scene=record["preset"].split(".")[1],
        preset=record["preset"],
        describe_text=(stage / "describe.md").read_text(),
        facts=facts,
        hints=hints,
        rules=rules,
    )
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    submissions = run_dir / "submissions"
    submissions.mkdir(parents=True)

    cname = f"rb_{exp.name}_{run_name}"
    cmd = [
        "docker", "run", "-d", "--name", cname,
        "--device", f"nvidia.com/gpu={args.gpu}", "--shm-size", "2g",
        "-v", f"{stage / 'bench'}:/bench:ro",
        "-v", f"{task_dir}:/task:ro",
        "-v", f"{workspace}:/workspace",
        "-v", f"{submissions}:/submissions",
        "-v", "rb-ovcache:/ovcache",
    ]
    if args.model:
        cmd += ["-e", f"MODEL={args.model}"]
    for var in CRED_VARS:
        if os.environ.get(var):
            cmd += ["-e", var]
    cmd += [args.image, "/opt/entrypoints/interactive-entry.sh"]
    subprocess.run(cmd, check=True)

    (run_dir / "run.json").write_text(json.dumps({
        "exp": str(exp), "stage": stage.name, "preset": record["preset"],
        "mode": "interactive", "hints": hints, "rules": rules,
        "config": args.config, "model": args.model, "image": args.image, "gpu": args.gpu,
        "task_files": {str(p.relative_to(task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(task_dir.rglob("*")) if p.is_file()},
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
    }, indent=2) + "\n")

    print(f"interactive session up: {cname}"
          f"\n  attach:  docker exec -it {cname} runuser -u agent -- tmux attach -t agent"
          f"\n  detach:  Ctrl-b d   (session keeps running)"
          f"\n  files:   {workspace}"
          f"\n  stop:    docker stop {cname}")


if __name__ == "__main__":
    main()
