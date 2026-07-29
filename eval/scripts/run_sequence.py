#!/usr/bin/env python3
"""Run a multi-stage experiment: one agent session per stage, workspace carried forward.

    python eval/scripts/run_sequence.py experiments/screw_transfer \\
        [--config eval/prompts/configs/<cond>.yaml] [--agent claude] \\
        [--model <id>] [--budget-min 240] [--gpu 0] [--run NAME] [--dry-run]

The transfer/continual-learning runner. Stages run strictly in order, each as
a FRESH session in its own disposable container (continuity is carried by
files, never by the agent's conversation):

  stage 01: empty workspace
  stage 02+: workspace = a COPY of the previous stage's final workspace,
             minus .agent/ (harness telemetry is not agent work-product);
             the stage's instructions carry the carryover note

Every stage always runs — success is not machine-detectable until grading,
and how partial artifacts transfer is part of the study. The
'transfer_experience' rule (if selected in the condition) applies to every
stage: read /workspace/experiences/, maintain your own note there. Per-stage
artifacts mirror run_agent's; sequence.json is the rollup. The no-transfer
baseline needs no support here: build the later scene as its own
single-stage world and run it cold — equal tree hashes prove the
comparison fair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envbuild import prompts  # noqa: E402

DEFAULT_IMAGE = "rb-l1-agent:2.1.216"
CRED_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def stage_records(exp: Path) -> list[dict]:
    receipt = json.loads((exp / "resolved.json").read_text())
    records = sorted(receipt["stages"], key=lambda r: r["dir"])
    for r in records:
        r["set_states"] = receipt.get("set_states", True)
        r.setdefault("control_mode_frozen", False)
    return records


def seed_workspace(ws: Path, prev_ws: Path | None) -> None:
    """Empty for the first stage; a copy of the previous stage's final
    workspace (minus harness telemetry) afterwards."""
    if prev_ws is not None and prev_ws.exists():
        shutil.copytree(prev_ws, ws, ignore=shutil.ignore_patterns(".agent"))
    else:
        ws.mkdir(parents=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", help="built multi-stage experiment dir (from build_env.py)")
    ap.add_argument("--config", default=str(prompts.PROMPTS_DIR / "configs" / "default.yaml"),
                    help="condition yaml, applied to every stage; CLI flags override "
                         "(default: %(default)s)")
    ap.add_argument("--agent", default=None, choices=["claude", "codex"])
    ap.add_argument("--model", default=None, help="model override (MODEL env for the agent CLI)")
    ap.add_argument("--budget-min", type=float, default=None, help="wall-clock kill budget PER STAGE (minutes)")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--run", default=None, help="sequence name (default: <agent>_seq_<timestamp>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble every stage's task folder, print the docker commands, and exit")
    args = ap.parse_args()

    import yaml
    loaded = yaml.safe_load(Path(args.config).read_text())
    if loaded is not None and not isinstance(loaded, dict):
        sys.exit(f"--config must be a yaml mapping, got {type(loaded).__name__}")
    cfg: dict = loaded or {}

    def pick(cli_value, key, default=None):
        return cli_value if cli_value is not None else cfg.get(key, default)

    exp_arg = pick(args.exp, "exp")
    if not exp_arg:
        ap.error("an experiment dir is required (positional or 'exp:' in --config)")
    agent = pick(args.agent, "agent", "claude")
    hints = list(cfg.get("hints") or [])
    rules = list(cfg.get("rules") or [])
    budget_min = float(pick(args.budget_min, "budget_min") or 240)
    model = pick(args.model, "model")
    gpu = pick(args.gpu, "gpu", "0")
    image = cfg.get("image", DEFAULT_IMAGE)

    exp = Path(exp_arg).resolve()
    records = stage_records(exp)
    stages = [exp / "stages" / r["dir"] for r in records]
    for st in stages:
        if not (st / "describe.md").exists():
            sys.exit(f"missing {st / 'describe.md'} — rebuild the experiment with the current build_env.py")

    # fail on any prompt-vs-world contradiction before touching disk, for every stage
    for r in records:
        facts = {"set_states": r["set_states"], "control_mode_frozen": r["control_mode_frozen"]}
        prompts.check_condition(hints, rules, facts)

    run_name = pick(args.run, "run") or f"{agent}_seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seq_dir = exp / "runs" / run_name
    if seq_dir.exists():
        sys.exit(f"refusing to overwrite existing {seq_dir}")

    if not args.dry_run and not any(os.environ.get(v) for v in CRED_VARS):
        sys.exit(f"no credential in env — set one of {CRED_VARS}")

    rollup = []
    prev_ws: Path | None = None
    for i, (r, stage) in enumerate(zip(records, stages)):
        stage_run = seq_dir / r["dir"]
        workspace = stage_run / "workspace"
        submissions = stage_run / "submissions"
        stage_rules = list(rules)  # transfer_experience applies to every stage (it reads AND writes)
        facts = {"set_states": r["set_states"], "control_mode_frozen": r["control_mode_frozen"]}

        task_dir = prompts.render_task_dir(
            stage_run / "task",
            scene=r["preset"].split(".")[1],
            preset=r["preset"],
            describe_text=(stage / "describe.md").read_text(),
            facts=facts,
            hints=hints,
            rules=stage_rules,
            budget_min=budget_min,
            carryover=i > 0,
        )
        task_files = {
            str(p.relative_to(task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(task_dir.rglob("*")) if p.is_file()
        }

        cname = f"rb_{exp.name}_{run_name}_{r['dir']}"
        cmd = [
            "docker", "run", "-d", "--name", cname,
            "--device", f"nvidia.com/gpu={gpu}", "--shm-size", "2g",
            "-v", f"{stage / 'bench'}:/bench:ro",
            "-v", f"{task_dir}:/task:ro",
            "-v", f"{workspace}:/workspace",
            "-v", f"{submissions}:/submissions",
            "-v", "rb-ovcache:/ovcache",
            "-e", f"AGENT={agent}",
        ]
        if model:
            cmd += ["-e", f"MODEL={model}"]
        for var in CRED_VARS:
            if os.environ.get(var):
                cmd += ["-e", var]
        cmd += [image, "/opt/entrypoints/agent-entry.sh"]

        if args.dry_run:
            import shlex
            print(f"[{r['dir']}] task folder assembled: {task_dir}")
            print(shlex.join(cmd))
            continue

        seed_workspace(workspace, prev_ws)
        submissions.mkdir(parents=True)
        started = datetime.now(timezone.utc)
        subprocess.run(cmd, check=True)
        print(f"[{r['dir']}] running: {cname}  (budget {budget_min} min)"
              f"\n  watch: tail -f {workspace}/.agent/transcript.jsonl")

        status, exit_code = "completed", None
        try:
            w = subprocess.run(["docker", "wait", cname], capture_output=True, text=True,
                               timeout=budget_min * 60, check=True)
            exit_code = int(w.stdout.strip())
        except subprocess.TimeoutExpired:
            status = "timeout"
            print(f"[{r['dir']}] budget reached — stopping {cname}")
            subprocess.run(["docker", "stop", "-t", "30", cname], check=False)

        logs = subprocess.run(["docker", "logs", cname], capture_output=True)
        (stage_run / "container.log").write_bytes(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True, check=False)

        record_out = {
            "exp": str(exp), "stage": r["dir"], "stage_index": i, "preset": r["preset"],
            "set_states": r["set_states"], "control_mode_frozen": r["control_mode_frozen"],
            "hints": hints, "rules": stage_rules, "carryover": i > 0,
            "task_files": task_files, "config": args.config, "agent": agent, "model": model,
            "image": image, "gpu": gpu, "budget_min": budget_min,
            "started": started.isoformat(timespec="seconds"),
            "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status, "container_exit_code": exit_code,
        }
        (stage_run / "run.json").write_text(json.dumps(record_out, indent=2) + "\n")
        rollup.append({"stage": r["dir"], "status": status, "container_exit_code": exit_code,
                       "started": record_out["started"], "ended": record_out["ended"]})
        print(f"[{r['dir']}] {status}: artifacts in {stage_run}")
        prev_ws = workspace

    if args.dry_run:
        return
    (seq_dir / "sequence.json").write_text(json.dumps({
        "exp": str(exp), "run": run_name, "config": args.config, "agent": agent,
        "model": model, "budget_min_per_stage": budget_min, "stages": rollup,
        "argv": sys.argv,
    }, indent=2) + "\n")
    print(f"sequence done: {seq_dir}  (per-stage run.json + sequence.json)")


if __name__ == "__main__":
    main()
