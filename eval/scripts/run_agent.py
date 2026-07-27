#!/usr/bin/env python3
"""Launch ONE agent run against a built experiment stage.

    python eval/scripts/run_agent.py experiments/<exp> \\
        [--agent claude] [--model <id>] [--budget-min 240] [--gpu 0] \\
        [--run NAME] [--dry-run]

This is the DEFAULT runner: one scene + one task — it only accepts
single-stage experiments (sequence/transfer experiments get their own future
script). The CLI selects INFRASTRUCTURE only: which experiment
(scene+robot+controller were fixed when it was built), which agent/model,
GPU, budget. The PROMPT CONDITION — hints, rules — comes exclusively from the
condition yaml (--config; default: eval/prompts/configs/default.yaml), so
every condition is an authored, reviewable file, never an ad-hoc flag. One run = one disposable container
(contract: eval/docker/README.md §2). The WORLD comes from the experiment
folder (built once by build_env.py); the PROMPT CONDITION is chosen here,
per run:
  1. picks the stage dir (default: the first under stages/)
  2. assembles <exp>/runs/<run>/task/ — instructions + task (from the stage's
     describe.md) + selected rules + selected hints. Everything except the
     auto-generated task text is hand-authored library content, included only
     when selected; contradicting the world refuses to launch.
  3. creates <exp>/runs/<run>/workspace (the agent's only writable dir)
  4. docker run rb-l1-agent with /bench,/task ro mounts + CDI GPU
  5. waits up to the budget, then docker stop; collects container log
  6. writes <exp>/runs/<run>/run.json (raw record; metrics are post-hoc)

Credentials come from YOUR shell env (CLAUDE_CODE_OAUTH_TOKEN or
ANTHROPIC_API_KEY for claude; OPENAI_API_KEY for codex) — passed through,
never stored. Open network (dev). Isolated-network runs: use
eval/docker/compose.agent.yaml.
"""

from __future__ import annotations

import argparse
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


def single_stage(exp: Path) -> Path:
    stages = sorted((exp / "stages").iterdir())
    if not stages:
        sys.exit(f"no stages under {exp}")
    if len(stages) > 1:
        sys.exit(f"run_agent runs one scene + one task; this experiment has "
                 f"{len(stages)} stages {[s.name for s in stages]} — sequence "
                 "experiments get their own runner")
    return stages[0]


def stage_record(exp: Path, stage: Path) -> dict:
    """The stage's entry in the world receipt."""
    receipt = json.loads((exp / "resolved.json").read_text())
    hits = [r for r in receipt["stages"] if r["dir"] == stage.name]
    if not hits:
        sys.exit(f"stage {stage.name} not in {exp / 'resolved.json'} — rebuild the experiment")
    record = dict(hits[0])
    record["set_states"] = receipt.get("set_states", True)
    record.setdefault("control_mode_frozen", False)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", help="built experiment dir (from build_env.py)")
    ap.add_argument("--config", default=str(prompts.PROMPTS_DIR / "configs" / "default.yaml"),
                    help="condition yaml with defaults for any option; CLI flags override "
                         "(default: %(default)s)")
    ap.add_argument("--agent", default=None, choices=["claude", "codex"])
    ap.add_argument("--model", default=None, help="model override (MODEL env for the agent CLI)")
    ap.add_argument("--budget-min", type=float, default=None, help="wall-clock kill budget (minutes)")
    ap.add_argument("--auto-submit-min", type=float, default=None,
                    help="also snapshot solution/ as a submission every N minutes (skipped when "
                         "unchanged) — uniform curve sampling even if the agent never submits")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--run", default=None, help="run name (default: <agent>_<timestamp>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the task folder, print the docker command, and exit")
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
    hints = list(cfg.get("hints") or [])   # condition only — no CLI override
    rules = list(cfg.get("rules") or [])
    budget_min = float(pick(args.budget_min, "budget_min") or 240)
    auto_submit_min = pick(args.auto_submit_min, "auto_submit_min")

    exp = Path(exp_arg).resolve()
    stage = single_stage(exp)
    record = stage_record(exp, stage)
    facts = {"set_states": record["set_states"],
             "control_mode_frozen": record["control_mode_frozen"]}
    prompts.check_condition(hints, rules, facts)  # fail before touching disk
    if not record["set_states"] and not rules:
        print("note: this world restricts set_states and the run discloses nothing (no rules selected)")

    describe_file = stage / "describe.md"
    if not describe_file.exists():
        sys.exit(f"missing {describe_file} — rebuild the experiment with the current build_env.py")

    run_name = pick(args.run, "run") or f"{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = exp / "runs" / run_name
    if run_dir.exists():
        sys.exit(f"refusing to overwrite existing {run_dir}")

    task_dir = prompts.render_task_dir(
        run_dir / "task",
        scene=record["preset"].split(".")[1],
        preset=record["preset"],
        describe_text=describe_file.read_text(),
        facts=facts,
        hints=hints,
        rules=rules,
        budget_min=budget_min,
    )
    task_files = {
        str(p.relative_to(task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(task_dir.rglob("*")) if p.is_file()
    }
    workspace = run_dir / "workspace"
    submissions = run_dir / "submissions"

    model = pick(args.model, "model")
    gpu = pick(args.gpu, "gpu", "0")
    image = cfg.get("image", DEFAULT_IMAGE)

    cname = f"rb_{exp.name}_{run_name}"
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
        print(f"task folder assembled: {task_dir}")
        print(shlex.join(cmd))
        return

    if not any(os.environ.get(v) for v in CRED_VARS):
        sys.exit(f"no credential in env — set one of {CRED_VARS}")

    workspace.mkdir(parents=True)
    submissions.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    subprocess.run(cmd, check=True)
    print(f"running: container {cname}\n  watch:  tail -f {workspace}/.agent/transcript.jsonl"
          f"\n  peek:   docker exec -it {cname} bash\n  budget: {budget_min} min")

    import shutil
    import time

    t0 = time.time()

    def transcript_lines() -> int:
        transcript = workspace / ".agent" / "transcript.jsonl"
        return sum(1 for _ in transcript.open()) if transcript.exists() else 0

    auto_state = {"last": t0, "hash": None}

    def auto_submit() -> None:
        """Snapshot solution/ on the harness's clock — one curve point every
        --auto-submit-min even if the agent never submits. Skips unchanged
        solutions; stamps submitted.json itself (with auto: true)."""
        sol = workspace / "solution"
        if not (sol / "solve.py").exists():
            return
        h = hashlib.sha256()
        for p in sorted(sol.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(sol)).encode())
                h.update(p.read_bytes())
        digest = h.hexdigest()
        if digest == auto_state["hash"]:
            return
        nums = [int(d.name) for d in submissions.iterdir() if d.is_dir() and d.name.isdigit()]
        name = f"{max(nums, default=0) + 1:02d}"
        tmp = submissions / f".tmp_{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(sol, tmp)
        (tmp / "submitted.json").write_text(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_s": round(time.time() - t0, 1),
            "transcript_lines": transcript_lines(),
            "auto": True,
        }, indent=2) + "\n")
        tmp.rename(submissions / name)
        auto_state["hash"] = digest
        print(f"auto-submission: {name}  (wall {round(time.time() - t0)}s)", flush=True)

    def scan_submissions() -> None:
        """Stamp new submission snapshots with wall clock + transcript position
        — the pointers that later price each one in tokens (submissions are
        atomic: `submit` mv's completed copies into place)."""
        if not submissions.is_dir():
            return
        for d in sorted(submissions.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or (d / "submitted.json").exists():
                continue
            lines = transcript_lines()
            (d / "submitted.json").write_text(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_s": round(time.time() - t0, 1),
                "transcript_lines": lines,
            }, indent=2) + "\n")
            print(f"submission: {d.name}  (wall {round(time.time() - t0)}s, "
                  f"{lines} transcript lines)", flush=True)

    status, exit_code = "completed", None
    while True:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}},{{.State.ExitCode}}",
                            cname], capture_output=True, text=True)
        scan_submissions()
        if r.returncode != 0:
            break  # container vanished
        running, _, code = r.stdout.strip().partition(",")
        if running != "true":
            exit_code = int(code)
            break
        if auto_submit_min and time.time() - auto_state["last"] >= float(auto_submit_min) * 60:
            auto_state["last"] = time.time()
            auto_submit()
        if time.time() - t0 > budget_min * 60:
            status = "timeout"
            print(f"budget reached — stopping {cname}")
            subprocess.run(["docker", "stop", "-t", "30", cname], check=False)
            break
        time.sleep(5)
    scan_submissions()  # catch a submission from the final seconds
    if auto_submit_min:
        auto_submit()  # the final state, cost-pinned — the curve's last point

    logs = subprocess.run(["docker", "logs", cname], capture_output=True)
    (run_dir / "container.log").write_bytes(logs.stdout + logs.stderr)
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True, check=False)

    record_out = {
        "exp": str(exp), "stage": stage.name, "preset": record["preset"],
        "set_states": record["set_states"],
        "control_mode_frozen": record["control_mode_frozen"],
        "hints": hints, "rules": rules,
        "task_files": task_files,
        "config": args.config, "agent": agent, "model": model,
        "image": image, "gpu": gpu, "budget_min": budget_min,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status, "container_exit_code": exit_code, "argv": sys.argv,
    }
    (run_dir / "run.json").write_text(json.dumps(record_out, indent=2) + "\n")
    print(f"{status}: raw artifacts in {run_dir}  (workspace/, task/, container.log, run.json)")


if __name__ == "__main__":
    main()
