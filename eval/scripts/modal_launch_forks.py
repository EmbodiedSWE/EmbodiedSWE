#!/usr/bin/env python3
"""Fork experiment (user directive 2026-08-15): Fable 5's post-nut_thread agent state —
conversation (agent_home.tgz) AND workspace — forked into 4 independent agents, each
continuing that same conversation onto a NEW task: bulb, ikea_table, pc_ram, spatula.

Mechanics: modal_launch_campaign's launcher pattern, with run_agent_runpod's
--resume-from (restores workspace + conversation on the pod) + --resume-new-task (the
first leg delivers the new task's carryover instructions INTO the resumed conversation,
instead of the keep-going nudge). Submissions start fresh; workspace telemetry (.agent/,
tmp/) was stripped from the fork source, so transcripts and logs are per-fork.

    modal run --detach eval/scripts/modal_launch_forks.py
    modal run eval/scripts/modal_launch_forks.py --only bulb --budget-min 8
"""
from __future__ import annotations

import modal

REPO = "/repo"
RUNS = "/runs"
FORK_SRC = f"{REPO}/fork_src/nut_fable5"     # workspace/ + agent_home.tgz (no submissions)
MODEL = "claude-fable-5"
BUDGET_MIN_DEFAULT = 240.0
RUN_NAME = "fable_5_forknut"

TASKS = {
    "bulb": "experiments/bulb_notools",
    "ikea_table": "experiments/ikea_table_notools",
    "pc_ram": "experiments/pc_ram_notools",
    "spatula": "experiments/spatula_notools",
}

app = modal.App("cosigen-forks")
runs_volume = modal.Volume.from_name("cosigen-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openssh-client", "rsync", "tar")
    .pip_install("pyyaml")
    .add_local_dir("eval", remote_path=f"{REPO}/eval")
    .add_local_dir("sim_gen/super_relay", remote_path=f"{REPO}/sim_gen/super_relay",
                   ignore=["logs/**"])
    .add_local_dir("experiments", remote_path=f"{REPO}/experiments",
                   ignore=["**/runs/**"])
    .add_local_dir("eval_result/fork_src/nut_fable5", remote_path=FORK_SRC)
    .add_local_file("/Users/bytedance/.ssh/cosigen_campaign",
                    remote_path="/root/.ssh/cosigen_campaign")
    .add_local_file("/Users/bytedance/.ssh/cosigen_campaign.pub",
                    remote_path="/root/.ssh/cosigen_campaign.pub")
)


@app.function(image=image, volumes={RUNS: runs_volume}, timeout=6 * 60 * 60,
              retries=0, max_containers=6)
def run_one(task: str, budget_min: float = BUDGET_MIN_DEFAULT) -> str:
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    exp = Path(REPO) / TASKS[task]
    label = f"{task}_fable_5_forknut"
    subprocess.run(["chmod", "600", "/root/.ssh/cosigen_campaign"], check=True)
    cmd = [
        sys.executable, f"{REPO}/eval/scripts/run_agent_runpod.py", str(exp),
        "--config", "no_tools",
        "--agent", "claude",
        "--model", MODEL,
        "--force-model", MODEL,
        "--budget-min", str(budget_min),
        "--keep-going",
        "--auto-submit-min", "30",
        "--run", RUN_NAME,
        "--resume-from", FORK_SRC,
        "--resume-new-task",
        "--ssh-key", "/root/.ssh/cosigen_campaign",
    ]
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True)

    run_dir = exp / "runs" / RUN_NAME
    dest = Path(RUNS) / label
    if run_dir.is_dir():
        shutil.copytree(run_dir, dest, dirs_exist_ok=True)
        runs_volume.commit()
        print(f"[{label}] artifacts committed to volume at {dest}", flush=True)
    else:
        print(f"[{label}] NO RUN DIR — launcher exited rc={proc.returncode}", flush=True)
    return f"{label}: launcher rc={proc.returncode}"


@app.local_entrypoint()
def main(only: str = "", budget_min: float = BUDGET_MIN_DEFAULT):
    tasks = [t for t in (only.split(",") if only else TASKS) if t]
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r} (choose from {list(TASKS)})")
    print(f"forking nut_thread fable-5 state onto {tasks}, budget {budget_min} min each",
          flush=True)
    calls = [run_one.spawn(t, budget_min) for t in tasks]
    for c in calls:
        print(c.get(), flush=True)
