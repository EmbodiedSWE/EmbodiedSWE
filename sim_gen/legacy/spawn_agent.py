"""Spawn Claude Code construction agents with trajectory logging (stage 2).

Agent API traffic is routed through the vendored super_relay proxy (sim_gen/super_relay)
so every request/response is logged; trajectories export to artifacts/trajectories/ in
the training_trajs.jsonl schema. Every run writes a machine-readable provenance record
to artifacts/reports/<task>.run.json (seed, model, agent session id, exit code).

Environment overrides (all optional; defaults work on this repo's standard layout):
  SIM_GEN_PYTHON     python interpreter for relay/validator subprocesses
  SIM_GEN_RELAY_DIR  super_relay checkout (default: the vendored sim_gen/super_relay)
  SIM_GEN_RELAY_PORT relay port (default 8118)
  SIM_GEN_OAUTH_ENV  file exporting CLAUDE_CODE_OAUTH_TOKEN (default ~/.claude_oauth_env)

Usage:
  # one task from one seed (task name auto-derived; --task-name to override)
  python sim_gen/pipeline/spawn_agent.py --seed rlbench/light_bulb_in --tier middle

  # a batch: N seeds sampled uniformly from the deduplicated pool
  python sim_gen/pipeline/spawn_agent.py --batch 10 --rng-seed 0 --tier easy

  python sim_gen/pipeline/spawn_agent.py --start-relay-only
  python sim_gen/pipeline/spawn_agent.py --export-trajs
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent
COSIGEN_ROOT = SIM_GEN_ROOT.parent
ARTIFACTS = SIM_GEN_ROOT / "artifacts"
RELAY_LOG_DIR = ARTIFACTS / "relay_logs"
TRAJ_DIR = ARTIFACTS / "trajectories"
AGENT_LOG_DIR = ARTIFACTS / "agent_logs"
REPORTS = ARTIFACTS / "reports"

PYTHON = os.environ.get("SIM_GEN_PYTHON") or sys.executable
RELAY_DIR = Path(os.environ.get("SIM_GEN_RELAY_DIR", SIM_GEN_ROOT / "super_relay"))
RELAY_PORT = int(os.environ.get("SIM_GEN_RELAY_PORT", "8118"))
OAUTH_ENV = Path(os.environ.get("SIM_GEN_OAUTH_ENV",
                                Path.home() / ".claude_oauth_env"))


def relay_raw_log() -> Path | None:
    """The running relay's raw request log (from /health), or None if not running.
    A relay may predate this script with a different --log-dir; always ask it."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{RELAY_PORT}/health",
                                    timeout=2) as r:
            return Path(json.loads(r.read())["log_file"])
    except Exception:
        return None


def start_relay() -> None:
    if relay_raw_log() is not None:
        print(f"[relay] already running on :{RELAY_PORT}")
        return
    RELAY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(RELAY_LOG_DIR / "relay_stdout.log", "ab")
    subprocess.Popen(
        [PYTHON, str(RELAY_DIR / "server.py"), "--port", str(RELAY_PORT),
         "--log-dir", str(RELAY_LOG_DIR)],
        stdout=out, stderr=subprocess.STDOUT, cwd=str(RELAY_DIR),
        start_new_session=True,
    )
    for _ in range(30):
        if relay_raw_log() is not None:
            print(f"[relay] started on :{RELAY_PORT}, logs -> {RELAY_LOG_DIR}")
            return
        time.sleep(0.5)
    raise RuntimeError("relay did not come up; see relay_stdout.log")


def oauth_token() -> str:
    for line in OAUTH_ENV.read_text().splitlines():
        if "CLAUDE_CODE_OAUTH_TOKEN=" in line:
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no CLAUDE_CODE_OAUTH_TOKEN in {OAUTH_ENV}")


def derive_task_name(seed_id: str) -> str:
    """<seed_stem>_dN — first free descendant index for this seed."""
    stem = re.sub(r"[^a-z0-9_]", "_", seed_id.split("/")[-1].lower())
    n = 1
    while (SIM_GEN_ROOT / "tasks" / f"{stem}_d{n}").exists():
        n += 1
    return f"{stem}_d{n}"


def spawn(seed_id: str, task_name: str | None, model: str, tier: str) -> int:
    from prompt import build_prompt
    from seeds import get_seed

    start_relay()
    seed_path, _ = get_seed(seed_id)
    task_name = task_name or derive_task_name(seed_id)
    (SIM_GEN_ROOT / "tasks" / task_name).mkdir(parents=True, exist_ok=True)
    prompt_text = build_prompt(seed_id, seed_path, task_name, PYTHON, tier)

    AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    log_path = AGENT_LOG_DIR / f"{task_name}.log"

    env = dict(os.environ)
    env.update({
        "CLAUDE_CODE_OAUTH_TOKEN": oauth_token(),
        # 127.0.0.1, not localhost: CC resolves localhost to ::1 and the relay
        # binds IPv4 (see super_relay README).
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{RELAY_PORT}",
        "ANTHROPIC_API_KEY": "",
        "MUJOCO_GL": "osmesa",
    })
    cmd = ["claude", "-p", prompt_text, "--model", model,
           "--dangerously-skip-permissions", "--verbose",
           "--output-format", "stream-json"]
    print(f"[agent] spawning ({model}, tier={tier}) seed={seed_id} "
          f"task={task_name}; log -> {log_path}")
    t0 = datetime.datetime.now().isoformat(timespec="seconds")
    with open(log_path, "wb") as log:
        rc = subprocess.call(cmd, cwd=str(COSIGEN_ROOT), env=env,
                             stdout=log, stderr=subprocess.STDOUT)

    # provenance record: link task <-> seed <-> agent session <-> trajectory source
    session_id = None
    m = re.search(rb'"session_id"\s*:\s*"([0-9a-f-]+)"', log_path.read_bytes())
    if m:
        session_id = m.group(1).decode()
    run = {"task": task_name, "seed": seed_id, "tier": tier, "model": model,
           "session_id": session_id, "rc": rc, "started": t0,
           "finished": datetime.datetime.now().isoformat(timespec="seconds"),
           "agent_log": str(log_path), "relay_raw_log": str(relay_raw_log())}
    (REPORTS / f"{task_name}.run.json").write_text(json.dumps(run, indent=2))
    print(f"[agent] exited rc={rc}; run record -> {REPORTS}/{task_name}.run.json")
    return rc


def export_trajs() -> None:
    raw = relay_raw_log()
    if raw is None or not raw.exists():
        raise RuntimeError("relay not running / no raw log — nothing to export")
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    out = TRAJ_DIR / "training_trajs.jsonl"
    subprocess.check_call(
        [PYTHON, str(RELAY_DIR / "build_training_trajs.py"),
         "--raw-log", str(raw), "--output", str(out), "--min-messages", "4"])
    n = sum(1 for _ in open(out))
    print(f"[trajs] exported {n} trajectories from {raw} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str,
                        help="seed id from the deduplicated pool (pipeline/seeds.py)")
    parser.add_argument("--task-name", type=str, default=None,
                        help="override the auto-derived task package name")
    parser.add_argument("--batch", type=int, default=0,
                        help="construct N tasks from uniformly sampled pool seeds")
    parser.add_argument("--rng-seed", type=int, default=0, help="batch sampling seed")
    parser.add_argument("--model", type=str, default="claude-fable-5")
    parser.add_argument("--tier", type=str, default="easy",
                        choices=("easy", "middle", "hard"))
    parser.add_argument("--start-relay-only", action="store_true")
    parser.add_argument("--export-trajs", action="store_true")
    args = parser.parse_args()

    if args.start_relay_only:
        start_relay()
        return
    if args.export_trajs:
        export_trajs()
        return
    if args.batch:
        from seeds import sample_seeds

        failures = 0
        for seed_id in sample_seeds(args.batch, args.rng_seed):
            failures += spawn(seed_id, None, args.model, args.tier) != 0
        print(f"[batch] done: {args.batch - failures}/{args.batch} agents exited cleanly")
        sys.exit(1 if failures else 0)
    if not args.seed:
        parser.error("--seed (or --batch N) is required to spawn")
    sys.exit(spawn(args.seed, args.task_name, args.model, args.tier))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
