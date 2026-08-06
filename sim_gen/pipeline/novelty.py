"""Novelty judgment (stage 5): LLM judge — is the new task strategically different
from its seed?  Judged within the seed family only (see PIPELINE.md).

Usage:  python sim_gen/pipeline/novelty.py --task <task_name> --seed <seed_id>
Writes artifacts/reports/<task_name>.novelty.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent
REPORTS = Path(os.environ.get("SIM_GEN_REPORTS_DIR",
                              SIM_GEN_ROOT / "artifacts" / "reports"))

JUDGE_PROMPT = """\
You are judging whether a NEW simulation task is STRATEGICALLY different from the SEED
task it was derived from. "Strategically different" means a solver needs a different
PLAN (different sequence of subgoals / different manipulation strategy), not merely
different parameters, sizes, distances, or object appearances.

## SEED task source
```python
{seed_source}
```

## NEW task
TASK.md:
{task_md}

scene.py (semantics excerpt — describe/success/score):
```python
{scene_source}
```

The demonstrated solution (solve.py — a teleport solution: teleports handle
transport, the load-bearing interactions run through contact dynamics):
```python
{smoke_source}
```

Answer with ONLY a JSON object, no other text:
{{"strategically_different": true/false,
  "seed_strategy": "<one sentence>",
  "new_strategy": "<one sentence>",
  "shared_structure": "<what carries over>",
  "reasoning": "<2-4 sentences>"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", default=None,
                        help="seed id; defaults to the task's run.json provenance")
    parser.add_argument("--model", default="claude-fable-5")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from seeds import get_seed

    seed_id = args.seed
    if seed_id is None:
        run_file = REPORTS / f"{args.task}.run.json"
        if not run_file.exists():
            raise SystemExit(f"--seed not given and no provenance at {run_file}")
        seed_id = json.loads(run_file.read_text())["seed"]

    _, seed_source = get_seed(seed_id)
    task_dir = Path(os.environ.get("SIM_GEN_TASKS_DIR", SIM_GEN_ROOT / "tasks")) / args.task
    # the demonstrated strategy is solve.py (teleport solution); legacy packages have
    # only smoke.py
    strategy_file = task_dir / "solve.py"
    if not strategy_file.exists():
        strategy_file = task_dir / "smoke.py"
    prompt = JUDGE_PROMPT.format(
        seed_source=seed_source,
        task_md=(task_dir / "TASK.md").read_text(),
        scene_source=(task_dir / "scene.py").read_text(),
        smoke_source=strategy_file.read_text(),
    )

    # Campaign mode: SIMGEN_NOVELTY_BASE_URL points at a relay that owns the upstream
    # key (platform billing); otherwise the default OAuth + logging-relay path.
    if os.environ.get("SIMGEN_NOVELTY_BASE_URL"):
        # OAuth mode (CLAUDE_CODE_OAUTH_TOKEN in env): the token is the credential and
        # the key stays empty. Key mode: nonempty placeholder — the CLI refuses to
        # start on an empty key, and the relay replaces client auth anyway.
        env = dict(os.environ,
                   ANTHROPIC_BASE_URL=os.environ["SIMGEN_NOVELTY_BASE_URL"],
                   ANTHROPIC_API_KEY=""
                   if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") else "relay-session")
    else:
        sys.path.insert(0, str(SIM_GEN_ROOT / "legacy_mujoco"))
        from spawn_agent import RELAY_PORT, oauth_token, start_relay
        start_relay()  # judge traffic is logged through the same relay
        env = dict(os.environ,
                   CLAUDE_CODE_OAUTH_TOKEN=oauth_token(),
                   ANTHROPIC_BASE_URL=f"http://127.0.0.1:{RELAY_PORT}",
                   ANTHROPIC_API_KEY="")
    out = subprocess.run(["claude", "-p", prompt, "--model", args.model],
                         capture_output=True, text=True, env=env,
                         timeout=600).stdout
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        raise RuntimeError(f"judge returned no JSON:\n{out[-1000:]}")
    verdict = json.loads(m.group(0))

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"{args.task}.novelty.json"
    report.write_text(json.dumps({"task": args.task, "seed": args.seed, **verdict},
                                 indent=2))
    ok = verdict.get("strategically_different") is True
    print(f"[novelty] {'PASS' if ok else 'FAIL'} — {verdict.get('reasoning', '')}")
    print(f"[novelty] report -> {report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
