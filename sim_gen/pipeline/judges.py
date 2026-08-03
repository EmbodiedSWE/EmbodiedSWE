"""LLM judges (stage 5): solution legitimacy and description clarity.

Usage:  python sim_gen/pipeline/judges.py --task <task_name> [--which both|legit|clarity]
Writes artifacts/reports/<task_name>.judges.json; exit 0 iff every requested judge passes.
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
REPORTS = SIM_GEN_ROOT / "artifacts" / "reports"

LEGIT_PROMPT = """\
You are auditing the SOLUTION of a robot simulation task. The rule: the solution must
genuinely perform the described task through arm-only physical manipulation — it may
command the robot's joints and gripper, and read scene state, but it must NOT write
task-object states (e.g. write_root_state_to_sim on task objects), apply external
forces to task objects, weaken or bypass the rubric, or reach success() through a
loophole that avoids doing the described task.

## The task card (TASK.md)
{task_md}

## The rubric (scene.py excerpt)
```python
{scene_source}
```

## The solution (solve.py)
```python
{solve_source}
```

Answer with ONLY a JSON object, no other text:
{{"legitimate": true/false,
  "violations": ["<each concrete violation found, empty if none>"],
  "reasoning": "<2-4 sentences>"}}
"""

CLARITY_PROMPT = """\
You are judging whether a simulation task's natural-language statement is complete
enough that a competent solver could perform the task from the statement ALONE —
without reading the scene code. Complete means: the goal state is stated; every
target/object the solver must act on is identifiable from the statement (by color,
marker, position, or name visible in the scene); and any required ordering or
prohibition whose violation fails the task is stated.

## The statement (describe() from scene.py)
```python
{describe_source}
```

## What the task actually requires (rubric excerpt: success/score)
```python
{rubric_source}
```

Answer with ONLY a JSON object, no other text:
{{"clear": true/false,
  "missing": ["<each requirement of success() not deducible from the statement>"],
  "reasoning": "<2-4 sentences>"}}
"""


def _excerpt(src: str, names: tuple[str, ...]) -> str:
    """The named method bodies, so judge prompts stay within budget."""
    out = []
    for name in names:
        m = re.search(rf"def {name}\(self.*?(?=\n    def |\nregister_env|\Z)", src, re.S)
        if m:
            out.append(m.group(0))
    return "\n\n".join(out) or src[:6000]


def _ask(prompt: str, model: str) -> dict:
    env = dict(os.environ, ANTHROPIC_API_KEY="")
    if os.environ.get("SIMGEN_NOVELTY_BASE_URL"):
        env["ANTHROPIC_BASE_URL"] = os.environ["SIMGEN_NOVELTY_BASE_URL"]
    out = subprocess.run(["claude", "-p", prompt, "--model", model],
                         capture_output=True, text=True, env=env, timeout=600).stdout
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        raise RuntimeError(f"judge returned no JSON:\n{out[-1000:]}")
    return json.loads(m.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--which", default="both", choices=("both", "legit", "clarity"))
    parser.add_argument("--model", default="claude-fable-5")
    args = parser.parse_args()

    task_dir = Path(os.environ.get("SIM_GEN_TASKS_DIR", SIM_GEN_ROOT / "tasks")) / args.task
    scene_src = (task_dir / "scene.py").read_text()
    verdicts: dict = {"task": args.task}

    if args.which in ("both", "legit"):
        verdicts["legit"] = _ask(LEGIT_PROMPT.format(
            task_md=(task_dir / "TASK.md").read_text(),
            scene_source=_excerpt(scene_src, ("success", "score")),
            solve_source=(task_dir / "solve.py").read_text()), args.model)
    if args.which in ("both", "clarity"):
        verdicts["clarity"] = _ask(CLARITY_PROMPT.format(
            describe_source=_excerpt(scene_src, ("describe",)),
            rubric_source=_excerpt(scene_src, ("success", "score"))), args.model)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{args.task}.judges.json").write_text(json.dumps(verdicts, indent=2))
    ok_legit = verdicts.get("legit", {}).get("legitimate", True) is True
    ok_clear = verdicts.get("clarity", {}).get("clear", True) is True
    for name, ok, v in (("legit", ok_legit, verdicts.get("legit")),
                        ("clarity", ok_clear, verdicts.get("clarity"))):
        if v is not None:
            print(f"[judge:{name}] {'PASS' if ok else 'FAIL'} — {v.get('reasoning', '')}")
    sys.exit(0 if (ok_legit and ok_clear) else 1)


if __name__ == "__main__":
    main()
