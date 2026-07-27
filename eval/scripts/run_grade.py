#!/usr/bin/env python3
"""Grade ONE delivered solution in a fresh grading container.

    python eval/scripts/run_grade.py experiments/<exp> --run <run>          # grade that run's delivery
    python eval/scripts/run_grade.py experiments/<exp> --solution DIR --out DIR

Grading never runs on the host: the container gets the same image and the
same read-only /bench tree the agent had, plus the suite's grader/ at
/graders (never in agent-facing trees) and the delivery at /solution. No
credentials; --network none grades fully offline. The in-container driver is
eval/grader/grade.py. Artifacts land in <out>/: verdict.json, progress.jsonl,
container.log, grade.json. One trajectory per grade for now; seeds later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
REPO = EVAL.parent
DEFAULT_IMAGE = "rb-l1-agent:2.1.216"


def single_stage(exp: Path) -> Path:
    stages = sorted((exp / "stages").iterdir())
    if not stages:
        sys.exit(f"no stages under {exp}")
    if len(stages) > 1:
        sys.exit(f"run_grade.py grades one stage; this experiment has {len(stages)} — "
                 "pass the run of a sequence stage explicitly via --solution/--out")
    return stages[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="built experiment dir (from build_env.py)")
    ap.add_argument("--run", help="grade this run's workspace/solution (out: runs/<run>/grade)")
    ap.add_argument("--solution", help="explicit solution dir containing solve.py (needs --out)")
    ap.add_argument("--out", help="output dir (default: <exp>/runs/<run>/grade)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--network", default="bridge",
                    help="container network (assets are vendored, so --network none also works)")
    ap.add_argument("--budget-min", type=float, default=30, help="wall-clock kill budget (minutes)")
    ap.add_argument("--dry-run", action="store_true", help="print the docker command and exit")
    args = ap.parse_args()

    exp = Path(args.exp).resolve()
    stage = single_stage(exp)
    receipt = json.loads((exp / "resolved.json").read_text())
    hits = [r for r in receipt["stages"] if r["dir"] == stage.name]
    if not hits:
        sys.exit(f"stage {stage.name} not in {exp / 'resolved.json'} — rebuild the experiment")
    preset = hits[0]["preset"]
    suite, scene = preset.split(".")[0], preset.split(".")[1]

    grader_dir = REPO / "robobench" / "suites" / suite / "grader"
    if not grader_dir.is_dir():
        sys.exit(f"suite '{suite}' has no grader/ — nothing can grade this stage yet")

    if args.solution:
        solution = Path(args.solution).resolve()
        if not args.out:
            sys.exit("--solution needs --out")
    elif args.run:
        solution = exp / "runs" / args.run / "workspace" / "solution"
    else:
        sys.exit("pick a delivery: --run <name> or --solution <dir>")
    if not (solution / "solve.py").is_file():
        sys.exit(f"no solve.py in {solution}")

    out = Path(args.out).resolve() if args.out else exp / "runs" / args.run / "grade"
    if out.exists():
        sys.exit(f"refusing to overwrite existing {out}")

    name = args.run or solution.parent.name
    cname = f"rb_grade_{exp.name}_{name}"
    cmd = [
        "docker", "run", "-d", "--name", cname,
        "--device", f"nvidia.com/gpu={args.gpu}", "--shm-size", "2g",
        "--network", args.network,
        "--user", "agent", "-e", "HOME=/home/agent",
        "-v", f"{stage / 'bench'}:/bench:ro",
        "-v", f"{grader_dir}:/graders:ro",
        "-v", f"{solution}:/solution:ro",
        "-v", f"{EVAL / 'grader'}:/grader:ro",
        "-v", f"{out}:/out",
        "-v", "rb-ovcache:/ovcache",
        DEFAULT_IMAGE, "python", "/grader/grade.py",
        "--preset", preset, "--scene", scene,
    ]

    if args.dry_run:
        import shlex
        print(shlex.join(cmd))
        return

    out.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    subprocess.run(cmd, check=True)
    print(f"grading: container {cname}  (budget {args.budget_min} min, network={args.network})")

    status, exit_code = "completed", None
    try:
        r = subprocess.run(["docker", "wait", cname], capture_output=True, text=True,
                           timeout=args.budget_min * 60, check=True)
        exit_code = int(r.stdout.strip())
    except subprocess.TimeoutExpired:
        status = "timeout"
        print(f"budget reached — stopping {cname}")
        subprocess.run(["docker", "stop", "-t", "30", cname], check=False)

    logs = subprocess.run(["docker", "logs", cname], capture_output=True)
    (out / "container.log").write_bytes(logs.stdout + logs.stderr)
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True, check=False)

    (out / "grade.json").write_text(json.dumps({
        "exp": str(exp), "stage": stage.name, "preset": preset,
        "solution": str(solution), "image": DEFAULT_IMAGE, "gpu": args.gpu,
        "network": args.network, "budget_min": args.budget_min,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status, "container_exit_code": exit_code, "argv": sys.argv,
    }, indent=2) + "\n")

    verdict_file = out / "verdict.json"
    if verdict_file.exists():
        v = json.loads(verdict_file.read_text())
        print(f"verdict: success={v.get('success')} score={v.get('score')}"
              + (f"\n  error: {v['error'].strip().splitlines()[-1]}" if v.get("error") else ""))
    else:
        print(f"{status}: no verdict.json — see {out / 'container.log'}")
    print(f"artifacts in {out}")


if __name__ == "__main__":
    main()
