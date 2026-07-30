#!/usr/bin/env python3
"""Grade delivered solutions in fresh grading containers.

    python eval/scripts/run_grade.py experiments/<exp> --run <run>                  # the final solution/
    python eval/scripts/run_grade.py experiments/<exp> --run <run> --submission 03  # one frozen snapshot
    python eval/scripts/run_grade.py experiments/<exp> --run <run> --all            # every UNgraded
                                                                                    #  submission + final,
                                                                                    #  then curve.json
    python eval/scripts/run_grade.py experiments/<exp> --solution DIR --out DIR

Grading never runs on the host: the container gets the same image and the
same read-only /bench tree the agent had, plus the suite's grader/ at
/graders (never in agent-facing trees) and the delivery at /solution. No
credentials; --network none grades fully offline. The in-container driver is
eval/grader/grade.py. Artifacts land in <out>/: verdict.json, progress.jsonl,
container.log, grade.json — every grade of a run also carries the spend
(wall clock + tokens) behind its delivery. One seeded trajectory per grade;
across seeds, name each grade (e.g. --grade final_s1 --seed 1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grader"))
from collect_spend import compute_spend  # noqa: E402 — all agent-transcript knowledge lives there

EVAL = Path(__file__).resolve().parents[1]
REPO = EVAL.parent
DEFAULT_IMAGE = "rb-l1-agent:2.1.216"


def grade_one(*, args, exp: Path, stage: Path, preset: str, scene: str, grader_dir: Path,
              solution: Path, out: Path, gname: str, run: str | None, submission: str | None) -> None:
    """One delivery -> one grading container -> one grade folder."""
    spend = compute_spend(exp / "runs" / run if run else None,
                          solution if submission else None)
    note_file = solution / "note.txt"
    note = note_file.read_text().strip() if note_file.exists() else None
    cname = f"rb_grade_{exp.name}_{run or solution.parent.name}_{gname}"
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
        "--preset", preset, "--scene", scene, "--seed", str(args.seed),
    ] + (["--render"] if args.render else [])

    if args.dry_run:
        import shlex
        print(shlex.join(cmd))
        return

    out.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    subprocess.run(cmd, check=True)
    print(f"grading [{gname}]: container {cname}  (budget {args.budget_min} min, network={args.network})")

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
        "exp": str(exp), "stage": stage.name, "preset": preset, "grade": gname,
        "seed": args.seed,
        **({"submission": submission} if submission else {}),
        **({"note": note} if note else {}),
        **({"spend": spend} if spend else {}),
        "solution": str(solution), "image": DEFAULT_IMAGE, "gpu": args.gpu,
        "network": args.network, "budget_min": args.budget_min,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status, "container_exit_code": exit_code, "argv": sys.argv,
    }, indent=2) + "\n")

    verdict_file = out / "verdict.json"
    if verdict_file.exists():
        v = json.loads(verdict_file.read_text())
        print(f"verdict [{gname}]: success={v.get('success')} score={v.get('score')}"
              + (f"\n  error: {v['error'].strip().splitlines()[-1]}" if v.get("error") else ""))
    else:
        print(f"{status}: no verdict.json — see {out / 'container.log'}")


def write_curve(run_dir: Path) -> None:
    """Roll every grade of the run into runs/<run>/curve.json + a table."""
    rows = []
    for gdir in sorted((run_dir / "grades").iterdir()):
        gj, vj = gdir / "grade.json", gdir / "verdict.json"
        if not (gj.exists() and vj.exists()):
            continue
        g, v = json.loads(gj.read_text()), json.loads(vj.read_text())
        rows.append({"grade": gdir.name, "submission": g.get("submission"),
                     "auto": bool((g.get("spend") or {}).get("auto")),
                     "note": g.get("note"), "seed": g.get("seed"),
                     "success": v.get("success"), "score": v.get("score"),
                     "spend": g.get("spend", {})})
    rows.sort(key=lambda r: (r["grade"] == "final", r["grade"]))
    (run_dir / "curve.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\ncurve -> {run_dir / 'curve.json'}")
    print(f"{'grade':<14}{'origin':<7}{'seed':>5}{'wall_s':>9}{'out_tokens':>12}{'score':>8}  success  note")
    for r in rows:
        u = (r["spend"] or {}).get("usage") or {}
        wall = (r["spend"] or {}).get("wall_s")
        origin = "auto" if r["auto"] else ("agent" if r["submission"] else "-")
        seed = r["seed"] if r["seed"] is not None else "-"
        print(f"{r['grade']:<14}{origin:<7}{seed:>5}{wall if wall is not None else '-':>9}"
              f"{u.get('output_tokens', '-'):>12}{r['score']:>8}  {str(r['success']):<7}  "
              f"{(r['note'] or '')[:48]}")


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
    ap.add_argument("--run", help="grade this run (out: runs/<run>/grades/<grade>)")
    ap.add_argument("--submission", default=None,
                    help="with --run: grade this snapshot under runs/<run>/submissions/ instead of "
                         "solution/; its pinned wall clock + token usage land in grade.json")
    ap.add_argument("--all", action="store_true",
                    help="with --run: grade every UNgraded submission plus the final solution, "
                         "oldest first (already-graded ones are skipped), then write curve.json")
    ap.add_argument("--grade", default=None,
                    help="name for this grade — a run can be graded many times "
                         "(default: the submission name, else timestamped)")
    ap.add_argument("--solution", help="explicit solution dir containing solve.py (needs --out)")
    ap.add_argument("--out", help="output dir (default: <exp>/runs/<run>/grades/<grade>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--network", default="bridge",
                    help="container network (assets are vendored, so --network none also works)")
    ap.add_argument("--budget-min", type=float, default=30, help="wall-clock kill budget PER GRADE (minutes)")
    ap.add_argument("--render", action="store_true",
                    help="render the run: <out>/frames/*.jpg + frames.jsonl + render.json")
    ap.add_argument("--dry-run", action="store_true", help="print the docker command(s) and exit")
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

    common = dict(args=args, exp=exp, stage=stage, preset=preset, scene=scene, grader_dir=grader_dir)

    if args.all:  # sweep: every ungraded submission, then the final solution
        if not args.run:
            sys.exit("--all needs --run")
        run_dir = exp / "runs" / args.run
        subs_dir = run_dir / "submissions"
        subs = sorted(d.name for d in subs_dir.iterdir()
                      if d.is_dir() and not d.name.startswith(".")) if subs_dir.is_dir() else []
        targets = [(s, subs_dir / s, s) for s in subs] + [("final", run_dir / "workspace" / "solution", None)]
        for gname, sol, submission in targets:
            out = run_dir / "grades" / gname
            if out.exists():
                print(f"[{gname}] already graded — skipped")
                continue
            if not (sol / "solve.py").is_file():
                print(f"[{gname}] no solve.py — skipped")
                continue
            grade_one(**common, solution=sol, out=out, gname=gname,
                      run=args.run, submission=submission)
        if not args.dry_run:
            write_curve(run_dir)
        return

    if args.solution:
        solution = Path(args.solution).resolve()
        if not args.out:
            sys.exit("--solution needs --out")
    elif args.run:
        run_dir = exp / "runs" / args.run
        solution = (run_dir / "submissions" / args.submission if args.submission
                    else run_dir / "workspace" / "solution")
    else:
        sys.exit("pick a delivery: --run <name> (+ --submission/--all) or --solution <dir>")
    if not (solution / "solve.py").is_file():
        sys.exit(f"no solve.py in {solution}")

    gname = args.grade or args.submission or f"g_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = Path(args.out).resolve() if args.out else exp / "runs" / args.run / "grades" / gname
    if out.exists():
        sys.exit(f"refusing to overwrite existing {out}")

    grade_one(**common, solution=solution, out=out, gname=gname,
              run=args.run, submission=args.submission)
    print(f"artifacts in {out}")


if __name__ == "__main__":
    main()
