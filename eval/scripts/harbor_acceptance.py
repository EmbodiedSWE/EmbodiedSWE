#!/usr/bin/env python3
"""Run reference deliveries through Harbor Docker and compare native verdicts.

A passing run validates only the supplied export/reference matrix. Full release
acceptance additionally needs the coverage described in VALIDATION.md.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from harbor_support.files import content_hash, inventory, write_json
from harbor_support.reference import compare_verdicts, load_references
from harbor_support.report import summarize
from harbor_support.validate import validate_dataset


def check_references(dataset, references):
    validate_dataset(dataset)
    names = {row["task"] for row in json.loads((dataset / "dataset.json").read_text())["tasks"]}
    if set(references["tasks"]) != names:
        raise ValueError("References must cover exactly the exported task roster")
    for name, case in references["tasks"].items():
        protocol = json.loads((dataset / name / "tests/protocol.json").read_text())
        if len(case["checkpoints"]) > protocol["max_submissions"]:
            raise ValueError(f"{name}: too many reference checkpoints")
        for record in [*case["checkpoints"], case["final"]]:
            if set(record["native"]) != {str(seed) for seed in protocol["seeds"]}:
                raise ValueError(f"{name}: native references must cover exactly the exported seeds")
            for seed in protocol["seeds"]:
                verdict = json.loads(Path(record["native"][str(seed)]["path"]).read_text())
                compare_verdicts(verdict, verdict, protocol, seed)


def check_job(dataset, job, references, score_tolerance=0.0):
    summary = summarize(dataset, [job])
    comparisons = []
    for path in sorted(job.glob("*/result.json")):
        trial = json.loads(path.read_text())
        name = trial["task_name"].removeprefix("robobench/")
        case = references["tasks"][name]
        verifier = path.parent / "verifier"
        report = json.loads((verifier / "results.json").read_text())
        records = [(f"{i + 1:06d}", r) for i, r in enumerate(case["checkpoints"])]
        records.append(("final", case["final"]))
        if [x["submission"] for x in report["submissions"]] != [n for n, _ in records]:
            raise ValueError(f"{name}: checkpoint/final delivery order or count differs")
        for entry, (submission, record) in zip(report["submissions"], records, strict=True):
            if entry.get("sha256") != record["sha256"]:
                raise ValueError(f"{name}/{submission}: transferred reference checksum differs")
            if content_hash(inventory(verifier / "inputs" / submission)) != record["sha256"]:
                raise ValueError(f"{name}/{submission}: frozen input checksum differs")
            if entry["status"] not in ("graded", "duplicate"):
                raise ValueError(f"{name}/{submission}: reference was not graded")
            graded = entry.get("duplicate_of", submission)
            for seed in report["protocol"]["seeds"]:
                rollout = verifier / "rollouts" / graded / f"seed_{seed}"
                actual = json.loads((rollout / "verdict.json").read_text())
                expected = json.loads(Path(record["native"][str(seed)]["path"]).read_text())
                comparison = compare_verdicts(actual, expected, report["protocol"], seed,
                                              score_tolerance=score_tolerance)
                if report["protocol"]["cameras"]:
                    for index in range(report["protocol"]["num_envs"]):
                        frame = rollout / f"traj_{index:03d}" / "final.png"
                        with frame.open("rb") as stream:
                            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                                raise ValueError(f"{name}: missing or invalid final-frame PNG")
                comparisons.append({"task": name, "submission": submission, **comparison})
    return {"schema_version": 1, "status": "passed_supplied_matrix", "summary": summary,
            "comparisons": comparisons,
            "scope": "Reference delivery, separate verification, and native per-trajectory parity; "
                     "not an adversarial-security or full-release certificate."}


async def run(args):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    dataset = args.dataset.resolve()
    references = load_references(args.references.resolve())
    check_references(dataset, references)
    output = args.output.absolute()
    if output.exists():
        raise ValueError("Use a fresh acceptance output directory")
    if any(json.loads((dataset / name / "tests/protocol.json").read_text())["cameras"]
           for name in references["tasks"]) and not os.environ.get("VLM_JUDGE_API_KEY"):
        raise ValueError("Live Newton acceptance requires VLM_JUDGE_API_KEY")
    # Fail before launching expensive builds if the configured daemon is unavailable.
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=30)
    output.mkdir(parents=True)
    write_json(output / "references.json", references)
    job = await Job.create(JobConfig(
        job_name="reference-replay", jobs_dir=output, n_concurrent_trials=1,
        retry=RetryConfig(max_retries=0), environment=EnvironmentConfig(type="docker", delete=True),
        agents=[AgentConfig(import_path="harbor_support.replay_agent:ReferenceReplayAgent",
                            kwargs={"manifest": str(output / "references.json")})],
        tasks=[TaskConfig(path=dataset / name) for name in sorted(references["tasks"])],
    ))
    await job.run()
    result = check_job(dataset, job.job_dir, references, args.score_tolerance)
    write_json(output / "acceptance.json", result)
    print(json.dumps({"status": result["status"], "tasks": result["summary"]["task_count"],
                      "comparisons": len(result["comparisons"]), "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-tolerance", type=float, default=0.0,
                        help="explicit per-trajectory score tolerance; default exact, maximum 0.01")
    args = parser.parse_args()
    try:
        # Validate even if execution fails before any comparison is reached.
        if not 0 <= args.score_tolerance <= 0.01:
            raise ValueError("Score tolerance must be in [0, 0.01]")
        asyncio.run(run(args))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Harbor acceptance failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
