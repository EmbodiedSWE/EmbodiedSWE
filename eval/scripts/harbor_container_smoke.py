#!/usr/bin/env python3
"""Boot every exported world in real Harbor Docker, without native grading.

Requires evaluator-provided GPU/runtime images. No model or grader credentials
are needed. This checks simulator boot/reset and separate-container lifecycle;
it does not establish task-solving performance or native score parity.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from harbor_support.files import write_json
from harbor_support.report import summarize
from harbor_support.validate import validate_dataset


def check_job(dataset, job):
    return check_jobs(dataset, [job])


def check_jobs(dataset, jobs):
    """Accept partitioned runs only when their union covers the exact export."""
    jobs = list(jobs)
    summary = summarize(dataset, jobs)
    if any(row["score"] != 0 or row["success"] != 0 for row in summary["per_task"].values()):
        raise ValueError("An empty delivery unexpectedly received a nonzero reward")
    for path in sorted(path for job in jobs for path in job.glob("*/result.json")):
        trial = json.loads(path.read_text())
        agent = trial.get("agent_info") or {}
        if (agent.get("name") != "robobench-container-boot" or agent.get("version") != "1"
                or trial["config"]["agent"].get("import_path")
                != "harbor_support.boot_agent:ContainerBootAgent"):
            raise ValueError("Container boot evidence must come from the declared boot fixture")
        # A timed-out solver can still have a valid benchmark score. This fixture
        # must actually finish its GPU check, so even AgentTimeoutError is fatal.
        if trial.get("exception_info"):
            raise ValueError("Container boot trial did not finish successfully")
        metadata = (trial.get("agent_result") or {}).get("metadata")
        if metadata != {"gpu_build_reset": "passed", "delivery": "empty"}:
            raise ValueError("Missing completed GPU boot evidence")
        name = trial["task_name"].removeprefix("robobench/")
        protocol = json.loads((dataset / name / "tests/protocol.json").read_text())
        lines = (path.parent / "agent/gpu-boot.log").read_text().splitlines()
        if "HARBOR_GPU_BOOT_OK " + protocol["preset"] not in lines:
            raise ValueError("Missing GPU boot completion marker for the exported preset")
        report = json.loads((path.parent / "verifier/results.json").read_text())
        deliveries = report.get("submissions", [])
        if (len(deliveries) != 1 or deliveries[0].get("submission") != "final"
                or deliveries[0].get("status") != "missing_solution"
                or deliveries[0].get("rollouts") != []):
            raise ValueError("Container boot fixture must leave exactly one empty final delivery")
    return {"schema_version": 1, "status": "passed", "summary": summary,
            "scope": "GPU build/reset in agent containers and empty separate verification; no native grading."}


async def run(args):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig, RetryConfig
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

    dataset = args.dataset.resolve()
    validate_dataset(dataset)
    names = [row["task"] for row in json.loads((dataset / "dataset.json").read_text())["tasks"]]
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Use a fresh smoke-test output directory")
    if args.job:
        result = check_jobs(dataset, [path.resolve() for path in args.job])
        output.mkdir(parents=True)
        write_json(output / "container-smoke.json", result)
        print(json.dumps({"status": result["status"], "tasks": result["summary"]["task_count"], "output": str(output)}))
        return
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=30)
    output.mkdir(parents=True)
    job = await Job.create(JobConfig(
        job_name="container-boots", jobs_dir=output, n_concurrent_trials=1,
        retry=RetryConfig(max_retries=0), environment=EnvironmentConfig(type="docker", delete=True),
        agents=[AgentConfig(import_path="harbor_support.boot_agent:ContainerBootAgent")],
        tasks=[TaskConfig(path=dataset / name) for name in sorted(names)],
    ))
    await job.run()
    result = check_job(dataset, job.job_dir)
    write_json(output / "container-smoke.json", result)
    print(json.dumps({"status": result["status"], "tasks": result["summary"]["task_count"], "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job", action="append", type=Path,
                        help="Validate existing jobs instead of running Docker; repeat for partitioned runs")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Harbor container smoke failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
