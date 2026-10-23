#!/usr/bin/env python3
"""Replay smoke programs against an exported dataset on an evaluator's GPU host.

This tests native simulation/grading and checkpoint selection directly. It does
not test Harbor's Docker lifecycle and is not a benchmark performance result.
"""

import argparse
import json
import os
import pwd
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from harbor_support.catalog import select_tasks
from harbor_support.files import write_json
from harbor_support.submit import snapshot
from harbor_support.validate import validate_dataset
from harbor_support.verifier import verify


NOOP = "def solve(env):\n    pass\n"
ONE_STEP = """def solve(env):
    import torch
    env.step(torch.zeros((env.num_envs, env.robot.action_dim), device=env.device))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physx-python", type=Path)
    parser.add_argument("--newton-python", type=Path)
    parser.add_argument("--task", action="append", help="task ID; repeat to test a declared subset")
    args = parser.parse_args()
    tasks = select_tasks(args.task)
    dataset, output = args.dataset.resolve(), args.output.absolute()
    interpreters = {"physx": args.physx_python, "newton": args.newton_python}
    try:
        validate_dataset(dataset)
        if output.exists() or output.is_symlink():
            raise ValueError(f"Use a fresh output directory: {output}")
        # Validate every prerequisite before consuming simulator time.
        for task in tasks:
            python = interpreters[task.backend]
            if python is None or not python.is_file():
                raise ValueError(f"Provide --{task.backend}-python for {task.id}")
            config = json.loads((dataset / task.id / "tests/protocol.json").read_text())
            if config["max_submissions"] < 2:
                raise ValueError("Smoke tests need an export with max_submissions >= 2")
            if config["worker_user"]:
                worker = pwd.getpwnam(config["worker_user"])
                if os.geteuid() != 0 or worker.pw_uid == 0:
                    raise ValueError("Run as root with the export's non-root worker account present")
            if config["cameras"] and not os.environ.get("VLM_JUDGE_API_KEY"):
                raise ValueError("Live Newton grading requires VLM_JUDGE_API_KEY")
        output.mkdir(parents=True)
        output.chmod(0o755)
        results = []
        for task in tasks:
            started = time.monotonic()
            context = dataset / task.id / "tests"
            config = json.loads((context / "protocol.json").read_text())
            root = output / task.id
            # Reproduce the Dockerfile's writable cache without modifying the
            # receipt-covered export. This mirror is removed after each task.
            bench = root / "bench"
            shutil.copytree(context / "bench", bench, ignore=shutil.ignore_patterns(".cache"))
            cache = bench / ".cache/robobench_bundles"
            cache.mkdir(parents=True)
            cache.chmod(0o777)
            solution = root / "solution"
            solution.mkdir(parents=True)
            (solution / "solve.py").write_text(NOOP)
            snapshot(solution, root / "submissions", note="no-op baseline", limit=2)
            (solution / "solve.py").write_text(ONE_STEP)
            snapshot(solution, root / "submissions", note="one robot action", limit=2)
            os.environ.update(
                PYTHONPATH=str(bench), COSIGEN_ASSET_DIR=str(bench),
                HF_HUB_OFFLINE="1", OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y",
            )
            result = {"task": task.id}
            try:
                report = verify(config, submissions=root / "submissions", final=solution,
                                out=root / "verifier", driver=context / "grade.py",
                                graders=context / "graders",
                                python=str(interpreters[task.backend].absolute()))
                statuses = [entry["status"] for entry in report["submissions"]]
                if statuses != ["graded", "graded", "duplicate"]:
                    raise RuntimeError(f"Unexpected checkpoint statuses: {statuses}")
                result.update(status="passed", reward=report["reward"],
                              selected_submission=report["selected_submission"])
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                shutil.rmtree(bench)
            result["wall_sec"] = round(time.monotonic() - started, 3)
            results.append(result)
            write_json(output / "smoke.json", {"results": results})
            print(json.dumps(result), flush=True)
        if any(result["status"] != "passed" for result in results):
            raise SystemExit(1)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"harbor GPU smoke: {exc}\n")


if __name__ == "__main__":
    main()
