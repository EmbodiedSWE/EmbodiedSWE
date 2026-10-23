#!/usr/bin/env python3
"""Build RoboBench worlds and export the 28-task benchmark for Harbor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO))

from harbor_support.catalog import select_tasks, validate_registry
from harbor_support.export import DEFAULT_CONDITION, Settings, export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list the fixed task/preset roster without Isaac")
    listing.add_argument("--validate-registry", action="store_true")
    validation = commands.add_parser("validate", help="check exported content hashes and Harbor definitions")
    validation.add_argument("dataset", type=Path)
    summary = commands.add_parser("summarize", help="reject failed/incomplete jobs before computing metrics")
    summary.add_argument("--dataset", type=Path, required=True)
    summary.add_argument("--job", type=Path, action="append", required=True)
    summary.add_argument("--attempts", type=int, default=1, help="expected trials per task")
    build = commands.add_parser("build", help="extract and boot-check worlds on a GPU host")
    export = commands.add_parser("export", help="export already boot-validated worlds")
    for command in (build, export):
        command.add_argument("--task", action="append", help="task ID; repeat to export a subset")
        command.add_argument("--worlds", type=Path, required=True)
        command.add_argument("--condition", type=Path, default=DEFAULT_CONDITION)
    build.add_argument("--physx-python", type=Path)
    build.add_argument("--newton-python", type=Path)
    build.add_argument("--seed", type=int, default=0)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--physx-image")
    export.add_argument("--newton-image")
    export.add_argument("--allow-mutable-images", action="store_true")
    export.add_argument("--agent-timeout-sec", type=float, default=14400)
    export.add_argument("--rollout-timeout-sec", type=float, default=1800)
    export.add_argument("--max-submissions", type=int, default=64)
    export.add_argument("--seeds", type=int, nargs="+", default=[0])
    export.add_argument("--num-envs", type=int, default=1)
    export.add_argument("--cpus", type=int, default=8)
    export.add_argument("--memory-mb", type=int, default=32768)
    args = parser.parse_args()
    try:
        if args.command == "summarize":
            from harbor_support.report import summarize
            print(json.dumps(summarize(args.dataset.resolve(), args.job, args.attempts), indent=2))
            return
        if args.command == "validate":
            from harbor_support.validate import validate_dataset
            count = validate_dataset(args.dataset.resolve())
            print(f"Validated {count} Harbor tasks and their content receipts")
            return
        if args.command == "list":
            tasks = select_tasks()
            if args.validate_registry:
                validate_registry(tasks)
            for task in tasks:
                print(f"{task.id:32s} {task.backend:7s} {task.preset}")
            print(f"{len(tasks)} tasks")
            return
        tasks = select_tasks(args.task)
        if args.command == "build":
            if not 0 <= args.seed < 2**32:
                raise ValueError("Seed must be an unsigned 32-bit integer")
            interpreters = {"physx": args.physx_python, "newton": args.newton_python}
            # Validate the whole request before beginning expensive builds.
            for task in tasks:
                python = interpreters[task.backend]
                if python is None or not python.is_file():
                    raise ValueError(f"Provide --{task.backend}-python for {task.id}")
                if (args.worlds / task.id).exists():
                    raise FileExistsError(f"World already exists: {args.worlds / task.id}; select remaining tasks")
            for task in tasks:
                subprocess.run([
                    str(interpreters[task.backend].absolute()), "-m", "harbor_support.prepare",
                    "--task", task.id, "--out", str(args.worlds.absolute()),
                    "--condition", str(args.condition.absolute()), "--seed", str(args.seed),
                ], check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "eval") + os.pathsep + str(REPO)})
        else:
            from envbuild.condition import load
            settings = Settings(**{name: getattr(args, name) for name in (
                "agent_timeout_sec", "rollout_timeout_sec", "max_submissions", "num_envs",
                "cpus", "memory_mb", "allow_mutable_images")}, seeds=tuple(args.seeds))
            path = export_dataset(worlds=args.worlds, output=args.output, tasks=tasks,
                                  images={k: v for k, v in {"physx": args.physx_image,
                                          "newton": args.newton_image}.items() if v},
                                  condition=load(args.condition), settings=settings)
            print(f"Exported {len(tasks)} tasks to {path}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"harbor: {exc}\n")


if __name__ == "__main__":
    main()
