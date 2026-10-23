"""Build one catalog world with the explicitly selected simulator interpreter."""

import argparse
import sys
from pathlib import Path

from envbuild import StageSpec, build_experiment, load_condition
from .catalog import select_tasks, validate_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    task, = select_tasks([args.task])
    validate_registry([task])
    build_experiment(name=task.id, stages=[StageSpec.parse(task.stage_arg)],
                     condition=load_condition(args.condition), out_root=args.out,
                     seed=args.seed, python_executable=Path(sys.executable))


if __name__ == "__main__":
    main()
