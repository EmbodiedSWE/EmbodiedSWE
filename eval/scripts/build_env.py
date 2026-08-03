#!/usr/bin/env python3
"""Build one experiment's environment from robobench's registries.

    python eval/scripts/build_env.py --list
    python eval/scripts/build_env.py --name bulb_smoke --stage bulb:franka
    python eval/scripts/build_env.py --name crate --stage packing.crate:franka:osc
    python eval/scripts/build_env.py --name bulb_hard --stage bulb:franka --config no_checkpoint

--stage is [suite.]scene:robot[:controller]; the suite defaults to assembly and
controller defaults by preference
(osc > impedance > joint), echoed. Single-stage is the primary path; repeated
--stage flags build an ordered multi-stage sequence (e.g. for transfer
experiments), each stage its own bench/task pair.

--config names the experimental CONDITION (a file in eval/configs/, by name or path). Which
code features are blocked comes from its `features:`, so the build and the prompt can no longer
disagree — a world built with `no_checkpoint` is patched to match the rule text it will ship
with. There are no per-ablation build flags.

Re-execs itself under the repo venv (robobench + its deps live there).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PY = REPO / ".venv" / "bin" / "python"

if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])

sys.path.insert(0, str(REPO / "eval"))

from envbuild import StageSpec, build_experiment, list_envs, load_condition  # noqa: E402
from envbuild.condition import list_rules, list_skills, list_tools  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list registered envs and exit")
    ap.add_argument("--list-condition", action="store_true",
                    help="list the rule / skill / tool libraries a condition can select from")
    ap.add_argument("--name", help="experiment name (output: <out>/<name>/)")
    ap.add_argument("--stage", action="append", default=[], metavar="[suite.]scene:robot[:controller]")
    ap.add_argument("--config", default="default",
                    help="experimental condition: a name in eval/configs/ or a path to a yaml "
                         "(default: %(default)s). Its features: decide which code patches apply")
    ap.add_argument("--seed", type=int, default=0, help="seed for the boot check")
    ap.add_argument("--out", default=str(REPO / "experiments"), help="output root")
    args = ap.parse_args()

    if args.list:
        for n in list_envs():
            print(n)
        return
    if args.list_condition:
        for label, names in (("rules", list_rules()), ("skills", list_skills()), ("tools", list_tools())):
            print(f"{label}: {', '.join(names) or '(empty)'}")
        return
    if not args.name or not args.stage:
        ap.error("--name and at least one --stage are required (or use --list)")

    condition = load_condition(args.config)
    print(f"=== condition {condition.path.name}: rules={list(condition.rules)} "
          f"skills={list(condition.skills)} tools={list(condition.tools)} "
          f"features={condition.features or '{}'} ===")
    build_experiment(
        name=args.name,
        stages=[StageSpec.parse(s) for s in args.stage],
        condition=condition,
        out_root=Path(args.out),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
