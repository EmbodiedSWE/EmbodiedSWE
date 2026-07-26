#!/usr/bin/env python3
"""Build one experiment's environment from robobench's registries.

    python eval/scripts/build_env.py --list
    python eval/scripts/build_env.py --name bulb_smoke --stage bulb:franka
    python eval/scripts/build_env.py --name bulb_hard --stage bulb:franka --no-set-states

--stage is scene:robot[:controller]; controller defaults by preference
(osc > impedance > joint), echoed. Single-stage is the primary path; repeated
--stage flags build an ordered multi-stage sequence (e.g. for transfer
experiments), each stage its own bench/task pair.

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

from envbuild import StageSpec, build_experiment, list_envs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list registered envs and exit")
    ap.add_argument("--name", help="experiment name (output: <out>/<name>/)")
    ap.add_argument("--stage", action="append", default=[], metavar="scene:robot[:controller]")
    ap.add_argument("--no-set-states", action="store_true",
                    help="ablation: disable env.set_states() in the extracted tree, so the "
                         "agent cannot restore snapshots and must solve from reset() forward")
    ap.add_argument("--no-freeze-controller", action="store_true",
                    help="allow the agent to switch control modes (default: frozen to the preset)")
    ap.add_argument("--out", default=str(REPO / "experiments"), help="output root")
    args = ap.parse_args()

    if args.list:
        for n in list_envs():
            print(n)
        return
    if not args.name or not args.stage:
        ap.error("--name and at least one --stage are required (or use --list)")

    build_experiment(
        name=args.name,
        stages=[StageSpec.parse(s) for s in args.stage],
        set_states=not args.no_set_states,
        freeze_controller=not args.no_freeze_controller,
        out_root=Path(args.out),
    )


if __name__ == "__main__":
    main()
