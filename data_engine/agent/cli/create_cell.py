"""create_cell — start a new cell from the session's start point.

Inside a diversify container this is on PATH and the session context comes from
the environment (DGEN_ROOT/LEVEL/SCENE/STRATEGY, set by the launcher from the
condition yaml), so:

    create_cell               one new cell of the session's level
    create_cell --count 3     three at once: scene_1, scene_2, scene_3 …

What is created depends on the level: scene = a full copy of the start scene
(assets links intact, fresh metas); strategy = an empty cell for a new
solve.py; phase = phases/phase_N/ stubs — one PROPOSAL of how to divide/enter
the strategy's solve (its own solve_by_phase.py lives inside). Flags override
the environment (e.g. --scene scene_1 to branch from a cell you just made).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.cells import create_cell  # noqa: E402


def main() -> None:
    env = os.environ.get
    ap = argparse.ArgumentParser(description="start new cell(s) from the session's start point")
    ap.add_argument("gen_root", nargs="?", default=env("DGEN_ROOT"),
                    help="the campaign (default: $DGEN_ROOT)")
    ap.add_argument("--level", default=env("DGEN_LEVEL"), choices=("scene", "strategy", "phase"))
    ap.add_argument("--scene", default=env("DGEN_SCENE", "scene_0"), help="start scene (copy source)")
    ap.add_argument("--strategy", default=env("DGEN_STRATEGY", "strategy_0"), help="start strategy (phase level)")
    ap.add_argument("--name", default=None, help="new folder name (default: next free index)")
    ap.add_argument("--count", type=int, default=1, help="how many cells to create")
    args = ap.parse_args()

    if not args.gen_root or not args.level:
        ap.error("need gen_root and --level (or $DGEN_ROOT/$DGEN_LEVEL from the session)")
    if args.level not in ("scene", "strategy", "phase"):
        # argparse does not validate a DEFAULT against `choices`: a bad $DGEN_LEVEL once fell
        # through to the phase branch and wrote phase stubs into the read-only start point
        ap.error(f"--level must be scene|strategy|phase, got {args.level!r}")
    gen_root = Path(args.gen_root).resolve()
    if not (gen_root / "gen.yaml").is_file():
        raise SystemExit(f"not a campaign (no gen.yaml): {gen_root}")
    if args.count > 1 and args.name:
        raise SystemExit("--name only makes sense with --count 1")

    for _ in range(args.count):
        target, base, name = create_cell(gen_root, args.level, args.scene, args.strategy, args.name)
        print(f"created {args.level} cell: {target.name}")


if __name__ == "__main__":
    main()
