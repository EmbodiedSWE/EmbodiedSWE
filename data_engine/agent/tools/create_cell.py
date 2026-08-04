"""Scaffold one new cell in a data_gen campaign — the agent-side tool.

    python data_engine/agent/tools/create_cell.py <gen_root> --level scene [--scene scene_0] [--name scene_2]
    python data_engine/agent/tools/create_cell.py <gen_root> --level strategy [--scene scene_0] [--name strategy_1]
    python data_engine/agent/tools/create_cell.py <gen_root> --level phase --scene scene_0 --strategy strategy_0 [--name phase_1]

One cell per proposed variant: run this once for EACH distinct idea and author
them side by side — a scene copy to edit, an empty strategy for a new solve.py,
or a stubbed phase (phase.yaml + reset.py). Prints the created cell's path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.cells import create_cell  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="scaffold one new cell (one per proposed variant)")
    ap.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
    ap.add_argument("--level", required=True, choices=("scene", "strategy", "phase"))
    ap.add_argument("--scene", default="scene_0", help="context scene (and the copy source for --level scene)")
    ap.add_argument("--strategy", default="strategy_0", help="context strategy (phase level)")
    ap.add_argument("--name", default=None, help="new folder name (default: next free index)")
    args = ap.parse_args()

    gen_root = Path(args.gen_root).resolve()
    if not (gen_root / "gen.yaml").is_file():
        raise SystemExit(f"not a campaign (no gen.yaml): {gen_root}")
    target, base, name = create_cell(gen_root, args.level, args.scene, args.strategy, args.name)
    print(f"created {args.level} cell: {target}" + (f" (phases/{name})" if args.level == "phase" else ""))


if __name__ == "__main__":
    main()
