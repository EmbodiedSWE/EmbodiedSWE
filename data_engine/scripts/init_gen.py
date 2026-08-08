"""Bake a data_gen campaign alongside an eval run — thin wrapper over
engine/initialization.py (no Isaac boot needed).

    python data_engine/scripts/init_gen.py <run_dir> [--name <gen_name>] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.initialization import init  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="bake a data_gen campaign alongside an eval run")
    ap.add_argument("run_dir", help="the eval run folder (contains run.json, workspace/solution/)")
    ap.add_argument("--name", default=None,
                    help="campaign name under data_gen/ (default: gen_<timestamp>)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing campaign")
    args = ap.parse_args()
    init(args.run_dir, name=args.name, force=args.force)


if __name__ == "__main__":
    main()
