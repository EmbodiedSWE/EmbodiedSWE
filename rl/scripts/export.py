"""Export a training checkpoint as a graded-able solution/ folder (no Isaac needed).

    python rl/scripts/export.py rl/runs/slice_franka_joint/shaped/<stamp> [--checkpoint model_29.pt] [--out DIR]
Default out: <run>/solutions/<checkpoint stem>/ . Then grade it like any agent solution, e.g.
    python eval/scripts/verify_solution.py --preset <preset> --solution <out>      (host, scene success only)
    python eval/scripts/run_grade.py <exp> --solution <out> --out <dir>            (container, full rubric)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robobench_rl.export import export_solution  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--checkpoint", default="", help="model_<it>.pt inside the run dir (default: highest iteration)")
ap.add_argument("--out", default="")
args = ap.parse_args()

run = Path(args.run_dir)
if args.checkpoint:
    ckpt = run / args.checkpoint
else:
    cands = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not cands:
        raise SystemExit(f"no model_*.pt in {run}")
    ckpt = cands[-1]
out = Path(args.out) if args.out else run / "solutions" / ckpt.stem
export_solution(run, ckpt, out)
print(f"exported {ckpt.name} -> {out}")
for p in sorted(out.iterdir()):
    print(f"  {p.name:18s} {p.stat().st_size:>9,d} B")
