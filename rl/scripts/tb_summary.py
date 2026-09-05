"""Print the scalars of a training run (tensorboard events) as a compact table.

    python rl/scripts/tb_summary.py rl/runs/slice_franka_joint/shaped/<stamp> [--every 5] [--tags Train/mean_reward,...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--every", type=int, default=0, help="print every N-th iteration (0 = first/last + quartiles)")
ap.add_argument("--tags", default="", help="comma list of tags to show (default: a useful subset)")
args = ap.parse_args()

acc = EventAccumulator(str(Path(args.run_dir)), size_guidance={"scalars": 0})
acc.Reload()
tags = sorted(t for t in acc.Tags()["scalars"] if not t.endswith("/time"))  # rsl_rl's wall-clock-indexed twins
print(f"{len(tags)} scalar tags:", ", ".join(tags))
want = [t for t in args.tags.split(",") if t] or [t for t in tags if any(
    k in t for k in ("mean_reward", "mean_episode_length", "episode/", "stage/", "value_function", "surrogate",
                     "learning_rate", "mean_noise_std", "fps", "collection_time", "learn_time"))]
series = {t: acc.Scalars(t) for t in want if t in tags}
if not series:
    raise SystemExit("no matching tags")
steps = sorted({s.step for v in series.values() for s in v})
if args.every:
    rows = [s for s in steps if s % args.every == 0 or s == steps[-1]]
else:
    q = [0, len(steps) // 4, len(steps) // 2, 3 * len(steps) // 4, len(steps) - 1]
    rows = sorted({steps[i] for i in q})
names = list(series)
def short(n: str) -> str:
    parts = n.split("/")
    return ("/".join(parts[-2:]) if parts[0] in ("Episode", "Train", "Perf", "Loss", "Policy") and len(parts) > 2 else parts[-1])[-16:]
print("iter  " + "  ".join(short(n).rjust(16) for n in names))
for st in rows:
    vals = []
    for n in names:
        m = {s.step: s.value for s in series[n]}
        vals.append(f"{m[st]:16.4g}" if st in m else " " * 16)
    print(f"{st:5d} " + "  ".join(vals))
