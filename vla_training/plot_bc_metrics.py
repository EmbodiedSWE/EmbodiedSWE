#!/usr/bin/env python3
"""Plot BC training metrics (loss / grad_norm / lr) from a training job's log.

Pulls the worker-0 stderr of the given merlin job, parses the trainer's
"step=N loss=... lr=... grad_norm=..." lines, writes a CSV and a PNG under
simgen_bc/plots/ (gitignored -- artifacts, not code).

  python3 CoSiGen/vla_training/plot_bc_metrics.py --job f7475af4a5619bcc \
      --tag bc_v1_h20x64
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import urllib.request
from pathlib import Path

CAPX = Path(__file__).resolve().parents[2]
PLOTS = CAPX / "simgen_bc" / "plots"


def fetch_points(job: str) -> list[tuple[int, float, float, float]]:
    run = lambda a: json.loads(subprocess.check_output(a, timeout=200).decode())  # noqa: E731
    d = run(["merlin-cli", "job", "get-run", "--control-plane", "cn-seed",
             "--json", json.dumps({"job_run_id": job})])
    raw = json.dumps(d)
    trial = re.search(r'arnold_trial_id\\?": *\\?"(\d+)', raw).group(1)
    logs = run(["merlin-cli", "job", "list-trial-logs", "--control-plane", "cn-seed",
                "--json", json.dumps({"job_run_id": job, "trial_id": trial})]
               ).get("log_list", [])
    for l in logs:
        if l.get("type") == "stderr" and "worker-0" in l.get("pod_name", ""):
            data = urllib.request.urlopen(l["url"], timeout=180).read().decode("utf-8", "replace")
            pts = re.findall(r"\[I\] step=(\d+) loss=([0-9.]+) lr=([0-9.e-]+) "
                             r"grad_norm=([0-9.]+)", data)
            return [(int(s), float(lo), float(lr), float(g)) for s, lo, lr, g in pts]
    raise SystemExit("no worker-0 stderr found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="f7475af4a5619bcc")
    ap.add_argument("--tag", default="bc_v1_h20x64")
    args = ap.parse_args()

    pts = fetch_points(args.job)
    PLOTS.mkdir(parents=True, exist_ok=True)
    csv_path = PLOTS / f"{args.tag}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "lr", "grad_norm"])
        w.writerows(pts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [p[0] for p in pts]
    loss = [p[1] for p in pts]
    lr = [p[2] for p in pts]
    gn = [p[3] for p in pts]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(steps, loss, lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss (log)")
    axes[0].set_title(f"{args.tag}  (job {args.job})  --  {len(pts)} points, "
                      f"latest step {steps[-1]}")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, gn, lw=1.2, color="tab:orange")
    axes[1].set_ylabel("grad_norm")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, lr, lw=1.2, color="tab:green")
    axes[2].set_ylabel("lr")
    axes[2].set_xlabel("step")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    png = PLOTS / f"{args.tag}_metrics.png"
    fig.savefig(png, dpi=110)
    print(f"wrote {csv_path}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
