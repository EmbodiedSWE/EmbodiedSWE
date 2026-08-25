#!/usr/bin/env python3
"""Plots from the RELIABLE replay grades (results/<label>/replay_scores.json), not the grep.

Each run's replay_scores.json carries, per delivered submission, the real peak score from a
fresh-reset rollout under the scene's own scorer, plus the submission's wall_s. The
best-state-so-far at time t is the cumulative max over submissions with wall_s <= t; the run's
final best state is `peak_score`. This module renders, using only that:

  * score-of-best-state vs wall-clock (tools vs no-tools; per task and mean), and
  * peak-score bar charts per task,

for the arms we care about. Runs still ungraded (no replay_scores.json) are simply absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)
TASKS = ["allen_bolt", "bulb", "coffee", "ikea_table", "nut_thread", "pc_gpu", "pc_gpu_ram",
         "pc_motherboard", "pc_ram", "pen_holder", "tool_packing", "spatula", "syringe"]
TIME_GRID = np.arange(0, 250, 2.0)


def load(label: str) -> dict | None:
    f = RESULTS / label / "replay_scores.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:  # noqa: BLE001
        return None


def peak(label: str) -> float | None:
    d = load(label)
    return None if d is None else d.get("peak_score")


def time_curve(label: str) -> np.ndarray | None:
    """Best-state score vs wall-clock minutes: cumulative max replay score over submissions
    delivered by each minute. `final` (wall_s=None) is pinned at the run's last known wall."""
    d = load(label)
    if d is None:
        return None
    pts = []
    walls = [g["wall_s"] for g in d["grades"] if g.get("wall_s") is not None]
    last_wall = max(walls) if walls else 0.0
    for g in d["grades"]:
        w = g["wall_s"] if g.get("wall_s") is not None else last_wall
        if g.get("score") is not None:
            pts.append((w / 60.0, g["score"]))
    if not pts:
        return None
    pts.sort()
    vals = np.zeros(len(TIME_GRID))
    for i, t in enumerate(TIME_GRID):
        s = 0.0
        for m, sc in pts:
            if m <= t:
                s = max(s, sc)
        vals[i] = s
    return vals


def mean_time_curve(labels):
    cs = [c for c in (time_curve(l) for l in labels) if c is not None]
    return (np.mean(cs, axis=0), len(cs)) if cs else (None, 0)


def base_label(model: str, task: str) -> str:
    special = {("opus_5", "allen_bolt"): "allen_bolt_opus_5_r2",
               ("gpt_5_6_sol", "allen_bolt"): "allen_bolt_gpt_5_6_sol_r3",
               ("gpt_5_6_sol", "pc_motherboard"): "pc_motherboard_gpt_5_6_sol_r3"}
    if (model, task) in special:
        return special[(model, task)]
    return f"{task}_opus_5_r2" if model == "opus_5" else f"{task}_{model}"


# ---- tools vs no-tools, per model: peak-score bars + mean score-vs-time --------------------
def tools_plots():
    for model, pretty in (("opus_5", "Opus 5"), ("gpt_5_6_sol", "GPT-5.6-Sol"),
                          ("fable_5", "Fable 5")):
        base = [base_label(model, t) for t in TASKS]
        tool = [f"{t}_{model}_rtools1" for t in TASKS]
        bp = [peak(l) for l in base]
        tp = [peak(l) for l in tool]
        if not any(v is not None for v in bp + tp):
            continue
        x = np.arange(len(TASKS))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - 0.2, [v or 0 for v in bp], 0.4, label=f"{pretty} no-tools", color="#5A5A5A")
        ax.bar(x + 0.2, [v or 0 for v in tp], 0.4, label=f"{pretty} tools", color="#0173B2")
        for i, v in enumerate(bp):
            if v is None:
                ax.text(i - 0.2, 0.02, "n/a", ha="center", fontsize=6, rotation=90)
        for i, v in enumerate(tp):
            if v is None:
                ax.text(i + 0.2, 0.02, "n/a", ha="center", fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS, rotation=35, ha="right")
        ax.set_ylabel("peak best-state score (real replay)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Tools vs no-tools ({pretty}) — reliable replay grades")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f"replay_tools_{model}_peak.png", dpi=140)
        plt.close(fig)
        print(f"wrote {PLOTS / f'replay_tools_{model}_peak.png'}")
        # mean score-vs-time
        bm, nb = mean_time_curve(base)
        tm, nt = mean_time_curve(tool)
        fig, ax = plt.subplots(figsize=(8, 5))
        if bm is not None:
            ax.plot(TIME_GRID, bm, label=f"{pretty} no-tools (n={nb})", color="#5A5A5A", lw=2)
        if tm is not None:
            ax.plot(TIME_GRID, tm, label=f"{pretty} tools (n={nt})", color="#0173B2", lw=2)
        ax.set_xlabel("agent wall-clock minutes")
        ax.set_ylabel("mean best-state score (real replay)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Tools vs no-tools over time ({pretty}) — reliable replay grades")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / f"replay_tools_{model}_time.png", dpi=140)
        plt.close(fig)
        print(f"wrote {PLOTS / f'replay_tools_{model}_time.png'}")


def coverage():
    got = [l.parent.name for l in RESULTS.glob("*/replay_scores.json")]
    print(f"\nreplay_scores present for {len(got)} runs")


if __name__ == "__main__":
    tools_plots()
    coverage()
    print("REPLAY PLOTS DONE")
