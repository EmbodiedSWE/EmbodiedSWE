#!/usr/bin/env python3
"""Analysis plots over the persisted milestones (results/milestones_index.json).

Curves are monotone step functions score(x) per run (x = minutes or uncached+output
tokens), carried forward past run end; group curves are the mean over the group's runs.

    python eval/grader/plot_curves.py          # writes results/plots/*.png
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

INDEX = {r["label"]: r for r in json.loads((RESULTS / "milestones_index.json").read_text())}

PHYSX_TASKS = ["allen_bolt", "bulb", "ikea_table", "nut_thread", "pc_gpu", "pc_gpu_ram",
               "pc_motherboard", "pc_ram", "pen_holder", "tool_packing", "spatula",
               "syringe", "coffee"]
MODELS = ["opus_5", "opus_4_8", "fable_5", "gpt_5_6_sol", "gpt_5_6_terra"]
PRETTY = {"opus_5": "Opus 5", "opus_4_8": "Opus 4.8", "fable_5": "Fable 5",
          "gpt_5_6_sol": "GPT-5.6-Sol", "gpt_5_6_terra": "GPT-5.6-Terra"}

TIME_GRID = np.arange(0, 266, 5)                 # minutes (runs overrun 240 slightly)
TOK_GRID = np.unique(np.round(np.logspace(4.0, 7.3, 90)))   # 10k..20M tokens, log-spaced


def finals(model: str) -> list[str]:
    """The final no-tools attempt per task for one model (post-cleanup canonical set)."""
    special = {
        ("fable_5", "bulb"): "bulb_fable_5_r3",
        ("fable_5", "syringe"): "syringe_fable_5_r3",
        ("fable_5", "pc_ram"): "pc_ram_fable_5_r3",
        ("gpt_5_6_sol", "allen_bolt"): "allen_bolt_gpt_5_6_sol_r3",
        ("gpt_5_6_sol", "pc_motherboard"): "pc_motherboard_gpt_5_6_sol_r3",
        ("gpt_5_6_terra", "ikea_table"): "ikea_table_gpt_5_6_terra_r3",
        ("gpt_5_6_terra", "pc_motherboard"): "pc_motherboard_gpt_5_6_terra_r3",
        ("gpt_5_6_terra", "syringe"): "syringe_gpt_5_6_terra_r2",
    }
    out = []
    for t in PHYSX_TASKS:
        if (model, t) in special:
            out.append(special[(model, t)])
        elif model in ("opus_5", "opus_4_8", "fable_5"):
            out.append(f"{t}_{model}_r2")
        else:
            out.append(f"{t}_{model}")
    return out


TOOLS_SET = [f"{t}_fable_5_rtools1b" if t == "allen_bolt" else f"{t}_fable_5_rtools1"
             for t in PHYSX_TASKS]
FORK_TASKS = ["bulb", "ikea_table", "pc_ram", "spatula"]


def curve(label: str, grid, axis: str):
    rec = INDEX.get(label)
    if rec is None:
        print(f"  MISSING from index: {label}")
        return None
    xs = [(e["minute"] if axis == "time" else e["cum_tokens"]) for e in rec["events"]]
    ss = [e["score"] for e in rec["events"]]
    vals = np.zeros(len(grid))
    for i, g in enumerate(grid):
        s = 0.0
        for x, sc in zip(xs, ss):
            if x <= g:
                s = sc
        vals[i] = s
    return vals


def mean_curve(labels, grid, axis):
    cs = [c for c in (curve(l, grid, axis) for l in labels) if c is not None]
    return np.mean(cs, axis=0) if cs else None


def plot(series, title, axis, fname):
    plt.figure(figsize=(7.5, 4.6))
    grid = TIME_GRID if axis == "time" else TOK_GRID / 1e6
    for name, y in series.items():
        if y is not None:
            plt.plot(grid, y, label=name, linewidth=2)
    if axis == "tokens":
        plt.xscale("log")
    plt.xlabel("minutes (agent wall clock)" if axis == "time"
               else "uncached input + output tokens (millions, log scale)")
    plt.ylabel("mean rubric score")
    plt.title(title, fontsize=11)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=130)
    plt.close()
    print(f"wrote {PLOTS / fname}")


for axis, suffix in (("time", "time"), ("tokens", "tokens")):
    grid = TIME_GRID if axis == "time" else TOK_GRID
    # 1. baseline: five models, no tools, mean over the 13 PhysX tasks
    plot({PRETTY[m]: mean_curve(finals(m), grid, axis) for m in MODELS},
         "Baseline (no tools): mean rubric score over 13 tasks",
         axis, f"baseline_{suffix}.png")
    # 2. generalization: fork vs cold, per target task
    cold_fable = dict(zip(PHYSX_TASKS, finals("fable_5")))
    for task in FORK_TASKS:
        plot({"cold (no nut_thread context)": mean_curve([cold_fable[task]], grid, axis),
              "forked from nut_thread state": mean_curve([f"{task}_fable_5_forknut"],
                                                         grid, axis)},
             f"nut_thread \u2192 {task} transfer (Fable 5)", axis,
             f"gen_{task}_{suffix}.png")
    # 2b. generalization aggregate: mean over the four fork tasks
    plot({"cold (no nut_thread context)":
              mean_curve([cold_fable[t] for t in FORK_TASKS], grid, axis),
          "forked from nut_thread state":
              mean_curve([f"{t}_fable_5_forknut" for t in FORK_TASKS], grid, axis)},
         "nut_thread \u2192 {bulb, ikea, pc_ram, spatula}: mean over 4 tasks (Fable 5)",
         axis, f"gen_average_{suffix}.png")
    # 3. tools vs no tools (fable), state-graded both arms
    plot({"Fable 5, no tools": mean_curve(finals("fable_5"), grid, axis),
          "Fable 5, with tools (state-graded)": mean_curve(TOOLS_SET, grid, axis)},
         "Tools vs no tools (Fable 5): mean rubric score over 13 tasks",
         axis, f"tools_{suffix}.png")

# ----- cross-task generalization campaign (2026-08-18): solution-file hints ------------------
# 12 (target, source, near/far) pairs x {opus_5, gpt_5_6_sol} x 3 seeds, plus no-hint
# baselines (seed 1 = the original campaign run, seeds 2-3 = rbase relaunches).
GEN_PAIRS = [
    ("bulb", "nut_thread", "near"), ("bulb", "tool_packing", "far"),
    ("allen_bolt", "nut_thread", "near"), ("allen_bolt", "pen_holder", "far"),
    ("pc_motherboard", "allen_bolt", "near"), ("pc_motherboard", "spatula", "far"),
    ("pc_ram", "pc_gpu", "near"), ("pc_ram", "bulb", "far"),
    ("tool_packing", "pen_holder", "near"), ("tool_packing", "nut_thread", "far"),
    ("spatula", "allen_bolt", "near"), ("spatula", "pc_gpu", "far"),
]
GEN_TASKS = sorted({t for t, _, _ in GEN_PAIRS})
GEN_MODELS = ["opus_5", "gpt_5_6_sol"]
SEEDS = (1, 2, 3)


def gen_labels(task: str, kind: str) -> list[str]:
    """All hint-run labels for one target task and one pair type (near/far)."""
    return [f"{task}_{m}_rgen_{src}_s{s}"
            for t, src, k in GEN_PAIRS if t == task and k == kind
            for m in GEN_MODELS for s in SEEDS]


def base_labels(task: str) -> list[str]:
    """No-hint baselines: the original campaign run (seed 1) + the rbase seeds."""
    seed1 = {m: dict(zip(PHYSX_TASKS, finals(m))) for m in GEN_MODELS}
    out = [seed1[m][task] for m in GEN_MODELS]
    out += [f"{task}_{m}_rbase_s{s}" for m in GEN_MODELS for s in (2, 3)]
    return out


# Publication styling: colorblind-safe palette, mean +- SEM bands, clean spines, vector +
# high-dpi raster outputs.
STYLE = {
    "near": dict(color="#0173B2", label="Near-task hint"),
    "far": dict(color="#DE8F05", label="Far-task hint"),
    "none": dict(color="#5A5A5A", label="No hint", linestyle="--"),
}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (camera-ready requirement)
})
TOKM = TOK_GRID / 1e6


def band(labels):
    """(mean, sem, n) over the group's step curves on the token grid."""
    cs = [c for c in (curve(l, TOK_GRID, "tokens") for l in labels) if c is not None]
    if not cs:
        return None
    a = np.vstack(cs)
    return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(len(cs)), len(cs)


def draw(ax, groups, title=None, style=None):
    style = style or STYLE
    for key, labels in groups.items():
        b = band(labels)
        if b is None:
            continue
        m, sem, n = b
        st = style[key]
        ax.plot(TOKM, m, linewidth=1.8, color=st["color"],
                linestyle=st.get("linestyle", "-"), label=f"{st['label']} (n={n})")
        ax.fill_between(TOKM, m - sem, m + sem, color=st["color"], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(TOKM[0], TOKM[-1])
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.5)
    if title:
        ax.set_title(title)


def save(fig, stem):
    for ext, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS / stem}.png/.pdf")


# per-task panels, individually and as one 2x3 grid figure
fig_g, axes = plt.subplots(2, 3, figsize=(9.2, 5.4), sharex=True, sharey=True)
for i, task in enumerate(GEN_TASKS):
    groups = {"near": gen_labels(task, "near"), "far": gen_labels(task, "far"),
              "none": base_labels(task)}
    ax = axes.flat[i]
    draw(ax, groups, title=task.replace("_", " "))
    if i % 3 == 0:
        ax.set_ylabel("rubric score")
    if i >= 3:
        ax.set_xlabel("uncached input + output tokens (M)")
    # standalone panel
    f1, a1 = plt.subplots(figsize=(3.6, 2.8))
    draw(a1, groups, title=task.replace("_", " "))
    a1.set_xlabel("uncached input + output tokens (M)")
    a1.set_ylabel("rubric score")
    a1.legend(loc="upper left")
    save(f1, f"genv2_{task}_tokens")
handles, labels_ = axes.flat[0].get_legend_handles_labels()
fig_g.legend(handles, labels_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig_g.tight_layout()
save(fig_g, "genv2_grid_tokens")

# aggregates over the 6 targets
all_groups = {
    "near": [l for t in GEN_TASKS for l in gen_labels(t, "near")],
    "far": [l for t in GEN_TASKS for l in gen_labels(t, "far")],
    "none": [l for t in GEN_TASKS for l in base_labels(t)],
}
for stem, keys, title in (
        ("genv2_near_tokens", ("near", "none"), "Near-task hints vs no hint (6 tasks)"),
        ("genv2_far_tokens", ("far", "none"), "Far-task hints vs no hint (6 tasks)"),
        ("genv2_all_tokens", ("near", "far", "none"),
         "Solution-file hint transfer (6 tasks, 2 models, 3 seeds)")):
    f, a = plt.subplots(figsize=(4.4, 3.2))
    draw(a, {k: all_groups[k] for k in keys}, title=title)
    a.set_xlabel("uncached input + output tokens (millions, log scale)")
    a.set_ylabel("mean rubric score")
    a.legend(loc="upper left")
    save(f, stem)

# ----- cross-embodiment campaign (2026-08-18): same-task Franka solution as the hint ---------
# 6 tasks x {gen3n7 (Kinova Gen3 + panda hand), xarm7 (+ panda hand)} x 2 models x 3 seeds.
# Reference: the same models on the FRANKA task with no hint (the genv2 baselines).
EMB_TASKS = ["allen_bolt", "bulb", "pc_motherboard", "pc_ram", "tool_packing", "spatula"]
STYLE_EMB = {
    "gen3n7": dict(color="#029E73", label="Kinova Gen3, Franka solution as hint"),
    "xarm7": dict(color="#CC78BC", label="xArm7, Franka solution as hint"),
    "none": dict(color="#5A5A5A", label="Franka, no hint (reference)", linestyle="--"),
}


def emb_labels(task: str, emb: str) -> list[str]:
    return [f"{task}_{emb}_{m}_remb_s{s}" for m in GEN_MODELS for s in SEEDS]


fig_e, axes_e = plt.subplots(2, 3, figsize=(9.2, 5.4), sharex=True, sharey=True)
for i, task in enumerate(EMB_TASKS):
    groups = {"gen3n7": emb_labels(task, "gen3n7"), "xarm7": emb_labels(task, "xarm7"),
              "none": base_labels(task)}
    ax = axes_e.flat[i]
    draw(ax, groups, title=task.replace("_", " "), style=STYLE_EMB)
    if i % 3 == 0:
        ax.set_ylabel("rubric score")
    if i >= 3:
        ax.set_xlabel("uncached input + output tokens (M)")
    f1, a1 = plt.subplots(figsize=(3.6, 2.8))
    draw(a1, groups, title=task.replace("_", " "), style=STYLE_EMB)
    a1.set_xlabel("uncached input + output tokens (M)")
    a1.set_ylabel("rubric score")
    a1.legend(loc="upper left")
    save(f1, f"embv1_{task}_tokens")
handles_e, labels_e = axes_e.flat[0].get_legend_handles_labels()
fig_e.legend(handles_e, labels_e, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig_e.tight_layout()
save(fig_e, "embv1_grid_tokens")

emb_all = {
    "gen3n7": [l for t in EMB_TASKS for l in emb_labels(t, "gen3n7")],
    "xarm7": [l for t in EMB_TASKS for l in emb_labels(t, "xarm7")],
    "none": [l for t in EMB_TASKS for l in base_labels(t)],
}
f, a = plt.subplots(figsize=(4.4, 3.2))
draw(a, emb_all, title="Cross-embodiment transfer (6 tasks, 2 models, 3 seeds)",
     style=STYLE_EMB)
a.set_xlabel("uncached input + output tokens (millions, log scale)")
a.set_ylabel("mean rubric score")
a.legend(loc="upper left")
save(f, "embv1_all_tokens")

# ----- tools vs no tools for Opus 5 and GPT-5.6-Sol (2026-08-18, in-progress) ----------------
# The 26-run tools campaign (13 PhysX tasks x {opus_5, gpt_5_6_sol}, `default` condition).
# Many runs are still mid-budget, so the time axis is truncated to the furthest elapsed
# minute any tool run has actually reached — beyond that the step curves would just be a flat
# carry-forward that reads as "done" when the runs are in fact still going.
TOOLS26_MODELS = [("opus_5", "Opus 5"), ("gpt_5_6_sol", "GPT-5.6-Sol")]


def _max_minute(labels) -> float:
    mins = [e["minute"] for l in labels if (r := INDEX.get(l))
            for e in r["events"]] + [0]
    return max(mins)


for model, pretty in TOOLS26_MODELS:
    tool_labels = [f"{t}_{model}_rtools1" for t in PHYSX_TASKS]
    base_labels_m = finals(model)
    present = [l for l in tool_labels if l in INDEX]
    if not present:
        continue
    # time axis, truncated to the deepest elapsed minute reached so far
    cap = max(30, min(TIME_GRID[-1], _max_minute(present) + 5))
    tgrid = TIME_GRID[TIME_GRID <= cap]
    plt.figure(figsize=(7.5, 4.6))
    for name, labs, color in ((f"{pretty}, no tools", base_labels_m, "#5A5A5A"),
                              (f"{pretty}, with tools", tool_labels, "#0173B2")):
        y = mean_curve(labs, tgrid, "time")
        if y is not None:
            plt.plot(tgrid, y, label=f"{name} (n={sum(l in INDEX for l in labs)})",
                     linewidth=2, color=color)
    plt.xlabel("minutes (agent wall clock; tools arm still in progress)")
    plt.ylabel("mean rubric score")
    plt.title(f"Tools vs no tools ({pretty}): 13 tasks, tools arm live @ \u2264{int(cap)} min",
              fontsize=11)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / f"tools26_{model}_time.png", dpi=130)
    plt.close()
    print(f"wrote {PLOTS / f'tools26_{model}_time.png'}")
    # token axis: honest for in-progress runs (each plotted to its own current token count)
    plt.figure(figsize=(7.5, 4.6))
    for name, labs, color in ((f"{pretty}, no tools", base_labels_m, "#5A5A5A"),
                              (f"{pretty}, with tools", tool_labels, "#0173B2")):
        y = mean_curve(labs, TOK_GRID, "tokens")
        if y is not None:
            plt.plot(TOK_GRID / 1e6, y, label=f"{name} (n={sum(l in INDEX for l in labs)})",
                     linewidth=2, color=color)
    plt.xscale("log")
    plt.xlabel("uncached input + output tokens (millions, log scale)")
    plt.ylabel("mean rubric score")
    plt.title(f"Tools vs no tools ({pretty}): 13 tasks (tools arm in progress)", fontsize=11)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / f"tools26_{model}_tokens.png", dpi=130)
    plt.close()
    print(f"wrote {PLOTS / f'tools26_{model}_tokens.png'}")

print("ALL PLOTS DONE")
