#!/usr/bin/env python3
"""Extract per-run rubric milestones ONCE and persist them for all later analysis.

For each run directory under results/ this writes results/<label>/milestones.json:

    {"label", "task", "model", "arm",            # run identity
     "t0_ms", "duration_min", "n_requests",
     "tokens_uncached_total", "tokens_output_total",
     "solved_official": bool,                    # legs.log's own marker
     "events": [{"minute", "cum_tokens", "score", "source", "evidence"}, ...],
     "final_score": float}

Events are monotone in score. Every event carries the EVIDENCE text that triggered it
(trajectory match ±80 chars, checkpoint-tree label, or the legs.log line), so each
milestone can be audited without re-reading the 100s-of-MB trajectory.

Scoring follows results/rubrics.md (tentative). Three sources, in trust order:
  legs-solved   — the harness's own passed success check -> 1.0 (strongest)
  tree-label    — a checkpoint node's label (an explicit state declaration)
  traj-marker   — measured-output patterns / seated-style boolean arrays in tool output
                  (strictly length-gated per task; the first record — the prompt — is
                  never matched, so goal text cannot score)

Usage:
    python eval/grader/extract_milestones.py            # all runs missing milestones.json
    python eval/grader/extract_milestones.py --force    # re-extract everything
    python eval/grader/extract_milestones.py --only coffee_fable_5_rtools1
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
TAIL = 250_000

TASKS = ["allen_bolt", "bulb", "ikea_table", "nut_thread", "pc_gpu", "pc_gpu_ram",
         "pc_motherboard", "pc_ram", "pen_holder", "tool_packing", "spatula",
         "syringe", "coffee", "tshirt", "latte"]
MODELS = ["opus_5", "opus_4_8", "fable_5", "gpt_5_6_sol", "gpt_5_6_terra"]


def identify(label: str):
    task = next((t for t in sorted(TASKS, key=len, reverse=True)
                 if label.startswith(t + "_")), None)
    model = next((m for m in MODELS if f"_{m}" in label), None)
    if "forknut" in label:
        arm = "fork"
    elif "_rgen_" in label:
        arm = "gen_hint"      # cross-task generalization: a source task's solution provided
    elif "_remb_" in label:
        arm = "emb_hint"      # cross-embodiment: same task, Franka solution provided
    elif "_rbase_" in label:
        arm = "no_tools"      # extra no-hint baseline seeds (2026-08-17 campaign)
    elif "tools" in label:
        arm = "tools"
    elif task in ("tshirt", "latte"):
        arm = "newton"
    else:
        arm = "no_tools"
    return task, model, arm


def R(*pairs):
    return [(s, re.compile(p)) for s, p in pairs]


# ---- measured-output patterns on trajectory tool text (strict; see rubrics.md) ----------
RULES = {
    "allen_bolt": R((0.75, r"depth[ =]+(8|9|1[0-9])\.[0-9]+ ?mm"),
                    (0.5, r"depth[ =]+[1-7]\.[0-9]+ ?mm"),
                    (0.35, r"GRIPPED key|key welded"),
                    (0.05, r"GRIPPED")),
    "bulb": R((0.75, r"screwing progress depth 0\.0[01][0-9]|bulbz=0\.0[0-2][0-9]|"
                     r"seated the bulb"),
              (0.5, r"\[insert\] rest: z=0\.0[0-4]|inserted \(depth"),
              (0.1, r"GRIPPED|bulb held")),
    "ikea_table": R((0.15, r"(resting|vertical) on (stud|bolt)")),
    "nut_thread": R((0.75, r"\bd=[ ]?1[2-6]\.[0-9]+mm|nutz=0\.01[0-9]"),
                    (0.5, r"\bd=[ ]?(1[7-9]|2[0-3])\.[0-9]+mm|nutz=0\.02[0-3]"),
                    (0.1, r"GRIPPED|gap=0\.02[0-8]")),
    "pc_gpu": R((0.1, r"GRIPPED card")),   # engaged() is nonzero at spawn: not a marker
    "pc_gpu_ram": R((0.05, r"GRIPPED card")),
    "pc_motherboard": R((0.25, r"seated=True depth=1[01]\.[0-9]+mm"),
                        (0.1, r"GRIPPED key|key welded")),
    "pc_ram": R((0.1, r"GRIPPED")),
    "pen_holder": R((1.0, r"(?<!\()(?<!erify )score[:= ]+100"), (0.8, r"(?<!\()(?<!erify )score[:= ]+8[0-9]"),
                    (0.6, r"(?<!\()(?<!erify )score[:= ]+[67][0-9]"), (0.4, r"(?<!\()(?<!erify )score[:= ]+[45][0-9]"),
                    (0.2, r"(?<!\()(?<!erify )score[:= ]+[123][0-9]")),
    "tool_packing": R((1.0, r"(?<!\()(?<!erify )score[:= ]+100"),
                      (0.6, r"stowed.{0,25}True, True|2 tools stowed|score[:= ]+[67][0-9]"),
                      (0.3, r"scissors stowed|knife (placed|stowed)|stapler stowed|"
                            r"score[:= ]+[34][0-9]"),
                      (0.1, r"drawer (open|pulled)")),
    "spatula": R((1.0, r"(?<!\()(?<!erify )score[:= ]+100(?!\))"), (0.8, r"(?<!\()score[:= ]+[78][0-9](?!/)"),
                 (0.6, r"(?<!\()score[:= ]+[56][0-9](?!\))|flip ok=True"),
                 (0.4, r"(?<!\()score[:= ]+(2[5-9]|[34][0-9])(?!\))"),
                 (0.25, r"swept=-?1[0-9]{2}"),
                 (0.1, r"spatula.{0,15}(grasped|welded|held)")),
    "syringe": R((0.85, r"doses[^()]{0,40}0\.3[0-9][^()]{0,25}0\.3[0-9][^()]{0,25}0\.3"),
                 (0.65, r"doses[^()]{0,30}0\.3[0-9]"),
                 (0.55, r"drawn[ =:]+\[?True|liquid[ =:]+0\.[89]"),
                 (0.35, r"seated at (the )?reservoir"),
                 (0.1, r"GRIPPED|gripped")),
    # coffee stages come from TREE labels; traj only trusts measured score prints
    # ("brew complete(s)" and sX names appear in task text / checkpoint-calling code)
    "coffee": R((1.0, r"SUCCESS=True score=100|(?<!\()(?<!erify )score[:= ]+100"),
                (0.65, r"(?<!\()score=7[0-9]\b"),
                (0.35, r"(?<!\()score=29\b")),
    # scene docstring says "folded flat": only a measured success predicate counts
    "tshirt": R((1.0, r"success[ =:]+\[?\[?True")),
    # task text / bench-source comments contain pour language: predicate only
    "latte": R((1.0, r"success[ =:]+\[?\[?True")),
}

# (allowed lengths, {true_count: score}) for seated-style boolean arrays
ARRAY_SCORES = {
    "ikea_table": ({4}, {1: 0.30, 2: 0.55, 3: 0.78, 4: 1.0}),
    "pc_gpu_ram": ({3}, {1: 0.40, 2: 0.70, 3: 1.0}),
    "pc_ram": ({2}, {1: 0.5, 2: 1.0}),
    "pc_motherboard": ({7}, {k: min(1.0, 0.25 + 0.125 * (k - 1)) for k in range(1, 8)}),
    "nut_thread": ({1}, {1: 1.0}),
    "allen_bolt": ({1}, {1: 1.0}),
    "bulb": ({1, 2}, {1: 1.0, 2: 1.0}),
    "pc_gpu": ({1}, {1: 1.0}),
}

TREE_RULES = {
    "allen_bolt": R((1.0, r"seated"), (0.5, r"engag|thread"), (0.35, r"key")),
    "bulb": R((1.0, r"seated|glowing"), (0.75, r"screwing"), (0.5, r"insert"),
              (0.1, r"held|lifted")),
    "ikea_table": R((0.3, r"seated"), (0.15, r"(resting|vertical) on|nut set"),
                    (0.05, r"picked|lifted|grasped")),
    "nut_thread": R((1.0, r"seated"), (0.1, r"grasped|lifted")),
    "pc_gpu": R((1.0, r"card seated(?!.*not)"), (0.5, r"pressed into slot|engag"),
                (0.3, r"hovering above seat"), (0.1, r"gripped|lifted")),
    "pc_gpu_ram": R((1.0, r"all seated|True, True, True"), (0.7, r"stick 1 seated"),
                    (0.4, r"card seated"), (0.2, r"lowered at placement|slide height"),
                    (0.05, r"gripped|lifted")),
    "pc_motherboard": R((0.375, r"bolt ?1 seated"), (0.25, r"bolt ?0 seated|bolt0 seated"),
                        (0.1, r"key (gripped|welded|grasped)")),
    "pc_ram": R((1.0, r"True, True|both seated"), (0.5, r"seated"),
                (0.1, r"grasped|lifted")),
    "pen_holder": R((1.0, r"score 100|all pens"), (0.6, r"3 pens"), (0.4, r"2 pens"),
                    (0.2, r"pen (in|stowed)")),
    "tool_packing": R((1.0, r"all (tools|three)"),
                      (0.3, r"stowed|placed in"), (0.1, r"drawer open|grasped and lifted")),
    "spatula": R((1.0, r"served"), (0.6, r"flip"), (0.4, r"on (the )?blade"),
                 (0.25, r"under (the )?bread|wedged")),
    "syringe": R((1.0, r"success"), (0.55, r"drawn"), (0.35, r"reservoir"),
                 (0.2, r"upright|vertical"), (0.1, r"gripped|lifted")),
    "coffee": R((1.0, r"s7|served"), (0.85, r"s6|brewed"), (0.65, r"s5|started"),
                (0.5, r"s4|cup staged"), (0.35, r"s3|bay closed"), (0.25, r"s2|pod seated"),
                (0.1, r"cover open")),
    "tshirt": R((1.0, r"folded"), (0.4, r"fold"), (0.1, r"grasped|pinched")),
    "latte": R((1.0, r"poured|success"), (0.5, r"over (the )?cup"), (0.15, r"grasped|lifted")),
}


def _arrays_after(text, keyword):
    out = []
    for m in re.finditer(
            keyword + r".{0,40}?\[?\[((?:True|False)(?:, (?:True|False))*)\]", text):
        vals = m.group(1).split(", ")
        out.append((len(vals), vals.count("True"), m.start(), m.group(0)))
    return out


def usage_tokens(rec):
    u = (rec.get("response") or {}).get("usage") or {}
    fresh = u.get("input_tokens", 0) or 0
    cw = u.get("cache_creation_input_tokens", 0) or 0
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        fresh = max(0, fresh - cached)
    out = u.get("output_tokens", 0) or 0
    return fresh + cw + out, out


def extract(label: str) -> dict | None:
    run_dir = RESULTS / label
    traj = run_dir / "raw_requests.jsonl"
    if not traj.exists():
        return None
    task, model, arm = identify(label)
    if task is None:
        return None
    rules = RULES.get(task, [])
    spec = ARRAY_SCORES.get(task)

    events, best = [], 0.0
    cum_unc = cum_out = n = 0
    t0 = None
    tok_at_min = []
    prefix_len = 0    # forks: length of record 1's body = inherited context boundary
    with traj.open(errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != 200:
                continue
            n += 1
            ts = rec.get("ts_ms", 0) / 1000.0
            if t0 is None:
                t0 = ts
            minute = (ts - t0) / 60.0
            unc, out = usage_tokens(rec)
            cum_unc += unc
            cum_out += out
            tok_at_min.append((minute, cum_unc))
            body = json.dumps(rec.get("request", {}))
            if n <= 1:
                prefix_len = len(body)   # prompt (+ any inherited fork conversation)
                continue                 # goal text must not score
            start = max(len(body) - TAIL, prefix_len if arm == "fork" else 0)
            text = body[start:] + json.dumps(rec.get("response", {}))
            cand = None      # (score, evidence)
            for score, rx in rules:
                if score > best:
                    m = rx.search(text)
                    if m:
                        cand = (score, text[max(0, m.start() - 80):m.end() + 80])
                        break
            if spec:
                lens, amap = spec
                for kw in ("seated", "welded", "RESULTS"):
                    for ln, k, pos, frag in _arrays_after(text, kw):
                        if ln in lens and k in amap and amap[k] > best and \
                                (cand is None or amap[k] > cand[0]):
                            cand = (amap[k], text[max(0, pos - 80):pos + len(frag) + 40])
            if cand and cand[0] > best:
                best = cand[0]
                events.append({"minute": round(minute, 2), "cum_tokens": cum_unc,
                               "score": cand[0], "source": "traj-marker",
                               "evidence": cand[1][:240]})

    if t0 is None:
        return None

    def tok_of(minute):
        prior = [tk for m, tk in tok_at_min if m <= minute]
        return prior[-1] if prior else 0

    tree_f = run_dir / "tree.json"
    if tree_f.exists():
        try:
            for node in json.loads(tree_f.read_text()).get("nodes", {}).values():
                ts = node.get("created") or 0
                if ts < 1e9:
                    continue
                minute = (ts - t0) / 60.0
                lbl = node.get("label") or ""
                for score, rx in TREE_RULES.get(task, []):
                    if rx.search(lbl):
                        events.append({"minute": round(minute, 2),
                                       "cum_tokens": tok_of(minute), "score": score,
                                       "source": "tree-label", "evidence": lbl[:240]})
                        break
        except Exception as exc:
            print(f"  [{label}] tree parse failed: {exc}")

    solved = False
    legs_f = run_dir / "legs.log"
    if legs_f.exists():
        legs = legs_f.read_text(errors="replace")
        if "solves the task" in legs:
            solved = True
            for line in legs.splitlines():
                if line.startswith("success check rc=0"):
                    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
                    if m:
                        # legs.log stamps are UTC ("date -Is" on the pod); parsing them naive
                        # made .timestamp() assume LOCAL time (UTC-4 here), shifting every
                        # solve +240 min and off the time-axis plots (bug found 2026-08-16).
                        ts_utc = datetime.fromisoformat(m.group(1)).replace(
                            tzinfo=timezone.utc).timestamp()
                        minute = (ts_utc - t0) / 60.0
                        events.append({"minute": round(minute, 2),
                                       "cum_tokens": tok_of(minute), "score": 1.0,
                                       "source": "legs-solved", "evidence": line[:240]})

    events.sort(key=lambda e: (e["minute"], e["score"]))
    mono, best = [], 0.0
    for e in events:
        if e["score"] > best and e["minute"] >= 0:
            best = e["score"]
            mono.append(e)

    return {"label": label, "task": task, "model": model, "arm": arm,
            "t0_ms": int(t0 * 1000), "duration_min": round(tok_at_min[-1][0], 1),
            "n_requests": n, "tokens_uncached_total": cum_unc,
            "tokens_output_total": cum_out, "solved_official": solved,
            "events": mono, "final_score": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    index = []
    labels = [args.only] if args.only else sorted(
        d.name for d in RESULTS.iterdir()
        if d.is_dir() and (d / "raw_requests.jsonl").exists())
    for label in labels:
        out_f = RESULTS / label / "milestones.json"
        if out_f.exists() and not args.force:
            index.append(json.loads(out_f.read_text()))
            print(f"{label}: cached (final={index[-1]['final_score']})")
            continue
        rec = extract(label)
        if rec is None:
            print(f"{label}: SKIPPED (no data / unknown task)")
            continue
        out_f.write_text(json.dumps(rec, indent=2) + "\n")
        index.append(rec)
        print(f"{label}: final={rec['final_score']} events={len(rec['events'])} "
              f"solved={rec['solved_official']}")
    (RESULTS / "milestones_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {len(index)} runs -> results/milestones_index.json")


if __name__ == "__main__":
    main()
