#!/usr/bin/env python3
"""Reached-state grading from the agents' own scene-API printouts.

Agents ran closed-loop: their scripts constantly printed live scene readings — both the
OFFICIAL scorers (scene.score()/success()/seated()/engaged()/stowed()/doses...) and raw
poses. Those prints are execution outputs (not model text), timestamped by the request
that carried them, and each print format is traceable to a verified scene-reading line in
the run's reconstructed code. This module parses TIER-1 evidence — direct scorer
printouts — into timestamped score events:

    results/<label>/telemetry_scores.json:
        {"label", "events": [{"ts_ms", "score", "kind", "evidence"}], "peak"}

Score mapping per task uses the OFFICIAL score shapes (score()/100 where defined; else
per-part fractions matching the fine rubric's stage weights for boolean part masks).
Validation: on runs whose in-run verify passed, the parsed curve must reach success.

    python3 telemetry_scores.py --label <label>          # one run
    python3 telemetry_scores.py --all                    # every run with a trajectory
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"

TASKS = ("allen_bolt", "bulb", "ikea_table", "nut_thread", "pc_gpu", "pc_ram",
         "pc_gpu_ram", "pc_motherboard", "coffee", "spatula", "syringe", "pen_holder",
         "tool_packing")

BOOL_VEC = r"\[((?:\s*(?:True|False|1|0)\s*,?)+)\]"
FLOAT_ = r"[-+]?\d+(?:\.\d+)?(?:e-?\d+)?"

# tier-1 patterns: the OFFICIAL scorers' printed values. Each maps a match to a score in
# [0,1] given the task. Boolean part-vectors score as (fraction of True) scaled to the
# task's per-part weight — the same shape as the official score for multi-part tasks.
def _frac_true(vec: str) -> float:
    toks = [t.strip() for t in vec.split(",") if t.strip()]
    vals = [t in ("True", "1") for t in toks]
    return (sum(vals) / len(vals)) if vals else 0.0


# Per-task glossaries: printed predicate -> its official-score meaning. A predicate only
# enters if it corresponds EXACTLY to a component of the scene's own scorer; sub-goal
# prints an agent invented for itself are excluded. Coffee/spatula map to their official
# latched-stage tables; multi-part tasks map boolean vectors to per-part fractions.
COFFEE_FLAGS = ("pod_loaded|pod_seated", "bay_closed|cover_closed",
                "cup_staged|cup_ok|cup_under", "started|brew_start",
                "brewed|filled", "cup_out", "served")
SPATULA_STAGES = (("lifted", 0.10), ("wedged", 0.25), ("flipped", 0.50),
                  ("loaded", 0.70), ("served", 1.00))


def make_patterns(task: str):
    """[(regex, score_fn, kind)] — tier-1: values of the OFFICIAL scorers only."""
    pats = []
    # scene success — word-anchored so pod_seated etc. can never match
    pats.append((re.compile(r"(?<![a-z_])success\(?\)?[^\n=]{0,16}?"
                            r"(?:\[?\s*(True|False)|tensor\(\[(True|False)\])"),
                 lambda m: 1.0 if "True" in m.group(0) else 0.0, "success_print"))
    # scene score() 0..100
    pats.append((re.compile(r"(?<![a-z_])score\(?\)?[^\n\d-]{0,12}(\d{1,3})(?!\d)"),
                 lambda m: min(int(m.group(1)), 100) / 100.0, "score_print"))
    N_PARTS = {"allen_bolt": 1, "bulb": 1, "nut_thread": 1, "pc_gpu": 1,
               "ikea_table": 4, "pc_ram": 2, "pc_gpu_ram": 3, "pc_motherboard": 7}
    if task in N_PARTS:
        # official seated() masks: per-part fraction IS the fine-grained shape
        pats.append((re.compile(r"(?<![a-z_])seated[^\n\[]{0,24}" + BOOL_VEC),
                     lambda m: _frac_true(m.group(1)), "seated_vector"))
        # a SCALAR seated print on a multi-part task is a PER-PART claim (agents print
        # "leg0 seated: True" for the part in hand): worth at most 1/n_parts. Only on a
        # single-part task does it claim the whole task (and then the >=0.999 rule below
        # demands verify corroboration anyway).
        n = N_PARTS[task]
        pats.append((re.compile(r"(?<![a-z_])seated[^\n]{0,12}?(True|False)(?![a-z])"),
                     lambda m, n=n: (1.0 / n) if m.group(1) == "True" else 0.0,
                     "seated_bool"))
    if task == "tool_packing":
        pats.append((re.compile(r"(?<![a-z_])stowed[^\n\[]{0,24}" + BOOL_VEC),
                     lambda m: 0.6 * _frac_true(m.group(1)), "stowed_vector"))
    if task == "pen_holder":
        # official score table: 0/1/2/3 counted -> 0/.10/.25/.40; all four -> .90
        def pens(m):
            f = _frac_true(m.group(1))
            k = round(f * 4)
            return {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.40, 4: 0.90}.get(k, 0.0)
        pats.append((re.compile(r"(?<![a-z_])(?:counted|inserted)[^\n\[]{0,24}" + BOOL_VEC),
                     pens, "pens_vector"))
    if task == "coffee":
        # the 7 latched stages, printed by name: score = distinct stages seen / 7.
        # (stateful: handled by the caller via kind='coffee_flag:<i>')
        for i, names in enumerate(COFFEE_FLAGS):
            pats.append((re.compile(r"(?<![a-z_])(?:" + names + r")\s*[=: ]\s*True"),
                         lambda m, i=i: -(i + 1),        # sentinel: flag index
                         f"coffee_flag:{i}"))
    if task == "spatula":
        for name, val in SPATULA_STAGES:
            pats.append((re.compile(r"(?<![a-z_])" + name + r"\s*[=: ]\s*True"),
                         lambda m, v=val: v, f"spatula_{name}"))
    # ---- tier-2: numeric depth/turn readings mapped through the SAME stage formulas as
    # rubrics.py. Only lines that look like runtime telemetry count: source-code echoes
    # (agents cat-ing scene files, cfg dumps) are rejected by the caller's line filter.
    # task -> (stage_baseline_m, seat_m). ONLY tasks whose printed-depth convention was
    # verified against scene code (engagement/tip depth INCREASING toward the seat).
    # bulb and ikea_table are excluded: their scene conventions descend and their agents'
    # 'depth' print semantics were not verifiable — unverified conventions inflate.
    DEPTH_MAPS = {
        "allen_bolt": (0.004898, 0.022), "pc_motherboard": (0.006, 0.011),
        "nut_thread": (0.0, 0.013), "pc_gpu": (0.0, 0.004),
        "pc_ram": (0.0, 0.0037), "pc_gpu_ram": (0.0, 0.004),
    }
    if task in DEPTH_MAPS:
        stage, seat = DEPTH_MAPS[task]

        def depth_score(m, stage=stage, seat=seat):
            v = float(m.group(1))
            if abs(v) > 1.0:        # printed in mm
                v = v / 1000.0
            frac = max(0.0, min((v - stage) / (seat - stage), 1.0))
            if frac <= 0.0:
                return 0.05         # engaged enough to measure depth = approach evidence
            return min(0.25 + 0.65 * frac, 0.99)
        pats.append((re.compile(
            r"(?<![a-z_])(?:depth|engag\w*|tip[_ ]?depth)\s*[=: ]\s*\+?(" + FLOAT_ + r")"),
            depth_score, "tier2_depth"))
    if task == "pc_motherboard":
        # screw turns: pitch 1 mm/turn from the 6 mm stage toward the 11 mm seat
        pats.append((re.compile(r"(?<![a-z_])turns?\s*[=: ]\s*(" + FLOAT_ + r")"),
                     lambda m: min(0.25 + 0.65 * max(0.0, min(
                         float(m.group(1)) * 0.001 / 0.005, 1.0)), 0.99), "tier2_turns"))
    if task == "syringe":
        # printed dose ledger floats -> official band membership
        def doses(m):
            vals = [float(x) for x in re.findall(FLOAT_, m.group(1))][:3]
            if not vals:
                return 0.0
            def tube(dv):
                if 0.23 <= dv <= 0.43:
                    return 1.0
                return max(0.0, dv / 0.23) if dv < 0.23 else max(0.0, 1 - (dv - 0.43) / 0.10)
            return 0.55 * sum(tube(v) for v in vals) / 3.0
        pats.append((re.compile(r"(?<![a-z_])doses?\s*[=: ]\s*(\[[^\]]{0,80}\])"),
                     doses, "doses_print"))
        pats.append((re.compile(r"(?<![a-z_])(?:drawn|_drawn_ok)\s*[=: ]\s*True"),
                     lambda m: 0.30, "drawn_print"))
    return pats


def task_of(label: str) -> str | None:
    for cand in (RESULTS / label / "run.json", RESULTS / label / "gt" / "run.json"):
        if cand.exists():
            try:
                p = json.loads(cand.read_text()).get("preset")
                if p:
                    return p.split(".")[1]
            except Exception:  # noqa: BLE001
                continue
    inf = Path(__file__).resolve().parent / "inferred_presets.json"
    if inf.exists():
        p = json.loads(inf.read_text()).get(label)
        if p:
            return p.split(".")[1]
    return None


def iter_outputs(traj: Path):
    """(ts_ms, output_text) for every tool result in the trajectory, both agent kinds."""
    with traj.open(errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            ts = rec.get("ts_ms")
            req = rec.get("request") or {}
            for m in req.get("messages") or []:          # claude
                c = m.get("content")
                if not isinstance(c, list):
                    continue
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        cc = b.get("content")
                        text = cc if isinstance(cc, str) else " ".join(
                            x.get("text", "") for x in cc or [] if isinstance(x, dict))
                        if text:
                            yield ts, text
            for it in req.get("input") or []:            # codex
                if isinstance(it, dict) and it.get("type") in (
                        "custom_tool_call_output", "function_call_output"):
                    out = it.get("output")
                    if isinstance(out, str) and out:
                        yield ts, out
                    elif isinstance(out, list):          # content blocks (input_text)
                        text = " ".join(x.get("text", "") for x in out
                                        if isinstance(x, dict))
                        if text:
                            yield ts, text


ANCHORS_FILE = RESULTS / "verify_anchors.json"
VERIFIED = set(json.loads(ANCHORS_FILE.read_text()).keys()) if ANCHORS_FILE.exists() else set()


def score_run(label: str) -> dict | None:
    task = task_of(label)
    if task not in TASKS:
        return None
    traj = None
    for name in ("raw_requests.jsonl", "raw_requests.full.jsonl"):
        if (RESULTS / label / name).exists():
            traj = RESULTS / label / name
            break
    if traj is None:
        return None
    pats = make_patterns(task)
    events = []
    seen_out = set()   # tool outputs repeat in every later request: parse each ONCE
    coffee_flags: set = set()
    for ts, text in iter_outputs(traj):
        h = hash(text[:400])
        if h in seen_out:
            continue
        seen_out.add(h)
        # reject source-code echoes (agents cat-ing scene files / cfg dumps): a "depth"
        # in `seat_depth: float = tunable(0.022)` is a constant, not a reading
        lines = [ln for ln in text.split("\n")
                 if not re.search(r"tunable\(|info\(|float =|def |import |^\s*#", ln)]
        text = "\n".join(lines)
        best = None
        for rx, fn, kind in pats:
            for m in rx.finditer(text):
                try:
                    v = fn(m)
                except Exception:  # noqa: BLE001
                    continue
                if v < 0:                      # coffee latched-flag sentinel
                    coffee_flags.add(int(-v) - 1)
                    v = len(coffee_flags) / 7.0
                ev = text[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")[:110]
                if best is None or v > best[0]:
                    best = (v, kind, ev)
        if best is not None:
            # SUCCESS-CLAIM GATE: a print-derived full success (>=0.999) may come from a
            # manipulated or differently-seeded dev env (measured: coffee_fable_5_r2
            # printed SUCCESS=True score=100 while its official in-run verify failed).
            # Full success counts ONLY when the run's official verify corroborates it;
            # otherwise the claim event is dropped (partial evidence below 0.999 stays).
            if best[0] >= 0.999 and label not in VERIFIED:
                continue
            events.append({"ts_ms": ts, "score": round(best[0], 4),
                           "kind": best[1], "evidence": best[2]})
    if not events:
        return {"label": label, "task": task, "events": [], "peak": 0.0}
    # monotone best-so-far
    mono, cur = [], -1.0
    for e in sorted(events, key=lambda e: e["ts_ms"] or 0):
        if e["score"] > cur:
            cur = e["score"]
            mono.append(e)
    return {"label": label, "task": task, "events": mono, "peak": cur,
            "n_raw_readings": len(events)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    labels = [args.label] if args.label else \
        sorted(d.name for d in RESULTS.iterdir() if d.is_dir())
    n = 0
    for label in labels:
        try:
            rec = score_run(label)
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        if rec is None:
            continue
        (RESULTS / label / "telemetry_scores.json").write_text(
            json.dumps(rec, indent=1) + "\n")
        n += 1
        print(f"{label}: peak={rec['peak']} events={len(rec['events'])} "
              f"(from {rec.get('n_raw_readings', 0)} readings)", flush=True)
    print(f"TELEMETRY DONE: {n} runs", flush=True)


if __name__ == "__main__":
    main()
