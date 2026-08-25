#!/usr/bin/env python3
"""Build results/milestones_index.json from REPLAY grades (grades_v2.json).

Replaces the old heuristic extraction: every event is now a replay-graded code version.
For each run:
    events = [{minute, cum_tokens, score}]   one per graded version, score = running max
where minute/cum_tokens locate the version's write-time in the run (cum_tokens = uncached
input + cache-write + output up to that moment, same accounting as extract_milestones).

Sources of grades, in priority order:
    results/<label>/grades_v2.json           (canonical local store)
    grading_out/grading_out_snapshot/<label>/grades_v2.json   (cluster results; copied in)

    python3 eval/grader/build_replay_index.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
CLUSTER = REPO / "grading_out" / "grading_out_snapshot"


def usage_tokens(rec: dict) -> int:
    u = (rec.get("response") or {}).get("usage") or {}
    fresh = u.get("input_tokens", 0) or 0
    cw = u.get("cache_creation_input_tokens", 0) or 0
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        fresh = max(0, fresh - cached)
    out = u.get("output_tokens", 0) or 0
    return fresh + cw + out


def traj_path(label: str) -> Path | None:
    d = RESULTS / label
    for name in ("raw_requests.trimmed.jsonl", "raw_requests.jsonl",
                 "raw_requests.full.jsonl"):
        if (d / name).exists():
            return d / name
    return None


def token_timeline(label: str):
    """[(ts_ms, cum_tokens)] ascending, from the run's trajectory."""
    tp = traj_path(label)
    if tp is None:
        return []
    pts = []
    cum = 0
    with tp.open(errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            cum += usage_tokens(rec)
            ts = rec.get("ts_ms")
            if ts:
                pts.append((ts, cum))
    pts.sort()
    return pts


def build(label: str) -> dict | None:
    gj = RESULTS / label / "grades_v2.json"
    if not gj.exists():
        return None
    g = json.loads(gj.read_text())
    rows = g.get("grades", [])
    if not rows:
        return None
    # fine-grained rubric scores over stored states (version ends + tool-run tree nodes):
    # each event's value = max(official peak score, end-state rubric) — both are
    # calibrated to [0,1] with 1.0 = official success and matching staged latch values
    rub_ver: dict = {}
    rub_node: list = []
    rj = RESULTS / label / "rubric_scores.json"
    if rj.exists():
        for srow in json.loads(rj.read_text()).get("states", []):
            if srow.get("rubric") is None:
                continue
            if srow["kind"] == "version_end":
                rub_ver[srow["version"]] = srow["rubric"]
            elif srow["kind"] == "tree_node" and srow.get("ts_ms"):
                rub_node.append((srow["ts_ms"],
                                 max(srow["rubric"], srow.get("official_score") or 0.0)))
    tl = token_timeline(label)
    t0 = tl[0][0] if tl else None
    last_min = (tl[-1][0] - t0) / 60000.0 if tl and t0 else 240.0
    last_tok = tl[-1][1] if tl else 0

    def locate(ts_ms):
        if ts_ms is None or not tl or t0 is None:
            return last_min, last_tok
        minute = (ts_ms - t0) / 60000.0
        cum = 0
        for t, c in tl:
            if t <= ts_ms:
                cum = c
            else:
                break
        return max(0.0, minute), cum

    pts = []
    for r in rows:
        score = r.get("score")
        if score is None:
            continue
        val = max(score, rub_ver.get(r["version"], 0.0))
        pts.append((r.get("ts_ms"), val, r["version"], "replay+rubric"))
    for ts, val in rub_node:
        pts.append((ts, val, None, "tree_node"))
    events = []
    best = 0.0
    for ts, val, ver, src in sorted(pts, key=lambda p: (p[0] is None, p[0] or 0)):
        minute, cum = locate(ts)
        if val > best:
            best = val
            events.append({"minute": round(minute, 2), "cum_tokens": cum,
                           "score": round(val, 4), "source": src,
                           **({"version": ver} if ver else {})})
    # runs whose best is 0 still need one event so the curve exists (flat zero)
    if not events:
        events = [{"minute": 0.0, "cum_tokens": 0, "score": 0.0, "source": "replay",
                   "version": rows[0]["version"]}]
    return {"label": label, "events": events, "final_score": best,
            "peak_score": g.get("peak_score", best),
            "ever_success": g.get("ever_success", False),
            "n_versions_graded": len(rows), "grader": g.get("grader", "?")}


def main() -> None:
    # 1) copy cluster grades into the canonical store
    copied = 0
    if CLUSTER.is_dir():
        for d in sorted(CLUSTER.iterdir()):
            gj = d / "grades_v2.json"
            if gj.exists() and json.loads(gj.read_text()).get("grades"):
                dst = RESULTS / d.name
                dst.mkdir(exist_ok=True)
                shutil.copy2(gj, dst / "grades_v2.json")
                copied += 1
    # 2) build the index
    out = []
    missing = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        rec = build(d.name)
        if rec is not None:
            out.append(rec)
        elif (d / "versions.tgz").exists():
            missing.append(d.name)
    idx = RESULTS / "milestones_index.json"
    if idx.exists():
        shutil.copy2(idx, RESULTS / "milestones_index.heuristic_backup.json")
    idx.write_text(json.dumps(out, indent=1) + "\n")
    print(f"cluster grades copied: {copied}")
    print(f"index: {len(out)} runs with replay grades -> {idx}")
    print(f"reconstructed but not yet graded: {len(missing)}")
    for l in missing:
        print("   ", l)


if __name__ == "__main__":
    main()
