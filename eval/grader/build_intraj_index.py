#!/usr/bin/env python3
"""Build results/milestones_index.json from IN-TRAJECTORY evidence ONLY (no replay):
  * telemetry_scores.json  — the agents' own scene-API printouts (tier-1 + tier-2)
  * verify_anchors.json    — official in-run verify passes (score 1.0 at a known time)
  * grades_ckpt.json       — tool runs' checkpoint-tree states (recorded in-run) with
                             their official scores; rubric refinements when present
Replay-derived grades are deliberately NOT consulted (OSC replay is brittle).
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_replay_index import token_timeline  # reuse ts->tokens machinery

ANCHORS = json.loads((RESULTS / "verify_anchors.json").read_text()) \
    if (RESULTS / "verify_anchors.json").exists() else {}


def build(label: str) -> dict | None:
    pts = []
    tj = RESULTS / label / "telemetry_scores.json"
    if tj.exists():
        for e in json.loads(tj.read_text()).get("events", []):
            if e.get("ts_ms") and e.get("score") is not None:
                pts.append((e["ts_ms"], e["score"], "telemetry:" + e.get("kind", "?")))
    ck = RESULTS / label / "grades_ckpt.json"
    if ck.exists():
        rj = RESULTS / label / "rubric_scores.json"
        rub = {}
        if rj.exists():
            for s in json.loads(rj.read_text()).get("states", []):
                if s.get("kind") == "tree_node" and s.get("rubric") is not None:
                    rub[s["cid"]] = s["rubric"]
        for n in json.loads(ck.read_text()).get("nodes", []):
            if n.get("created") and n.get("score") is not None:
                v = max(n["score"], rub.get(n["cid"], 0.0))
                pts.append((int(n["created"] * 1000), v, "ckpt_state"))
    if label in ANCHORS:
        pts.append((ANCHORS[label], 1.0, "verify_rc0"))
    if not pts:
        return None
    tl = token_timeline(label)
    t0 = tl[0][0] if tl else None
    last_tok = tl[-1][1] if tl else 0

    def locate(ts):
        if not tl or t0 is None:
            return 0.0, 0
        cum = 0
        for t, c in tl:
            if t <= ts:
                cum = c
            else:
                break
        return max(0.0, (ts - t0) / 60000.0), cum

    events, best = [], 0.0
    for ts, v, src in sorted(pts):
        if v > best:
            best = v
            minute, cum = locate(ts)
            events.append({"minute": round(minute, 2), "cum_tokens": cum,
                           "score": round(v, 4), "source": src})
    if not events:
        events = [{"minute": 0.0, "cum_tokens": 0, "score": 0.0, "source": "none"}]
    return {"label": label, "events": events, "final_score": best,
            "ever_success": best >= 0.999, "grading": "in-trajectory-v1"}


def main() -> None:
    out = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        rec = build(d.name)
        if rec is not None:
            out.append(rec)
    (RESULTS / "milestones_index.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"in-traj index: {len(out)} runs")


if __name__ == "__main__":
    main()
