"""Spend accounting: what a delivery cost, read from the run's archives.

The single home for agent-transcript knowledge, imported by
eval/scripts/run_grade.py — a new agent CLI (codex, ...) is supported by
extending THIS file only.

Formats:
  claude — stream-json at workspace/.agent/transcript.jsonl: per-message
  usage on assistant events; a final 'result' event carries session totals
  (total_cost_usd, num_turns, usage). Budget-killed runs have no result
  event — token counts are summed from the per-message records instead;
  cost is left unavailable rather than estimated from stale price tables.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def usage_upto(transcript: Path, n_lines: int) -> dict:
    """Sum token usage over the first n_lines events — the session's spend at
    the moment a submission was frozen (its stamped transcript position)."""
    total: dict = {}
    with transcript.open() as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            for k, v in ((ev.get("message") or {}).get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    total[k] = total.get(k, 0) + v
    return total


def session_totals(transcript: Path) -> dict:
    """Whole-session totals: the final result event's own numbers, else
    summed per-message usage (no result event = the run was killed)."""
    out: dict = {}
    for line in transcript.open():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            out = {"cost_usd": ev.get("total_cost_usd"),
                   "num_turns": ev.get("num_turns"),
                   "usage": ev.get("usage")}
    if "usage" not in out:
        out["usage"] = usage_upto(transcript, sys.maxsize)
    return out


def compute_spend(run_dir: Path | None, submission_dir: Path | None = None) -> dict:
    """The spend behind one delivery — the curve's x axis.

    A submission prices as its pinned pointers (submitted.json's wall clock +
    transcript position, tokens summed over that prefix); the final solution
    prices as the whole session (run.json wall + session totals). {} when
    there is no run context (an explicit --solution grade)."""
    if run_dir is None:
        return {}
    transcript = run_dir / "workspace" / ".agent" / "transcript.jsonl"

    if submission_dir is not None:
        spend: dict = {}
        meta = submission_dir / "submitted.json"
        if meta.exists():
            spend = json.loads(meta.read_text())
        if spend.get("transcript_lines") and transcript.exists():
            spend["usage"] = usage_upto(transcript, spend["transcript_lines"])
        return spend

    spend = {}
    run_meta = run_dir / "run.json"
    if run_meta.exists():
        rm = json.loads(run_meta.read_text())
        try:
            spend["wall_s"] = round((datetime.fromisoformat(rm["ended"])
                                     - datetime.fromisoformat(rm["started"])).total_seconds(), 1)
        except (KeyError, ValueError):
            pass
    if transcript.exists():
        spend.update(session_totals(transcript))
    return spend
