#!/usr/bin/env python3
"""Convert a codex CLI session rollout (~/.codex/sessions/.../rollout-*.jsonl) into the
relay's raw_requests.jsonl shape, for runs whose relay log is incomplete but whose agent
home preserved the full conversation.

The rollout logs one `response_item` event per conversation item (messages, reasoning,
custom_tool_call, custom_tool_call_output, ...) with ISO timestamps. The reconstructor
needs: records with ts_ms, response.output = [new custom_tool_call items], and
request.input containing the call outputs (to gate on executed calls). We synthesize one
record per assistant turn (a run of items up to the next tool outputs).

    python3 rollout_to_traj.py --rollout rollout-....jsonl --out raw_requests.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime


def iso_ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = []  # (ts_ms, payload)
    for line in open(args.rollout, errors="replace"):
        try:
            ev = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if ev.get("type") != "response_item":
            continue
        p = ev.get("payload") or {}
        ts = ev.get("timestamp")
        if p.get("type") in ("custom_tool_call", "custom_tool_call_output",
                             "function_call", "function_call_output"):
            items.append((iso_ms(ts) if ts else None, p))

    # one record per tool call, outputs accumulated into request.input so
    # collect_executions() sees them
    outputs = []
    n = 0
    with open(args.out, "w") as out:
        for ts, p in items:
            if p["type"] in ("custom_tool_call_output", "function_call_output"):
                outputs.append({"type": p["type"], "call_id": p.get("call_id")})
                continue
            rec = {"ts_ms": ts, "synthesized_from": "codex_rollout",
                   "request": {"input": list(outputs)},
                   "response": {"output": [p]}}
            out.write(json.dumps(rec) + "\n")
            n += 1
    # trailing record carrying any remaining outputs (so the last calls count as executed)
    with open(args.out, "a") as out:
        out.write(json.dumps({"ts_ms": items[-1][0] if items else None,
                              "synthesized_from": "codex_rollout",
                              "request": {"input": list(outputs)},
                              "response": {"output": []}}) + "\n")
    print(f"{n} tool-call records synthesized from {len(items)} rollout items")


if __name__ == "__main__":
    main()
