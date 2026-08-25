#!/usr/bin/env python3
"""Synthesize an agent CLI session (Claude Code or codex) from a relay trajectory, for
resuming runs whose pods died before agent_home.tgz was mirrored.

The trajectory's LAST successful request carries the full conversation (Anthropic:
request.messages; codex: request.input items). We rebuild the CLI's own session file so
`claude --continue` / `codex exec resume --last` picks the conversation up where the pod
died. Formats copied from real session files (references studied 2026-08-21):

Claude Code (~/.claude/projects/-/<session-uuid>.jsonl):
    {"type":"queue-operation","operation":"enqueue","timestamp":...,"sessionId":...,
     "content": <first user text>}
    {"type":"queue-operation","operation":"dequeue",...}
    {"parentUuid": <prev uuid or None>, "isSidechain": false, "promptId": <uuid>,
     "type": "user"|"assistant", "message": {...verbatim from the trajectory...},
     "uuid": <uuid>, "timestamp": ..., "permissionMode": "default",
     "promptSource": "sdk", "userType": "external", "entrypoint": "sdk-cli",
     "cwd": "/", "sessionId": ..., "version": "2.1.216", "gitBranch": "HEAD"}
    {"type":"last-prompt","lastPrompt": <first user text[:60]>, "leafUuid": <last uuid>,
     "sessionId": ...}

codex (~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl):
    {"timestamp":...,"type":"session_meta","payload":{...copied from a reference, with
     fresh id/timestamp...}}
    {"timestamp":...,"type":"response_item","payload": <item verbatim>} per input item

Usage:
    python3 synth_agent_session.py --agent claude --traj results/<label>/raw_requests.jsonl \
        --out-home <dir>   # writes <dir>/agent/.claude/... ; tar it as agent_home.tgz
"""
from __future__ import annotations

import argparse
import json
import tarfile
import uuid as uuidlib
from datetime import datetime, timezone
from pathlib import Path


def last_conversation(traj: Path, agent: str):
    """(ts_ms, conversation) from the last successful request of the trajectory."""
    best = None
    for line in traj.open(errors="replace"):
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        req = rec.get("request") or {}
        conv = req.get("messages") if agent == "claude" else req.get("input")
        if isinstance(conv, list) and conv:
            best = (rec.get("ts_ms"), conv)
    if best is None:
        raise SystemExit(f"no conversation found in {traj}")
    return best


def iso(ts_ms: float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def synth_claude(conv: list, ts_ms: int, home: Path) -> None:
    sid = str(uuidlib.uuid4())
    proj = home / "agent" / ".claude" / "projects" / "-"
    proj.mkdir(parents=True, exist_ok=True)
    out = proj / f"{sid}.jsonl"
    first_text = ""
    for m in conv:
        if m.get("role") == "user":
            c = m.get("content")
            first_text = c if isinstance(c, str) else next(
                (b.get("text", "") for b in c if isinstance(b, dict)
                 and b.get("type") == "text"), "")
            break
    events = [
        {"type": "queue-operation", "operation": "enqueue", "timestamp": iso(ts_ms),
         "sessionId": sid, "content": first_text},
        {"type": "queue-operation", "operation": "dequeue", "timestamp": iso(ts_ms),
         "sessionId": sid},
    ]
    prev = None
    prompt_id = str(uuidlib.uuid4())
    t = ts_ms
    for m in conv:
        # Claude Code's session loader requires content as an ARRAY of block objects
        # (it probes `"tool_use_id" in v` on each item — a plain string crashes the CLI,
        # measured 2026-08-21); the API accepts both, so normalize.
        m = dict(m)
        if isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"]}]
        elif isinstance(m.get("content"), list):
            # tool_result blocks with string content (377 in nut_thread's trajectory)
            # need the same block-array normalization one level down
            blocks = []
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result" \
                        and isinstance(b.get("content"), str):
                    b = {**b, "content": [{"type": "text", "text": b["content"]}]}
                blocks.append(b)
            m["content"] = blocks
        uid = str(uuidlib.uuid4())
        events.append({
            "parentUuid": prev, "isSidechain": False, "promptId": prompt_id,
            "type": "user" if m.get("role") == "user" else "assistant",
            "message": m, "uuid": uid, "timestamp": iso(t),
            "permissionMode": "default", "promptSource": "sdk",
            "userType": "external", "entrypoint": "sdk-cli", "cwd": "/",
            "sessionId": sid, "version": "2.1.216", "gitBranch": "HEAD"})
        prev = uid
        t += 1000
    events.append({"type": "last-prompt", "lastPrompt": first_text[:200],
                   "leafUuid": prev, "sessionId": sid})
    with out.open("w") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"claude session synthesized: {out} ({len(events)} events, "
          f"{len(conv)} messages)")


def synth_codex(conv: list, ts_ms: int, home: Path, meta_ref: dict) -> None:
    day = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    sid = str(uuidlib.uuid4())
    d = home / "agent" / ".codex" / "sessions" / day.strftime("%Y/%m/%d")
    d.mkdir(parents=True, exist_ok=True)
    stamp = day.strftime("%Y-%m-%dT%H-%M-%S")
    out = d / f"rollout-{stamp}-{sid}.jsonl"
    meta = dict(meta_ref)
    meta["id"] = sid
    meta["timestamp"] = iso(ts_ms)
    t = ts_ms
    with out.open("w") as fh:
        fh.write(json.dumps({"timestamp": iso(ts_ms), "type": "session_meta",
                             "payload": meta}, ensure_ascii=False) + "\n")
        for item in conv:
            fh.write(json.dumps({"timestamp": iso(t), "type": "response_item",
                                 "payload": item}, ensure_ascii=False) + "\n")
            t += 500
    print(f"codex rollout synthesized: {out} ({len(conv)} items)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["claude", "codex"])
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out-home", required=True,
                    help="directory to become the tarred home (creates agent/... inside)")
    ap.add_argument("--codex-meta-ref", default=None,
                    help="a real rollout jsonl to copy session_meta from (codex only)")
    ap.add_argument("--make-tgz", default=None, help="also write agent_home.tgz here")
    args = ap.parse_args()

    ts_ms, conv = last_conversation(Path(args.traj), args.agent)
    home = Path(args.out_home)
    if args.agent == "claude":
        synth_claude(conv, ts_ms or 0, home)
    else:
        if not args.codex_meta_ref:
            raise SystemExit("--codex-meta-ref required for codex")
        meta = None
        for line in open(args.codex_meta_ref, errors="replace"):
            ev = json.loads(line)
            if ev.get("type") == "session_meta":
                meta = ev["payload"]
                break
        synth_codex(conv, ts_ms or 0, home, meta or {})
    if args.make_tgz:
        with tarfile.open(args.make_tgz, "w:gz") as tf:
            tf.add(home / "agent", arcname="agent")
        print(f"wrote {args.make_tgz}")


if __name__ == "__main__":
    main()
