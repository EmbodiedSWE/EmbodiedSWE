"""Self-test for build_training_trajs.py using a synthetic raw request log.

Simulates an agentic session the way Claude Code / any agent produces it
(each request resends the full conversation), including:
- a 3-request chain ending in a leaf,
- a fork (two leaves sharing a prefix, like lines 0/1 of the reference file),
- an exact-duplicate retry (must be collapsed into sources, not emitted twice),
- an errored request (must be skipped),
- an OpenAI-format record with a mid-conversation system message
  (must be folded into the adjacent user message and counted),
- consecutive same-role messages (must be merged and counted).

Also validates the output schema against relevant_repos/training_trajs.jsonl.

Run:  python3 test_merge.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "training_trajs.jsonl"

SYSTEM = [{"type": "text", "text": "You are a helpful agent.", "cache_control": {"type": "ephemeral"}}]
TOOLS = [{"name": "Bash", "description": "Run a command", "input_schema": {"type": "object"}}]


def rid(ts, suffix):
    return f"{ts}-{suffix:012x}"


def anthropic_record(request_id, session, messages, response_content, status=200):
    return {
        "request_id": request_id,
        "ts_ms": int(request_id.split("-")[0]),
        "session_id": session,
        "api_format": "anthropic",
        "model": "anthropic/claude-sonnet-5",
        "status": status,
        "request": {"model": "anthropic/claude-sonnet-5", "system": SYSTEM, "tools": TOOLS, "messages": messages},
        "response": {
            "type": "message", "role": "assistant", "model": "anthropic/claude-sonnet-5",
            "content": response_content, "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        } if status == 200 else {"type": "error", "error": {"type": "api_error", "message": "boom"}},
    }


def main():
    t = 1783990000000
    u1 = {"role": "user", "content": [{"type": "text", "text": "list files"}]}
    a1 = [{"type": "text", "text": "Listing."},
          {"type": "tool_use", "id": "toolu_001", "name": "Bash", "input": {"command": "ls"}}]
    u2 = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_001", "content": "a.txt b.txt"}]}
    a2 = [{"type": "thinking", "thinking": "two files", "signature": ""},
          {"type": "text", "text": "You have a.txt and b.txt."}]
    u3a = {"role": "user", "content": [{"type": "text", "text": "delete a.txt"}]}
    a3a = [{"type": "tool_use", "id": "toolu_002", "name": "Bash", "input": {"command": "rm a.txt"}}]
    u3b = {"role": "user", "content": [{"type": "text", "text": "read b.txt"}]}
    a3b = [{"type": "text", "text": "b.txt says hello."}]

    records = [
        # chain request 1
        anthropic_record(rid(t + 0, 0xA1), "sessA", [u1], a1),
        # retry of request 1 (identical messages) — must collapse
        anthropic_record(rid(t + 1000, 0xA2), "sessA", [u1], a1),
        # chain request 2
        anthropic_record(rid(t + 2000, 0xA3), "sessA",
                         [u1, {"role": "assistant", "content": a1}, u2], a2),
        # fork branch A (leaf 1)
        anthropic_record(rid(t + 3000, 0xA4), "sessA",
                         [u1, {"role": "assistant", "content": a1}, u2,
                          {"role": "assistant", "content": a2}, u3a], a3a),
        # fork branch B (leaf 2)
        anthropic_record(rid(t + 4000, 0xA5), "sessA",
                         [u1, {"role": "assistant", "content": a1}, u2,
                          {"role": "assistant", "content": a2}, u3b], a3b),
        # errored request — must be skipped entirely
        anthropic_record(rid(t + 5000, 0xA6), "sessA", [u1], [], status=500),
        # toolless housekeeping request (e.g. CC topic detection) — must be
        # skipped by default (no tool definitions)
        {
            "request_id": rid(t + 5500, 0xA7), "ts_ms": t + 5500, "session_id": "sessA",
            "api_format": "anthropic", "model": "anthropic/claude-sonnet-5", "status": 200,
            "request": {"model": "anthropic/claude-sonnet-5",
                        "system": "Analyze if this message indicates a new conversation topic.",
                        "messages": [{"role": "user", "content": "list files"}]},
            "response": {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "{}"}],
                         "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}},
        },
        # separate OpenAI-format session with a mid-conversation system message
        # and consecutive same-role user messages
        {
            "request_id": rid(t + 6000, 0xB1), "ts_ms": t + 6000, "session_id": "sessB",
            "api_format": "openai", "model": "openai/gpt-5.2", "status": 200,
            "request": {
                "model": "openai/gpt-5.2",
                "tools": [{"type": "function", "function": {
                    "name": "tell_joke", "description": "Tell a joke",
                    "parameters": {"type": "object", "properties": {}}}}],
                "messages": [
                    {"role": "system", "content": "You are concise."},
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "are you there?"},
                    {"role": "assistant", "content": "Yes."},
                    {"role": "system", "content": "<reminder>be brief</reminder>"},
                    {"role": "user", "content": "ok tell me a joke"},
                ],
            },
            "response": {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Why did..."},
                                      "finish_reason": "stop"}]},
        },
    ]

    tmp = Path(tempfile.mkdtemp(prefix="super_relay_test_"))
    raw = tmp / "raw_requests.jsonl"
    out = tmp / "training_trajs.jsonl"
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")

    proc = subprocess.run(
        [sys.executable, str(HERE / "build_training_trajs.py"), "--raw-log", str(raw), "--output", str(out)],
        capture_output=True, text=True,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr)
        sys.exit(f"merger failed rc={proc.returncode}")

    trajs = [json.loads(l) for l in out.read_text().splitlines()]
    failures = []

    def check(cond, desc):
        print(("PASS " if cond else "FAIL ") + desc)
        if not cond:
            failures.append(desc)

    # --- structural expectations -------------------------------------------
    check(len(trajs) == 3, f"3 trajectories emitted (2 fork leaves + 1 openai) — got {len(trajs)}")

    sessA = [tr for tr in trajs if tr["metadata"]["model"] == "openrouter/anthropic/claude-sonnet-5"]
    sessB = [tr for tr in trajs if tr["metadata"]["model"] == "openrouter/openai/gpt-5.2"]
    check(len(sessA) == 2 and len(sessB) == 1, "trajectories grouped per session/model")

    leafA1 = next(tr for tr in sessA if tr["metadata"]["leaf_request_id"].endswith(f"{0xA4:012x}"))
    leafA2 = next(tr for tr in sessA if tr["metadata"]["leaf_request_id"].endswith(f"{0xA5:012x}"))

    # fork: each leaf lists shared prefix (incl. retry duplicate) + itself
    for leaf, own in [(leafA1, 0xA4), (leafA2, 0xA5)]:
        srcs = leaf["metadata"]["source_request_ids"]
        check(len(srcs) == 4, f"leaf {own:x}: 4 sources (req1, retry, req2, self) — got {len(srcs)}")
        check(leaf["metadata"]["leaf_request_id"] == srcs[-1], f"leaf {own:x}: leaf id is last source")
        check(leaf["metadata"]["merged_request_count"] == len(srcs), f"leaf {own:x}: merged_request_count matches")
    check(
        leafA1["metadata"]["leaf_request_id"] not in leafA2["metadata"]["source_request_ids"],
        "fork leaves do not contain each other",
    )

    # errored request must not appear anywhere
    err_id = rid(t + 5000, 0xA6)
    check(
        all(err_id not in tr["metadata"]["source_request_ids"] for tr in trajs),
        "errored request excluded from all sources",
    )

    # toolless housekeeping request must not appear anywhere
    toolless_id = rid(t + 5500, 0xA7)
    check(
        all(toolless_id not in tr["metadata"]["source_request_ids"] for tr in trajs),
        "toolless housekeeping request excluded (no tool definitions)",
    )
    check(
        all(json.loads(tr["messages"][0]["content"])["tools"] for tr in trajs),
        "every emitted trajectory carries tool definitions",
    )

    # message encoding: system message 0 packs system+tools; content is JSON string
    msgs = leafA1["messages"]
    sys0 = json.loads(msgs[0]["content"])
    check(set(sys0.keys()) == {"system", "tools"}, "system message packs {system, tools}")
    check(sys0["tools"] == TOOLS, "tools preserved in system message")
    check(msgs[0]["role"] == "system", "first message role=system")
    roles = [m["role"] for m in msgs]
    check(roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
          f"leaf A4 role sequence — got {roles}")
    last = json.loads(msgs[-1]["content"])
    check(last == a3a, "final assistant message == leaf response content")
    a_blocks = json.loads(msgs[2]["content"])
    check(any(b["type"] == "tool_use" for b in a_blocks), "tool_use blocks preserved")

    # openai record: normalization counts
    trB = sessB[0]
    norm = trB["metadata"]["normalization"]
    check(norm["mid_system_merged"] == 1, f"mid_system_merged==1 — got {norm['mid_system_merged']}")
    check(norm["same_role_merged"] == 1, f"same_role_merged==1 — got {norm['same_role_merged']}")
    rolesB = [m["role"] for m in trB["messages"]]
    check("system" not in rolesB[1:], "no mid-conversation system messages remain")
    # the reminder text must have been folded into a user message
    check(any("be brief" in m["content"] for m in trB["messages"] if m["role"] == "user"),
          "mid-system text folded into a user message")
    sysB = json.loads(trB["messages"][0]["content"])
    check(sysB["system"][0]["text"] == "You are concise.", "leading openai system message became top-level system")

    # --- schema parity with the reference file ------------------------------
    if REFERENCE.exists():
        with open(REFERENCE) as f:
            ref = json.loads(f.readline())
        check(set(ref.keys()) == set(trajs[0].keys()), "top-level keys match reference")
        check(set(ref["metadata"].keys()) == set(trajs[0]["metadata"].keys()), "metadata keys match reference")
        check(
            set(ref["metadata"]["normalization"].keys())
            == set(trajs[0]["metadata"]["normalization"].keys()),
            "normalization keys match reference",
        )
        check(
            all(set(m.keys()) == {"role", "content"} and isinstance(m["content"], str)
                for tr in trajs for m in tr["messages"]),
            "all messages are {role, content:str} like reference",
        )
        import re
        check(
            all(re.fullmatch(r"\d{13}-[0-9a-f]{12}", i)
                for tr in trajs for i in tr["metadata"]["source_request_ids"]),
            "request id format matches reference (<13-digit ms>-<12 hex>)",
        )

    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) FAILED")
    print(f"All checks passed. Output kept at {out}")


if __name__ == "__main__":
    main()
