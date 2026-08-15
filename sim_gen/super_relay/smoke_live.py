"""Live smoke test: drive an agentic 3-request chain through the relay.

Simulates what Claude Code does — each request resends the full conversation —
using Anthropic Messages format with a tool, against a running server.py.
Then run build_training_trajs.py and print the resulting trajectory summary.

Run (server must be up):
    /home/tiger/cap-x/.venv/bin/python smoke_live.py --port 8118
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent

TOOLS = [{
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression and return the result.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}]
SYSTEM = [{"type": "text", "text": "You are a terse assistant. Use the calculator tool for any arithmetic."}]
MODEL = "anthropic/claude-haiku-4.5"


def call(base, messages, stream=False):
    body = {
        "model": MODEL,
        "max_tokens": 4096,
        "system": SYSTEM,
        "tools": TOOLS,
        "messages": messages,
        "stream": stream,
    }
    headers = {"x-api-key": "smoke-session-token", "anthropic-version": "2023-06-01"}
    if not stream:
        r = httpx.post(f"{base}/v1/messages", json=body, headers=headers, timeout=180)
        r.raise_for_status()
        return r.json()
    # streaming: parse the SSE burst back into a response dict
    content = []
    stop_reason = None
    with httpx.stream("POST", f"{base}/v1/messages", json=body, headers=headers, timeout=180) as r:
        r.raise_for_status()
        current = None
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            et = ev.get("type")
            if et == "content_block_start":
                current = ev["content_block"]
            elif et == "content_block_delta":
                d = ev["delta"]
                if d["type"] == "text_delta":
                    current["text"] = current.get("text", "") + d["text"]
                elif d["type"] == "thinking_delta":
                    current["thinking"] = current.get("thinking", "") + d["thinking"]
                elif d["type"] == "input_json_delta":
                    current["input"] = json.loads(d["partial_json"])
            elif et == "content_block_stop":
                content.append(current)
            elif et == "message_delta":
                stop_reason = ev["delta"].get("stop_reason")
    return {"content": content, "stop_reason": stop_reason, "role": "assistant"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8118)
    args = parser.parse_args()
    base = f"http://localhost:{args.port}"

    print(f"health: {httpx.get(base + '/health').json()}")

    # ── request 1: user asks a question that requires the tool ──
    messages = [{"role": "user", "content": [{"type": "text", "text": "What is 137 * 249? Then I'll have a follow-up."}]}]
    r1 = call(base, messages)
    print(f"\nturn 1 stop_reason={r1['stop_reason']}")
    print("turn 1 blocks:", [(b["type"], str(b.get("text") or b.get("input"))[:80]) for b in r1["content"]])
    tool_use = next((b for b in r1["content"] if b["type"] == "tool_use"), None)
    if tool_use is None:
        sys.exit("FAIL: model did not call the calculator tool on turn 1")

    # ── request 2: send tool result back (full conversation resent) ──
    expr = tool_use["input"].get("expression", "137*249")
    result = eval(expr.replace("^", "**"))  # trusted arithmetic from our own smoke test
    messages = messages + [
        {"role": "assistant", "content": r1["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": str(result)}]},
    ]
    r2 = call(base, messages)
    print(f"\nturn 2 stop_reason={r2['stop_reason']}")
    print("turn 2 blocks:", [(b["type"], str(b.get("text"))[:120]) for b in r2["content"]])

    # ── request 3 (streamed): follow-up question ──
    messages = messages + [
        {"role": "assistant", "content": r2["content"]},
        {"role": "user", "content": [{"type": "text", "text": "Now divide that by 3 (round to 2 decimals). No tool needed, just answer."}]},
    ]
    r3 = call(base, messages, stream=True)
    print(f"\nturn 3 (streamed) stop_reason={r3['stop_reason']}")
    print("turn 3 blocks:", [(b["type"], str(b.get("text") or b.get("input"))[:120]) for b in r3["content"]])

    # ── build trajectories ──
    print("\n--- building trajectories ---")
    proc = subprocess.run(
        [sys.executable, str(HERE / "build_training_trajs.py")],
        capture_output=True, text=True,
    )
    print(proc.stdout, proc.stderr)
    if proc.returncode != 0:
        sys.exit("merger failed")

    out = HERE / "logs" / "training_trajs.jsonl"
    trajs = [json.loads(l) for l in out.read_text().splitlines()]
    print(f"{len(trajs)} trajectory line(s):")
    for tr in trajs:
        md = tr["metadata"]
        roles = [m["role"] for m in tr["messages"]]
        print(f"  leaf={md['leaf_request_id']} merged={md['merged_request_count']} "
              f"norm={md['normalization']} model={md['model']}")
        print(f"  roles: {roles}")
        print(f"  last assistant: {tr['messages'][-1]['content'][:200]}")

    ok = (
        len(trajs) == 1
        and trajs[0]["metadata"]["merged_request_count"] == 3
        and [m["role"] for m in trajs[0]["messages"]]
        == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
    )
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
