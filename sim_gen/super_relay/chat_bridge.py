"""Responses-API ⇄ Chat-Completions bridge for chat-only upstreams (e.g. the AIDP gateway).

codex speaks ONLY the OpenAI Responses API (chat wire support was removed from the CLI in
Feb 2026), but some upstreams (ByteDance AIDP modelhub `/v2/crawl`) speak ONLY chat
completions. This module translates one codex turn each way:

  responses_to_chat(body)          — Responses request  -> chat payload
  chat_to_responses(chat, body)    — chat completion    -> Responses response object
  responses_sse(resp)              — Responses object   -> the SSE event bytes codex expects

Shapes are taken from REAL logged codex traffic (results/*/raw_requests.jsonl), not from API
docs: `additional_tools` developer items carrying namespaced `custom`/`function` tools,
`message` items with typed content parts, `custom_tool_call`/`custom_tool_call_output` and
`function_call`/`function_call_output` history items, and `reasoning` items with
`encrypted_content`.

Translation losses, by design:
  - history `reasoning` items are DROPPED (chat upstreams cannot accept them, and codex
    tolerates responses that carry no reasoning items — observed in real OpenAI traffic);
  - freeform `custom` tools become function tools with a single required string field
    `input`; calls are unwrapped back into `custom_tool_call` items so codex (and the
    trajectory log) still sees its native shapes.
"""
from __future__ import annotations

import json
import time
import uuid


def _part_text(parts) -> str:
    """Flatten a Responses content list (or plain string) to text."""
    if isinstance(parts, str):
        return parts
    out = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text"):
            out.append(p.get("text", ""))
        elif isinstance(p, str):
            out.append(p)
    return "".join(out)


def _collect_tools(body: dict) -> tuple[list[dict], set[str]]:
    """Flatten every tool the request grants into chat function specs.

    Returns (chat_tools, custom_tool_names). Tools arrive either at the top level
    (`body["tools"]`) or inside `additional_tools` input items, possibly nested one level
    in `namespace` groups.
    """
    flat: list[dict] = []

    def _walk(tools):
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "namespace":
                _walk(t.get("tools"))
            else:
                flat.append(t)

    _walk(body.get("tools"))
    for item in body.get("input") or [] if isinstance(body.get("input"), list) else []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            _walk(item.get("tools"))

    chat_tools, custom_names = [], set()
    for t in flat:
        name = t.get("name", "")
        if not name:
            continue
        if t.get("type") == "custom":
            custom_names.add(name)
            chat_tools.append({"type": "function", "function": {
                "name": name,
                "description": (t.get("description") or "") +
                               "\nPass the ENTIRE raw tool input as the `input` string field.",
                "parameters": {"type": "object",
                               "properties": {"input": {"type": "string"}},
                               "required": ["input"]},
            }})
        elif t.get("type") == "function":
            chat_tools.append({"type": "function", "function": {
                "name": name,
                "description": t.get("description") or "",
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            }})
        # anything else (web_search etc.) has no chat equivalent — omitted
    return chat_tools, custom_names


def responses_to_chat(body: dict) -> tuple[dict, set[str]]:
    """Translate a Responses request into a chat-completions payload."""
    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})

    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": items}]}]

    for item in items or []:
        if not isinstance(item, dict):
            continue
        typ = item.get("type") or ("message" if item.get("role") else None)
        if typ == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"      # chat upstreams may reject the developer role
            messages.append({"role": role, "content": _part_text(item.get("content"))})
        elif typ == "custom_tool_call":
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": item.get("call_id", ""), "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": json.dumps({"input": item.get("input", "")})},
            }]})
        elif typ == "function_call":
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": item.get("call_id", ""), "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments") or "{}"},
            }]})
        elif typ in ("custom_tool_call_output", "function_call_output"):
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                             "content": _part_text(item.get("output"))})
        # additional_tools handled by _collect_tools; reasoning items dropped (see module doc)

    chat_tools, custom_names = _collect_tools(body)
    payload: dict = {"model": body.get("model", ""), "stream": False, "messages": messages}
    if chat_tools:
        payload["tools"] = chat_tools
        tc = body.get("tool_choice")
        if isinstance(tc, str) and tc in ("auto", "none", "required"):
            payload["tool_choice"] = tc
        if isinstance(body.get("parallel_tool_calls"), bool):
            payload["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("max_output_tokens"):
        payload["max_tokens"] = body["max_output_tokens"]
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        payload["reasoning_effort"] = effort   # stripped on upstream validation error
    return payload, custom_names


# Fields a strict gateway may reject; stripped one by one on validation errors.
OPTIONAL_CHAT_FIELDS = ("reasoning_effort", "parallel_tool_calls", "tool_choice")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def chat_to_responses(chat: dict, body: dict, custom_names: set[str]) -> dict:
    """Translate a chat completion into a Responses response object (codex-shaped)."""
    now = int(time.time())
    choice = (chat.get("choices") or [{}])[0]
    msg = choice.get("message") or {}

    output: list[dict] = []
    content = msg.get("content")
    if content:
        output.append({"id": _new_id("msg"), "type": "message", "status": "completed",
                       "role": "assistant", "phase": "commentary",
                       "content": [{"type": "output_text", "annotations": [],
                                    "logprobs": [], "text": content}]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name, args = fn.get("name", ""), fn.get("arguments") or "{}"
        if name in custom_names:
            try:
                tool_input = json.loads(args).get("input", "")
            except (json.JSONDecodeError, AttributeError):
                tool_input = args      # model emitted the raw string: pass it through
            output.append({"id": _new_id("ctc"), "type": "custom_tool_call",
                           "status": "completed", "call_id": tc.get("id", _new_id("call")),
                           "name": name, "input": tool_input})
        else:
            output.append({"id": _new_id("fc"), "type": "function_call",
                           "status": "completed", "call_id": tc.get("id", _new_id("call")),
                           "name": name, "arguments": args})

    u = chat.get("usage") or {}
    usage = {
        "input_tokens": u.get("prompt_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "cache_write_tokens": 0,
        },
        "output_tokens": u.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens":
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        },
        "total_tokens": u.get("total_tokens", 0),
    }
    return {
        "id": _new_id("resp"), "object": "response", "created_at": now,
        "completed_at": now, "status": "completed", "error": None,
        "incomplete_details": None, "instructions": body.get("instructions"),
        "model": chat.get("model") or body.get("model", ""),
        "output": output, "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": None, "reasoning": body.get("reasoning"),
        "store": body.get("store", False), "temperature": None, "top_p": None,
        "text": body.get("text") or {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"), "tools": body.get("tools") or [],
        "truncation": "disabled", "usage": usage, "metadata": {}, "user": None,
        "background": False, "max_output_tokens": body.get("max_output_tokens"),
        "service_tier": "default",
    }


def responses_sse(resp: dict) -> list[bytes]:
    """The SSE event sequence for one completed response, matching upstream framing
    (`event:` + `data:` lines; codex keys off data["type"] == "response.completed")."""
    seq = 0

    def ev(event_type: str, data: dict) -> bytes:
        nonlocal seq
        seq += 1
        payload = {"type": event_type, "sequence_number": seq, **data}
        return (f"event: {event_type}\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n").encode()

    events = [ev("response.created",
                 {"response": {**resp, "status": "in_progress", "output": [],
                               "usage": None, "completed_at": None}})]
    for i, item in enumerate(resp.get("output", [])):
        events.append(ev("response.output_item.added",
                         {"output_index": i, "item": {**item, "status": "in_progress"}
                          if item.get("type") == "message" else item}))
        if item.get("type") == "message":
            for part in item.get("content", []):
                events.append(ev("response.output_text.delta",
                                 {"item_id": item["id"], "output_index": i,
                                  "content_index": 0, "delta": part.get("text", ""),
                                  "logprobs": []}))
        events.append(ev("response.output_item.done", {"output_index": i, "item": item}))
    events.append(ev("response.completed", {"response": resp}))
    return events
