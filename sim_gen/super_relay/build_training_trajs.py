"""Build training trajectories from super_relay raw request logs.

Input:  logs/raw_requests.jsonl written by server.py — one record per API
        request: {request_id, ts_ms, session_id, api_format, model, status,
        request, response}.

Output: training_trajs.jsonl — one line per LEAF request, matching the
        super-relay format (see relevant_repos/training_trajs.jsonl):

    {"messages": [{"role": ..., "content": "<json-encoded blocks>"}, ...],
     "metadata": {"model": "openrouter/<model>",
                  "leaf_request_id": "<epoch-ms>-<12hex>",
                  "source_request_ids": [...all requests on the prefix chain...],
                  "merged_request_count": N,
                  "normalization": {"mid_system_merged": n, "same_role_merged": m}}}

How it works
------------
Agents resend the whole conversation with every API call, so a session of N
calls yields N requests where each request's message list extends the previous
one.  Within each (session_id) group we:

1. Canonicalize every request into an Anthropic-style message list
   (OpenAI-format records are converted: tool_calls -> tool_use,
   role="tool" -> user/tool_result, reasoning_content -> thinking).
2. Hash each message (cache_control stripped) so prefix checks are cheap.
3. A request R is a LEAF iff no other request has R's messages as a proper
   prefix.  Chains can fork (e.g. retries / parallel branches): every maximal
   branch produces its own line.
4. For each leaf, source_request_ids = all requests whose messages are a
   prefix of the leaf's (including the leaf itself), in timestamp order.
5. messages = packed system message ({"system": blocks, "tools": tools}) +
   the leaf request's messages + the leaf's response as the final assistant
   message.  Normalization then merges consecutive same-role messages and
   folds mid-conversation system messages into the adjacent user message
   (string-appended after the serialized JSON, as super-relay does).

Usage:
    python build_training_trajs.py \
        --raw-log logs/raw_requests.jsonl \
        --output logs/training_trajs.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

COMPACT = {"ensure_ascii": False, "separators": (",", ":")}


# ---------------------------------------------------------------------------
# Canonicalization: every record -> {system_blocks, tools, messages(list of
# {role, content: blocks-list}), response_blocks}
# ---------------------------------------------------------------------------


def _as_blocks(content) -> list:
    """Normalize message content into a list of Anthropic content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def canonicalize_anthropic(request: dict, response: dict) -> dict:
    system = request.get("system")
    if isinstance(system, str):
        system_blocks = [{"type": "text", "text": system}]
    else:
        system_blocks = system or []
    messages = [
        {"role": m["role"], "content": _as_blocks(m.get("content"))}
        for m in request.get("messages", [])
    ]
    return {
        "system_blocks": system_blocks,
        "tools": request.get("tools") or [],
        "messages": messages,
        "response_blocks": response.get("content", []),
    }


def _openai_tools_to_anthropic(tools: list) -> list:
    result = []
    for t in tools or []:
        fn = t.get("function", t)
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        })
    return result


def _openai_message_to_blocks(msg: dict) -> dict:
    """Convert one OpenAI chat message to an Anthropic-style message."""
    role = msg.get("role")
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }],
        }
    if role == "assistant":
        blocks: list = []
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            blocks.append({"type": "thinking", "thinking": reasoning, "signature": msg.get("signature", "")})
        content = msg.get("content")
        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    print(f"WARNING: unparseable tool_call arguments ({e}): {args[:200]!r}")
                    args = {"_raw": args}
            blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": args})
        return {"role": "assistant", "content": blocks}
    # user / system: text parts and images
    content = msg.get("content")
    if isinstance(content, list):
        blocks = []
        for part in content:
            if not isinstance(part, dict):
                blocks.append({"type": "text", "text": str(part)})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    media_type = header[len("data:"):].split(";")[0] or "image/png"
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            else:
                blocks.append({"type": "text", "text": part.get("text", "")})
        return {"role": role, "content": blocks}
    return {"role": role, "content": _as_blocks(content)}


def canonicalize_openai(request: dict, response: dict) -> dict:
    raw_messages = request.get("messages", [])
    # Leading system messages become the top-level system prompt.
    system_blocks: list = []
    body_start = 0
    for m in raw_messages:
        if m.get("role") == "system":
            system_blocks.extend(_as_blocks(m.get("content")))
            body_start += 1
        else:
            break

    messages: list = []
    for m in raw_messages[body_start:]:
        conv = _openai_message_to_blocks(m)
        # Group consecutive tool results into one user message (Anthropic style).
        if (
            messages
            and conv["role"] == "user"
            and messages[-1]["role"] == "user"
            and conv["content"]
            and conv["content"][0].get("type") == "tool_result"
            and messages[-1]["content"]
            and messages[-1]["content"][-1].get("type") == "tool_result"
        ):
            messages[-1]["content"].extend(conv["content"])
        else:
            messages.append(conv)

    choice = (response.get("choices") or [{}])[0]
    resp_msg = choice.get("message", {})
    response_blocks = _openai_message_to_blocks(resp_msg)["content"]

    return {
        "system_blocks": system_blocks,
        "tools": _openai_tools_to_anthropic(request.get("tools")),
        "messages": messages,
        "response_blocks": response_blocks,
    }


def _responses_item_to_message(item: dict) -> dict | None:
    """Convert one OpenAI Responses API item (input or output) to an Anthropic-style
    message, or None for items that carry no conversational content."""
    kind = item.get("type", "message")
    if kind == "message":
        role = item.get("role", "user")
        content = item.get("content")
        blocks: list = []
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            for part in content or []:
                if not isinstance(part, dict):
                    blocks.append({"type": "text", "text": str(part)})
                elif part.get("type") in ("input_text", "output_text", "text"):
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "input_image":
                    url = part.get("image_url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        header, _, data = url.partition(",")
                        media_type = header[len("data:"):].split(";")[0] or "image/png"
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": media_type, "data": data}})
                    else:
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
                else:
                    blocks.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
        return {"role": "assistant" if role == "assistant" else role, "content": blocks}
    if kind == "function_call":
        args = item.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                print(f"WARNING: unparseable function_call arguments ({e}): {args[:200]!r}")
                args = {"_raw": args}
        return {"role": "assistant", "content": [{
            "type": "tool_use", "id": item.get("call_id", item.get("id", "")),
            "name": item.get("name", ""), "input": args}]}
    if kind == "custom_tool_call":
        # codex's freeform tools (e.g. `exec`): input is a raw string, not JSON args.
        raw = item.get("input", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                args = {"input": raw}
        except json.JSONDecodeError:
            args = {"input": raw}
        return {"role": "assistant", "content": [{
            "type": "tool_use", "id": item.get("call_id", item.get("id", "")),
            "name": item.get("name", ""), "input": args}]}
    if kind in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output", "")
        if isinstance(output, list):   # [{type:"input_text", text}, ...] -> text blocks
            output = [{"type": "text", "text": p.get("text", "")} if isinstance(p, dict)
                      else {"type": "text", "text": str(p)} for p in output]
        return {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": item.get("call_id", ""),
            "content": output}]}
    if kind == "reasoning":
        summary = item.get("summary") or []
        text = "\n".join(p.get("text", "") for p in summary if isinstance(p, dict))
        if not text and not item.get("encrypted_content"):
            return None
        return {"role": "assistant", "content": [{
            "type": "thinking", "thinking": text,
            "signature": item.get("encrypted_content", "") or ""}]}
    # Hosted/unknown item kinds (web_search_call, local_shell_call, ...): keep the record
    # readable rather than dropping it silently.
    print(f"WARNING: unhandled responses item type {kind!r}; folding as text")
    return {"role": "assistant", "content": [{
        "type": "text", "text": json.dumps(item, ensure_ascii=False)}]}


def _responses_tools_to_anthropic(tools: list, prefix: str = "") -> list:
    """Responses-format tools are FLAT ({type:'function', name, ...}), unlike chat's nesting.
    codex additionally nests them in namespaces (an `additional_tools` input item); those
    flatten to dotted names ("functions.exec"), which is how the calls reference them."""
    result = []
    for t in tools or []:
        if t.get("type") == "namespace":
            result.extend(_responses_tools_to_anthropic(
                t.get("tools") or [], prefix=f"{prefix}{t.get('name', '')}."))
            continue
        if t.get("type") not in (None, "function", "custom"):
            continue   # hosted tools (web_search etc.) have no schema to carry
        schema = t.get("parameters")
        if not schema and t.get("format"):   # custom (freeform) tools carry a format instead
            schema = {"type": "custom", "format": t["format"]}
        result.append({
            "name": f"{prefix}{t.get('name', '')}",
            "description": t.get("description", ""),
            "input_schema": schema or {},
        })
    return result


def canonicalize_openai_responses(request: dict, response: dict) -> dict:
    """OpenAI Responses API (what codex speaks): instructions -> system, input items ->
    messages, output items -> the response blocks."""
    system_blocks: list = []
    instructions = request.get("instructions")
    if instructions:
        system_blocks.append({"type": "text", "text": instructions})

    raw_input = request.get("input")
    items = ([{"type": "message", "role": "user", "content": raw_input}]
             if isinstance(raw_input, str) else list(raw_input or []))

    # codex sends its tool definitions as `additional_tools` INPUT ITEMS (namespaced), not
    # as the top-level `tools` field. They are definitions, not conversation: they join the
    # tool list and never become messages.
    tools = _responses_tools_to_anthropic(request.get("tools"))
    kept: list = []
    for item in items:
        if item.get("type") == "additional_tools":
            tools.extend(_responses_tools_to_anthropic(item.get("tools") or []))
        else:
            kept.append(item)
    items = kept

    messages: list = []
    body_start = 0
    for item in items:   # leading system/developer messages join the system prompt
        if item.get("type", "message") == "message" and item.get("role") in ("system", "developer"):
            conv = _responses_item_to_message(item)
            if conv:
                system_blocks.extend(conv["content"])
            body_start += 1
        else:
            break
    for item in items[body_start:]:
        conv = _responses_item_to_message(item)
        if conv is None:
            continue
        if (
            messages
            and conv["role"] == messages[-1]["role"]
            and conv["content"]
            and messages[-1]["content"]
            and (
                (conv["content"][0].get("type") == "tool_result"
                 and messages[-1]["content"][-1].get("type") == "tool_result")
                or conv["role"] == "assistant"
            )
        ):
            # Consecutive assistant items (reasoning -> message -> function_call) are ONE
            # assistant turn in Anthropic form, exactly as consecutive tool results are one
            # user message. Deterministic, so prefix hashing still matches across requests.
            messages[-1]["content"].extend(conv["content"])
        else:
            messages.append(conv)

    response_blocks: list = []
    for item in response.get("output") or []:
        conv = _responses_item_to_message(item)
        if conv and conv["role"] == "assistant":
            response_blocks.extend(conv["content"])

    return {
        "system_blocks": system_blocks,
        "tools": tools,
        "messages": messages,
        "response_blocks": response_blocks,
    }


def canonicalize(record: dict) -> dict | None:
    """Canonicalize a raw log record; returns None for unusable records."""
    response = record.get("response")
    if record.get("status") != 200 or not isinstance(response, dict):
        return None
    # `.get("error")`, not `"error" in response`: a Responses API object always carries an
    # `error` FIELD (null on success), so membership rejected every successful codex record.
    if response.get("type") == "error" or response.get("error"):
        return None
    if record.get("api_format") == "openai":
        canon = canonicalize_openai(record["request"], response)
    elif record.get("api_format") == "openai_responses":
        canon = canonicalize_openai_responses(record["request"], response)
    else:
        canon = canonicalize_anthropic(record["request"], response)
    canon["request_id"] = record["request_id"]
    canon["ts_ms"] = record["ts_ms"]
    canon["model"] = record.get("model", "")
    # Provider prefix for metadata.model (reference file style:
    # "openrouter/anthropic/claude-sonnet-5"). Old logs without the field
    # default to openrouter, matching the reference traffic.
    canon["provider"] = record.get("provider", "openrouter")
    return canon


# ---------------------------------------------------------------------------
# Prefix chains / leaf detection
# ---------------------------------------------------------------------------


def _without_cache_control(obj):
    if isinstance(obj, dict):
        return {k: _without_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_without_cache_control(item) for item in obj]
    return obj


def message_hashes(messages: list[dict]) -> list[str]:
    """Per-message content hash (cache_control stripped) for cheap prefix checks."""
    hashes = []
    for m in messages:
        stripped = _without_cache_control({"role": m["role"], "content": m["content"]})
        raw = json.dumps(stripped, sort_keys=True, ensure_ascii=False, default=str)
        hashes.append(hashlib.sha256(raw.encode()).hexdigest())
    return hashes


def find_leaves(canons: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Return [(leaf_canon, source_canons)] for one session group.

    A request is a leaf iff its message list is not a proper prefix of any
    other request's.  Exact-duplicate requests (identical message lists, e.g.
    client retries) are collapsed: only the latest is a leaf candidate, but
    all duplicates appear in the sources.
    """
    for c in canons:
        c["_hashes"] = message_hashes(c["messages"])
    canons = sorted(canons, key=lambda c: (c["ts_ms"], c["request_id"]))

    def is_prefix(a: list[str], b: list[str]) -> bool:
        return len(a) <= len(b) and b[: len(a)] == a

    # Collapse exact duplicates (same full hash list): keep the latest occurrence.
    full_key = lambda c: "|".join(c["_hashes"])  # noqa: E731
    latest_for_key: dict[str, dict] = {}
    for c in canons:
        latest_for_key[full_key(c)] = c

    results = []
    for leaf in canons:
        if latest_for_key[full_key(leaf)] is not leaf:
            continue  # superseded duplicate
        is_leaf = not any(
            other is not leaf
            and len(other["_hashes"]) > len(leaf["_hashes"])
            and is_prefix(leaf["_hashes"], other["_hashes"])
            for other in canons
        )
        if not is_leaf:
            continue
        sources = [c for c in canons if is_prefix(c["_hashes"], leaf["_hashes"])]
        results.append((leaf, sources))
    return results


# ---------------------------------------------------------------------------
# Response enrichment
# ---------------------------------------------------------------------------


def _replay_matches_response(replay_blocks: list, response_blocks: list) -> bool:
    """Check that a replayed assistant message corresponds to a logged response.

    Clients (e.g. Claude Code) strip thinking blocks when replaying assistant
    turns, so we compare only tool_use ids and text content.
    """
    def summarize(blocks):
        ids = sorted(b.get("id", "") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use")
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        return ids, text

    r_ids, r_text = summarize(replay_blocks)
    s_ids, s_text = summarize(response_blocks)
    if r_ids or s_ids:
        return r_ids == s_ids
    return r_text == s_text


def enrich_assistant_messages(leaf: dict, sources: list[dict]) -> int:
    """Replace replayed assistant messages in the leaf's history with the
    corresponding logged responses from the chain's source requests.

    Rationale: agents replay prior assistant turns WITHOUT thinking blocks
    (Claude Code sends ``thinking: {"display": "omitted"}``), so the leaf
    history alone loses reasoning content.  The logged response of the source
    request with N history messages is the ground truth for the assistant
    message at index N of the leaf's history.

    Returns the number of messages enriched.
    """
    enriched = 0
    for src in sources:
        n = len(src["messages"])
        if src is leaf or n >= len(leaf["messages"]):
            continue
        msg = leaf["messages"][n]
        if msg["role"] != "assistant" or not src["response_blocks"]:
            continue
        if msg["content"] == src["response_blocks"]:
            continue
        if _replay_matches_response(msg["content"], src["response_blocks"]):
            msg["content"] = src["response_blocks"]
            enriched += 1
        else:
            print(
                f"WARNING: response of {src['request_id']} does not match the replayed "
                f"assistant message at index {n}; keeping the replayed version"
            )
    return enriched


# ---------------------------------------------------------------------------
# Output normalization + serialization
# ---------------------------------------------------------------------------


def build_trajectory(leaf: dict, source_canons: list[dict]) -> dict:
    enrich_assistant_messages(leaf, source_canons)
    sources = [c["request_id"] for c in source_canons]

    # 1. Assemble the full structured conversation.
    conv = [{"role": "system", "_packed": {"system": leaf["system_blocks"], "tools": leaf["tools"]}}]
    conv += [{"role": m["role"], "content": list(m["content"])} for m in leaf["messages"]]
    conv.append({"role": "assistant", "content": list(leaf["response_blocks"])})

    # 2. Merge consecutive same-role messages (structured, block-list concat).
    same_role_merged = 0
    merged: list[dict] = []
    for m in conv:
        if (
            merged
            and "_packed" not in m
            and "_packed" not in merged[-1]
            and merged[-1]["role"] == m["role"]
        ):
            merged[-1]["content"].extend(m["content"])
            same_role_merged += 1
        else:
            merged.append(m)

    # 3. Serialize content: block lists -> compact JSON strings.
    out_msgs: list[dict] = []
    for m in merged:
        if "_packed" in m:
            out_msgs.append({"role": "system", "content": json.dumps(m["_packed"], **COMPACT)})
        else:
            out_msgs.append({"role": m["role"], "content": json.dumps(m["content"], **COMPACT)})

    # 4. Fold mid-conversation system messages into the adjacent user message
    #    (string-append after the serialized JSON, matching super-relay output).
    mid_system_merged = 0
    final_msgs: list[dict] = [out_msgs[0]]
    pending_head: list[str] = []  # system text waiting for the NEXT user message
    for m in out_msgs[1:]:
        if m["role"] == "system":
            blocks = json.loads(m["content"])
            text = "\n".join(
                b.get("text", "") for b in blocks if isinstance(b, dict)
            ) if isinstance(blocks, list) else str(blocks)
            mid_system_merged += 1
            # Prefer the preceding user message; else hold for the next one.
            target = next(
                (fm for fm in reversed(final_msgs[1:]) if fm["role"] == "user"),
                None,
            )
            if target is final_msgs[-1] and target is not None:
                target["content"] = target["content"] + "\n\n" + text
            else:
                pending_head.append(text)
            continue
        if pending_head and m["role"] == "user":
            m = {"role": m["role"], "content": "\n\n".join(pending_head) + "\n\n" + m["content"]}
            pending_head = []
        final_msgs.append(m)
    if pending_head:
        print(f"WARNING: {len(pending_head)} trailing mid-system message(s) had no user message to merge into; appended as user")
        final_msgs.append({"role": "user", "content": "\n\n".join(pending_head)})

    model = leaf["model"]
    prefix = leaf.get("provider", "")
    if model and prefix and not model.startswith(f"{prefix}/"):
        model = f"{prefix}/{model}"

    return {
        "messages": final_msgs,
        "metadata": {
            "model": model,
            "leaf_request_id": leaf["request_id"],
            "source_request_ids": sources,
            "merged_request_count": len(sources),
            "normalization": {
                "mid_system_merged": mid_system_merged,
                "same_role_merged": same_role_merged,
            },
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_all(raw_log: Path, min_messages: int = 0, keep_toolless: bool = False) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    n_records = n_skipped = n_toolless = 0
    with open(raw_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            record = json.loads(line)
            canon = canonicalize(record)
            if canon is None:
                n_skipped += 1
                continue
            # Requests without tool definitions are agent housekeeping
            # (Claude Code topic detection, warmup, etc.), not agentic
            # trajectories — the reference training_trajs.jsonl contains
            # only tool-carrying conversations.  Skipped unless requested.
            if not canon["tools"] and not keep_toolless:
                n_toolless += 1
                continue
            groups[record.get("session_id", "anon")].append(canon)

    trajs = []
    for session_id, canons in groups.items():
        for leaf, sources in find_leaves(canons):
            traj = build_trajectory(leaf, sources)
            if len(traj["messages"]) < min_messages:
                continue
            trajs.append(traj)
    trajs.sort(key=lambda t: t["metadata"]["leaf_request_id"])
    print(
        f"read {n_records} records ({n_skipped} skipped: errors/non-200, "
        f"{n_toolless} skipped: no tool definitions), "
        f"{len(groups)} session group(s), {len(trajs)} trajectorie(s)"
    )
    return trajs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = Path(__file__).resolve().parent / "logs"
    parser.add_argument("--raw-log", default=str(default_dir / "raw_requests.jsonl"))
    parser.add_argument("--output", default=str(default_dir / "training_trajs.jsonl"))
    parser.add_argument("--min-messages", type=int, default=0,
                        help="skip trajectories with fewer than N messages")
    parser.add_argument("--keep-toolless", action="store_true",
                        help="also emit trajectories from requests without tool definitions "
                             "(agent housekeeping calls; skipped by default)")
    args = parser.parse_args()

    trajs = build_all(Path(args.raw_log), min_messages=args.min_messages,
                      keep_toolless=args.keep_toolless)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in trajs:
            f.write(json.dumps(t, **COMPACT) + "\n")
    print(f"wrote {len(trajs)} trajectories to {out}")


if __name__ == "__main__":
    main()
