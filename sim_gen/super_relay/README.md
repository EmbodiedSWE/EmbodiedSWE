# super_relay — Claude Code trajectory logging relay

A super-relay-style API proxy for logging Claude Code (cc) agent trajectories.
Point CC agents at the relay; it forwards to the real Anthropic API using the
agent's own credentials (OAuth token / API key — the relay needs no key of its
own) and logs every request/response.  Trajectories are exported in the format
of `relevant_repos/training_trajs.jsonl` (one JSONL line per **leaf request**,
with `leaf_request_id` / `source_request_ids` / `merged_request_count` /
`normalization` metadata).

## Spawning CC agents with logging (main workflow)

```bash
VENV=/home/tiger/cap-x/.venv/bin/python

# 1. start the relay (defaults: upstream https://api.anthropic.com/v1,
#    auth passthrough, logs to ./logs/raw_requests.jsonl)
$VENV server.py --port 8118

# 2. spawn CC agents pointed at it (works with claude -p, the Agent SDK, etc.)
source ~/.claude_oauth_env   # provides CLAUDE_CODE_OAUTH_TOKEN
ANTHROPIC_BASE_URL=http://127.0.0.1:8118 ANTHROPIC_API_KEY= \
    claude -p "your task" --dangerously-skip-permissions
# NOTE: use 127.0.0.1, not localhost — CC resolves localhost to ::1 (IPv6)
# and gets ConnectionRefused since uvicorn binds IPv4.

# 3. export trajectories any time (offline, idempotent, safe while relay runs)
$VENV build_training_trajs.py \
    --raw-log logs/raw_requests.jsonl --output logs/training_trajs.jsonl
```

Concurrent agents are separated automatically: sessions are grouped by CC's
per-session `metadata.user_id` (falling back to a hash of the auth token), so
many CC processes sharing one OAuth token still produce distinct trajectories.

CC housekeeping calls (topic detection etc., sent without tool definitions)
are skipped by the merger by default — every emitted trajectory carries the
full tool definitions in its system message, like the reference file.  Pass
`--keep-toolless` to emit them anyway.

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI relay. `POST /v1/messages` (Anthropic format, incl. `count_tokens` passthrough) and `POST /chat/completions` + `/v1/chat/completions` (OpenAI format, for OpenAI-compatible upstreams). Logs every request/response pair to `logs/raw_requests.jsonl` with a `<epoch-ms>-<12hex>` request id. |
| `build_training_trajs.py` | Offline merger: raw log → `training_trajs.jsonl`. Prefix-tree leaf detection, fork handling, retry collapsing, same-role merge, mid-system fold, super-relay content serialization. |
| `test_merge.py` | Offline self-test on a synthetic agentic session (chain + fork + retry + error + OpenAI mid-system case); validates schema parity with the reference file. |
| `mock_upstream.py` | Deterministic mock upstream (Anthropic + OpenAI endpoints) for end-to-end testing without credentials. |
| `smoke_live.py` | End-to-end smoke: drives a 3-request agentic chain (incl. a streamed SSE turn with tool_use) through the running relay, then builds and checks the trajectory. |

## Auth modes

- **Passthrough (default)** — the client's `x-api-key` / `Authorization` /
  `anthropic-beta` headers are forwarded verbatim.  This is what CC needs: its
  OAuth token goes straight through to api.anthropic.com.
- **Relay-owned key** — pass `--api-key <key>`; the relay substitutes it and
  the client token is only used as a session id.  Use with
  `--upstream-base https://openrouter.ai/api/v1` to front OpenRouter for
  keyless clients (the reference `training_trajs.jsonl` came from
  OpenRouter-routed traffic).

`metadata.model` in exported trajectories is prefixed by the upstream provider
(`anthropic/...` or `openrouter/...`), matching the reference file's style.

## Streaming

- `/v1/messages` with `stream=true`: upstream call is made non-streaming;
  `event: ping` heartbeats are sent while waiting (prevents CC timeouts), then
  the full response is emitted as a standard Anthropic SSE burst.  Mirrors
  swalm's AnthropicRelay; guarantees the complete response is logged.
- `/chat/completions` with `stream=true`: chunks passed through unmodified
  while deltas are accumulated into a full response object for the log.

## Tests

```bash
# offline (no network, no credentials)
python3 test_merge.py

# end-to-end against the mock upstream
$VENV mock_upstream.py --port 8119 &
$VENV server.py --port 8118 --upstream-base http://127.0.0.1:8119/v1 &
$VENV smoke_live.py --port 8118

# end-to-end with a real CC agent (verified working)
$VENV server.py --port 8118 &
source ~/.claude_oauth_env
ANTHROPIC_BASE_URL=http://127.0.0.1:8118 ANTHROPIC_API_KEY= \
    claude -p "Create hello.txt with 'hi', read it back." --dangerously-skip-permissions
$VENV build_training_trajs.py
```

## How trajectory reconstruction works

Agents resend the full conversation on every API call, so an N-turn session
produces N requests where each request's message list is a prefix of the next.
The merger:

1. Canonicalizes every logged request to Anthropic-style content blocks
   (OpenAI records converted: `tool_calls`→`tool_use`, `role:"tool"`→
   `tool_result`, `reasoning_content`→`thinking`).
2. Hashes each message (with `cache_control` stripped) and finds **leaves** —
   requests that are not a proper prefix of any other request. Chains can fork
   (retries / parallel branches): every maximal branch emits its own line.
3. Per leaf, emits `system` message 0 = `{"system": blocks, "tools": tools}`,
   the leaf's messages, and the leaf's response as the final assistant message.
   All content is serialized as compact JSON strings, matching super-relay.
4. Normalization: consecutive same-role messages are merged
   (`same_role_merged`); mid-conversation `system` messages are folded into
   the adjacent user message (`mid_system_merged`).
5. `source_request_ids` = all requests on the leaf's prefix chain (leaf last);
   errored/non-200 requests are excluded; exact-duplicate retries are
   collapsed into the source list.
