"""Super-relay clone: LLM API relay with per-request trajectory logging.

Primary use case: log Claude Code (cc) trajectories.  Point Claude Code at the
relay (``ANTHROPIC_BASE_URL=http://<host>:<port>``) and it forwards to the real
Anthropic API by default, passing the client's own credentials through
(OAuth token / ANTHROPIC_API_KEY) — the relay needs no key of its own.

Routes (both API formats):

- ``POST /v1/messages``           — Anthropic Messages format, forwarded as-is
                                    to ``<upstream_base>/messages``
                                    (default https://api.anthropic.com/v1).
- ``POST /chat/completions`` and
  ``POST /v1/chat/completions``   — OpenAI chat format, forwarded to
                                    ``<upstream_base>/chat/completions`` (for
                                    OpenAI-compatible upstreams, e.g.
                                    ``--upstream-base https://openrouter.ai/api/v1``).

Auth: by default the client's auth headers (``x-api-key`` / ``Authorization`` /
``anthropic-beta``) are forwarded verbatim.  Pass ``--api-key`` to substitute a
relay-owned key instead (e.g. when fronting OpenRouter for keyless clients);
in that mode the client token is only used as a session id for grouping.

Every request/response pair is appended to ``logs/raw_requests.jsonl`` with a
super-relay-style request id (``<epoch-ms>-<12 hex>``).  Trajectories in the
training format (one line per leaf request) are built offline from that raw log
by ``build_training_trajs.py``.

Streaming:
- ``/v1/messages`` with ``stream=true``: the upstream call is made
  non-streaming; while waiting, ``event: ping`` heartbeats are sent every
  ``heartbeat_interval`` seconds (prevents Claude Code timeouts), then the full
  response is emitted as a standard Anthropic SSE event burst.  This mirrors
  swalm's AnthropicRelay behavior and guarantees the complete response object
  is available for logging.
- ``/chat/completions`` with ``stream=true``: upstream SSE chunks are passed
  through to the client unmodified while deltas are accumulated into a full
  response object for the log.

Usage:
    /home/tiger/cap-x/.venv/bin/python server.py --port 8118
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("super_relay")

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

_NON_RETRYABLE = {400, 401, 403, 404, 405, 413, 415, 422}

# "maximum context length is 262144 tokens. However, you requested about 270344 tokens"
_CONTEXT_FULL_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?requested about (\d+) tokens", re.S)


def fitted_max_tokens(requested_max: int, message: str) -> int | None:
    """A response reservation that fits the window, or None if the message is not about that.

    A model's context window is shared between the prompt and the response, so a large
    `max_tokens` is a claim on the same budget the conversation is filling: reserving 128k of a
    262k window fails every request whose prompt passes ~134k, and an agent's prompt grows all
    run. The reservation is a ceiling, not a requirement, so fit it to what is left instead of
    losing the turn.
    """
    m = _CONTEXT_FULL_RE.search(message or "")
    if not m or requested_max <= 0:
        return None
    limit, requested = int(m.group(1)), int(m.group(2))
    room = limit - (requested - requested_max)   # what the prompt already occupies
    room -= max(512, room // 50)                 # the upstream's count is approximate
    return room if 1024 <= room < requested_max else None


def new_request_id() -> str:
    """Super-relay style id: `<epoch-ms>-<12 hex chars>`."""
    return f"{int(time.time() * 1000)}-{secrets.token_hex(6)}"


class RequestLogger:
    """Append-only JSONL logger for raw request/response records."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "raw_requests.jsonl"
        self._lock = threading.Lock()

    def log(
        self,
        request_id: str,
        session_id: str,
        api_format: str,
        request_body: dict,
        response_body: dict | None,
        status: int,
        provider: str = "",
    ) -> None:
        record = {
            "request_id": request_id,
            "ts_ms": int(request_id.split("-")[0]),
            "session_id": session_id,
            "api_format": api_format,
            "provider": provider,
            "model": request_body.get("model", ""),
            "status": status,
            "request": request_body,
            "response": response_body,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()


def derive_session_id(headers, body: dict) -> str:
    """Derive a stable per-agent session id (never log raw keys).

    Preference order:
    1. ``metadata.user_id`` from the request body — Claude Code embeds a
       per-session uuid here, so concurrently spawned CC agents sharing one
       OAuth token still get separate trajectory groups.
    2. Hash of the client auth token (distinct tokens = distinct sessions).
    3. "anon".
    """
    import hashlib

    user_id = (body.get("metadata") or {}).get("user_id", "")
    if user_id:
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    token = headers.get("x-api-key", "")
    if not token:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:]
    if not token:
        return "anon"
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def strip_openrouter_prefix(model: str) -> str:
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


def provider_from_upstream(upstream_base: str) -> str:
    """Provider tag used to prefix metadata.model in trajectories."""
    from urllib.parse import urlparse

    host = urlparse(upstream_base).hostname or ""
    if "openrouter" in host:
        return "openrouter"
    if "anthropic" in host:
        return "anthropic"
    return ""


# ---------------------------------------------------------------------------
# Anthropic response -> SSE burst (copied verbatim in structure from
# swalm/agent/native_claude_code/agent/relay.py::_anthropic_response_to_sse,
# which is the known-working implementation; swalm itself is not importable
# from the cap-x venv because it requires aiohttp at module level).
# ---------------------------------------------------------------------------


def anthropic_response_to_sse(resp: dict) -> list[bytes]:
    events: list[bytes] = []

    def _sse(event_type: str, data: dict) -> bytes:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    usage = resp.get("usage", {})

    events.append(
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": resp.get("id", "msg_unknown"),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": resp.get("model", ""),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": 0,
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                    },
                },
            },
        )
    )

    for idx, block in enumerate(resp.get("content", [])):
        block_type = block.get("type", "text")

        if block_type == "thinking":
            start_block: dict = {"type": "thinking", "thinking": ""}
        elif block_type == "tool_use":
            start_block = {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": {},
            }
        else:
            start_block = {"type": "text", "text": ""}

        events.append(
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": idx, "content_block": start_block},
            )
        )

        if block_type == "thinking":
            thinking_text = block.get("thinking", "")
            if thinking_text:
                events.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "thinking_delta", "thinking": thinking_text},
                        },
                    )
                )
            signature = block.get("signature", "")
            if signature:
                events.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "signature_delta", "signature": signature},
                        },
                    )
                )
        elif block_type == "tool_use":
            input_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            events.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": input_json},
                    },
                )
            )
        else:
            text = block.get("text", "")
            if text:
                events.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )

        events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))

    events.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": resp.get("stop_reason", "end_turn"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    events.append(_sse("message_stop", {"type": "message_stop"}))
    return events


# ---------------------------------------------------------------------------
# OpenAI streaming-chunk accumulation
# ---------------------------------------------------------------------------


def accumulate_openai_chunks(chunks: list[dict]) -> dict:
    """Assemble streamed OpenAI chat chunks into a full ChatCompletion dict."""
    result: dict = {"id": "", "object": "chat.completion", "created": 0, "model": "", "choices": [], "usage": None}
    message: dict = {"role": "assistant", "content": "", "tool_calls": []}
    finish_reason = None
    reasoning = ""

    for chunk in chunks:
        result["id"] = chunk.get("id") or result["id"]
        result["created"] = chunk.get("created") or result["created"]
        result["model"] = chunk.get("model") or result["model"]
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                message["content"] += delta["content"]
            if delta.get("reasoning") or delta.get("reasoning_content"):
                reasoning += delta.get("reasoning") or delta.get("reasoning_content")
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                while len(message["tool_calls"]) <= idx:
                    message["tool_calls"].append(
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                slot = message["tool_calls"][idx]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    if not message["tool_calls"]:
        message.pop("tool_calls")
    if reasoning:
        message["reasoning_content"] = reasoning
    result["choices"] = [{"index": 0, "message": message, "finish_reason": finish_reason}]
    return result


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(
    api_key: str | None = None,
    log_dir: Path = DEFAULT_LOG_DIR,
    max_retries: int = 5,
    heartbeat_interval: int = 15,
    read_timeout: int = 600,
    upstream_base: str = ANTHROPIC_BASE,
    force_model: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Super Relay", version="1.0.0")
    req_logger = RequestLogger(log_dir)
    provider = provider_from_upstream(upstream_base)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=read_timeout, write=60, pool=30),
    )

    def _auth_headers(request: Request, anthropic: bool) -> dict:
        """Build upstream auth headers.

        With a relay-owned ``api_key``: substitute it (client token is only a
        session id).  Without one (default, the Claude Code case): forward the
        client's own credentials verbatim — ``x-api-key`` (API keys),
        ``Authorization: Bearer`` (CC OAuth tokens), and ``anthropic-beta``
        (required for OAuth, e.g. ``oauth-2025-04-20``).
        """
        headers = {"Content-Type": "application/json"}
        if anthropic:
            headers["anthropic-version"] = request.headers.get("anthropic-version", "2023-06-01")
            if request.headers.get("anthropic-beta"):
                headers["anthropic-beta"] = request.headers["anthropic-beta"]
        if api_key:
            headers["x-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            if request.headers.get("x-api-key"):
                headers["x-api-key"] = request.headers["x-api-key"]
            if request.headers.get("authorization"):
                headers["Authorization"] = request.headers["authorization"]
        return headers

    # Providers seen rejecting a request for their own configuration (see _blamed_provider).
    ignored_providers: set[str] = set()

    def _with_ignored(payload: dict) -> dict:
        """Ask the gateway to skip providers already known to reject these requests."""
        if provider != "openrouter" or not ignored_providers:
            return payload
        merged = dict(payload)
        routing = dict(merged.get("provider") or {})
        routing["ignore"] = sorted(set(routing.get("ignore") or []) | ignored_providers)
        merged["provider"] = routing
        return merged

    def _blamed_provider(status: int, body: dict) -> str:
        """The provider a gateway names in a refusal it returned on that provider's behalf.

        One model is served by many providers, and a request the MODEL supports can still be
        refused by whichever one the gateway picked — seen on 2026-08-07 for qwen3.6-27b as both
        a misconfiguration ('"auto" tool choice requires --enable-auto-tool-choice', so every
        request carrying tools failed) and a geo-block ('Country, region, or territory not
        supported' for the sandbox's egress). Either kills an agent run while other providers
        answer the same request fine.

        These statuses are normally the client's fault and so not retried; a named provider is
        what says otherwise, because the gateway only fills provider_name in for a refusal it is
        passing on. Its own refusals (bad request, out of credit, everything ignored) carry no
        provider, and stay fatal. Naming an already-excluded provider again is not redundant:
        asking the gateway to skip one does not guarantee it will, and such a response still has
        to be retried rather than handed to the client as its own error — that is what made
        every agent turn eat a failed call before the client's own retry got through.
        """
        if provider != "openrouter" or status not in _NON_RETRYABLE:
            return ""
        return ((body.get("error") or {}).get("metadata") or {}).get("provider_name") or ""

    async def _forward_with_retries(url: str, payload: dict, headers: dict) -> tuple[int, dict]:
        """POST to upstream, retrying 429/5xx and network errors with backoff."""
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(url, json=_with_ignored(payload), headers=headers)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(2**attempt, 60)
                    logger.warning(
                        "upstream %s (attempt %d/%d), retrying in %ds",
                        type(e).__name__, attempt, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            try:
                body = resp.json()
            except Exception:
                body = {"error": {"message": resp.text[:2000]}}
            blamed = _blamed_provider(resp.status_code, body)
            if blamed and attempt < max_retries:
                if blamed not in ignored_providers:
                    ignored_providers.add(blamed)
                    logger.warning("provider %s rejected the request for its own configuration; "
                                   "excluding it and retrying (ignored so far: %s)",
                                   blamed, ", ".join(sorted(ignored_providers)))
                else:
                    logger.info("provider %s again (attempt %d/%d), retrying",
                                blamed, attempt, max_retries)
                continue
            if resp.status_code == 400 and attempt < max_retries:
                fitted = fitted_max_tokens(int(payload.get("max_tokens") or 0),
                                           (body.get("error") or {}).get("message") or "")
                if fitted:
                    logger.warning("the prompt leaves room for %d response tokens, not the %s "
                                   "asked for; requesting that instead",
                                   fitted, payload.get("max_tokens"))
                    payload = dict(payload, max_tokens=fitted)
                    continue
            if resp.status_code == 200 or resp.status_code in _NON_RETRYABLE:
                return resp.status_code, body
            last_error = RuntimeError(f"upstream {resp.status_code}: {json.dumps(body)[:500]}")
            if attempt < max_retries:
                delay = min(2**attempt, 60)
                logger.warning(
                    "upstream HTTP %d (attempt %d/%d), retrying in %ds",
                    resp.status_code, attempt, max_retries, delay,
                )
                await asyncio.sleep(delay)
            else:
                return resp.status_code, body
        raise last_error  # network errors exhausted retries

    # ------------------------------------------------------------------
    # Anthropic Messages format
    # ------------------------------------------------------------------

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        request_id = new_request_id()
        session_id = derive_session_id(request.headers, body)
        model = strip_openrouter_prefix(body.get("model", ""))
        if force_model:
            # Model-name rewrite for gateways whose model ids the client rejects
            # client-side (Claude Code only accepts claude-* names; byted
            # super-relay wants e.g. model_hub/es1_orange_o48).
            model = force_model
        stream_requested = bool(body.get("stream", False))
        logger.info(
            "[%s] anthropic %s msgs=%d tools=%d stream=%s",
            request_id, model, len(body.get("messages", [])),
            len(body.get("tools") or []), stream_requested,
        )

        payload = dict(body)
        payload["model"] = model
        payload["stream"] = False  # upstream call is always non-streaming
        # Claude Code sends thinking.display="omitted", which makes the API
        # return thinking blocks with EMPTY text (signature only) — useless
        # for trajectory logging.  Strip the flag so full reasoning text is
        # returned and logged; CC handles the extra text fine.
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("display") == "omitted":
            payload["thinking"] = {k: v for k, v in thinking.items() if k != "display"}
        headers = _auth_headers(request, anthropic=True)
        url = f"{upstream_base}/messages"

        if not stream_requested:
            status, resp_body = await _forward_with_retries(url, payload, headers)
            req_logger.log(request_id, session_id, "anthropic", body, resp_body, status, provider=provider)
            return JSONResponse(resp_body, status_code=status)

        # Streaming: heartbeat pings while waiting, then SSE burst (swalm relay pattern).
        llm_task = asyncio.create_task(_forward_with_retries(url, payload, headers))

        async def event_stream():
            try:
                while not llm_task.done():
                    yield b'event: ping\ndata: {"type": "ping"}\n\n'
                    await asyncio.wait({llm_task}, timeout=heartbeat_interval)
                status, resp_body = llm_task.result()
                req_logger.log(request_id, session_id, "anthropic", body, resp_body, status, provider=provider)
                if status != 200:
                    err = {"type": "error", "error": resp_body.get("error", resp_body)}
                    yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n".encode()
                    return
                for event_bytes in anthropic_response_to_sse(resp_body):
                    yield event_bytes
            except Exception as e:
                llm_task.cancel()
                logger.error("[%s] streaming error: %s", request_id, e, exc_info=True)
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                req_logger.log(request_id, session_id, "anthropic", body, err, 500, provider=provider)
                yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n".encode()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        # Forward to the real endpoint when the upstream is Anthropic; other
        # gateways (e.g. OpenRouter) don't implement it, so fall back to the
        # same stub swalm's AnthropicRelay uses.
        body = await request.json()
        try:
            resp = await client.post(
                f"{upstream_base}/messages/count_tokens",
                json=body,
                headers=_auth_headers(request, anthropic=True),
            )
            if resp.status_code == 200:
                return JSONResponse(resp.json())
            logger.warning("count_tokens upstream HTTP %d, using stub", resp.status_code)
        except Exception as e:
            logger.warning("count_tokens upstream failed (%s), using stub", e)
        return JSONResponse({"input_tokens": 100})

    @app.post("/api/event_logging/batch")
    async def event_logging(request: Request):
        return JSONResponse({})

    # ------------------------------------------------------------------
    # OpenAI chat format
    # ------------------------------------------------------------------

    async def _chat_completions(request: Request):
        body = await request.json()
        request_id = new_request_id()
        session_id = derive_session_id(request.headers, body)
        model = strip_openrouter_prefix(body.get("model", ""))
        stream_requested = bool(body.get("stream", False))
        logger.info(
            "[%s] openai %s msgs=%d stream=%s",
            request_id, model, len(body.get("messages", [])), stream_requested,
        )

        payload = dict(body)
        payload["model"] = model
        headers = _auth_headers(request, anthropic=False)
        url = f"{upstream_base}/chat/completions"

        if not stream_requested:
            status, resp_body = await _forward_with_retries(url, payload, headers)
            req_logger.log(request_id, session_id, "openai", body, resp_body, status, provider=provider)
            return JSONResponse(resp_body, status_code=status)

        # Streaming passthrough with accumulation for the log.
        async def event_stream():
            chunks: list[dict] = []
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        text = (await resp.aread()).decode(errors="replace")
                        try:
                            err_body = json.loads(text)
                        except Exception:
                            err_body = {"error": {"message": text[:2000]}}
                        req_logger.log(request_id, session_id, "openai", body, err_body, resp.status_code, provider=provider)
                        yield f"data: {json.dumps(err_body, ensure_ascii=False)}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break  # our own [DONE] is emitted below
                            try:
                                chunks.append(json.loads(data))
                            except json.JSONDecodeError as e:
                                logger.warning("[%s] unparseable chunk (%s): %r", request_id, e, data[:200])
                        if line:
                            yield (line + "\n\n").encode()
                yield b"data: [DONE]\n\n"
                req_logger.log(request_id, session_id, "openai", body, accumulate_openai_chunks(chunks), 200, provider=provider)
            except Exception as e:
                logger.error("[%s] streaming error: %s", request_id, e, exc_info=True)
                err = {"error": {"message": str(e)}}
                req_logger.log(request_id, session_id, "openai", body, err, 500, provider=provider)
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode()
                yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    app.post("/chat/completions")(_chat_completions)
    app.post("/v1/chat/completions")(_chat_completions)

    # ------------------------------------------------------------------
    # OpenAI Responses format — what codex speaks. Same contract as the other
    # two: forward, and log the COMPLETE request/response pair.
    # ------------------------------------------------------------------

    async def _responses(request: Request):
        body = await request.json()
        request_id = new_request_id()
        session_id = derive_session_id(request.headers, body)
        model = strip_openrouter_prefix(body.get("model", ""))
        if force_model:
            model = force_model
        stream_requested = bool(body.get("stream", False))
        logger.info(
            "[%s] responses %s input_items=%s stream=%s",
            request_id, model,
            len(body.get("input", [])) if isinstance(body.get("input"), list) else 1,
            stream_requested,
        )

        payload = dict(body)
        payload["model"] = model
        headers = _auth_headers(request, anthropic=False)
        url = f"{upstream_base}/responses"

        if not stream_requested:
            status, resp_body = await _forward_with_retries(url, payload, headers)
            req_logger.log(request_id, session_id, "openai_responses", body, resp_body, status,
                           provider=provider)
            return JSONResponse(resp_body, status_code=status)

        # Streaming passthrough. The final `response.completed` event carries the whole
        # response object, so that is what the log gets — the same completeness the
        # anthropic path buys by calling upstream non-streaming.
        async def event_stream():
            final: dict | None = None
            try:
                async with client.stream("POST", url, json=_with_ignored(payload),
                                         headers=headers) as resp:
                    if resp.status_code != 200:
                        text = (await resp.aread()).decode(errors="replace")
                        try:
                            err_body = json.loads(text)
                        except Exception:  # noqa: BLE001
                            err_body = {"error": {"message": text[:2000]}}
                        req_logger.log(request_id, session_id, "openai_responses", body,
                                       err_body, resp.status_code, provider=provider)
                        yield f"data: {json.dumps(err_body, ensure_ascii=False)}\n\n".encode()
                        return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[len("data: "):])
                                if chunk.get("type") == "response.completed":
                                    final = chunk.get("response")
                            except json.JSONDecodeError:
                                pass
                        if line:
                            yield (line + "\n").encode()
                        else:
                            yield b"\n"
                req_logger.log(request_id, session_id, "openai_responses", body,
                               final if final is not None
                               else {"error": {"message": "stream ended without "
                                                          "response.completed"}},
                               200 if final is not None else 502, provider=provider)
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] responses streaming error: %s", request_id, e, exc_info=True)
                err = {"error": {"message": str(e)}}
                req_logger.log(request_id, session_id, "openai_responses", body, err, 500,
                               provider=provider)
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    app.post("/responses")(_responses)
    app.post("/v1/responses")(_responses)

    @app.get("/health")
    async def health():
        return {"status": "ok", "log_file": str(req_logger.path)}

    @app.get("/")
    async def root():
        return Response(status_code=200, content="ok")

    return app


def main(
    host: str = "0.0.0.0",
    port: int = 8118,
    api_key: str | None = None,
    force_model: str | None = None,
    log_dir: str = str(DEFAULT_LOG_DIR),
    max_retries: int = 5,
    heartbeat_interval: int = 15,
    read_timeout: int = 600,
    upstream_base: str = ANTHROPIC_BASE,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(
        api_key=api_key,
        log_dir=Path(log_dir),
        max_retries=max_retries,
        heartbeat_interval=heartbeat_interval,
        read_timeout=read_timeout,
        upstream_base=upstream_base,
        force_model=force_model,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Super-relay clone with trajectory logging")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8118)
    # Also from the environment: on a shared box /proc/<pid>/cmdline is readable by
    # every user, so a key passed as an argument is a key handed to whoever is running
    # there — including an agent under evaluation. /proc/<pid>/environ is owner-only.
    parser.add_argument("--api-key", default=os.environ.get("SUPER_RELAY_API_KEY"),
                        help="relay-owned upstream key (or SUPER_RELAY_API_KEY); "
                             "default: pass the client's own auth through")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--heartbeat-interval", type=int, default=15)
    parser.add_argument("--read-timeout", type=int, default=600)
    parser.add_argument("--force-model", default=os.environ.get("SUPER_RELAY_FORCE_MODEL"),
                        help="rewrite every request's model to this upstream id "
                             "(or SUPER_RELAY_FORCE_MODEL)")
    parser.add_argument("--upstream-base", default=ANTHROPIC_BASE,
                        help="upstream API base URL (default: Anthropic; use "
                             "https://openrouter.ai/api/v1 for OpenRouter)")
    args = parser.parse_args()
    main(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        log_dir=args.log_dir,
        max_retries=args.max_retries,
        heartbeat_interval=args.heartbeat_interval,
        read_timeout=args.read_timeout,
        upstream_base=args.upstream_base,
        force_model=args.force_model,
    )
