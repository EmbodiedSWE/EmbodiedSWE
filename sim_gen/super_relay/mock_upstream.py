"""Minimal mock upstream implementing Anthropic /v1/messages and OpenAI
/chat/completions, for end-to-end testing of server.py without a live
OpenRouter key.

Behavior (deterministic):
- /v1/messages: if the last user message contains a tool_result -> text answer;
  else if tools are offered -> a tool_use call to the first tool;
  else -> plain text answer.
- /chat/completions: echoes a short text answer (streaming supported).

Run:  /home/tiger/cap-x/.venv/bin/python mock_upstream.py --port 8119
"""

import argparse
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
_counter = {"n": 0}


@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    _counter["n"] += 1
    n = _counter["n"]
    msgs = body.get("messages", [])
    last = msgs[-1] if msgs else {}
    last_blocks = last.get("content") if isinstance(last.get("content"), list) else []
    has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in last_blocks)

    if has_tool_result:
        content = [{"type": "text", "text": f"The result is {last_blocks[0].get('content', '?')}."}]
        stop = "end_turn"
    elif body.get("tools") and len(msgs) == 1:
        tool = body["tools"][0]
        content = [
            {"type": "text", "text": "Let me compute that."},
            {"type": "tool_use", "id": f"toolu_mock{n:04d}", "name": tool["name"],
             "input": {"expression": "137*249"}},
        ]
        stop = "tool_use"
    else:
        content = [{"type": "text", "text": "34113 / 3 = 11371.00"}]
        stop = "end_turn"

    return JSONResponse({
        "id": f"msg_mock{n:06d}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", ""),
        "content": content,
        "stop_reason": stop,
        "usage": {"input_tokens": 42, "output_tokens": 17},
    })


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    _counter["n"] += 1
    n = _counter["n"]
    if body.get("stream"):
        async def gen():
            for piece in ["Hello", " from", " mock."]:
                chunk = {"id": f"cc{n}", "object": "chat.completion.chunk", "created": 0,
                         "model": body.get("model", ""),
                         "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            final = {"id": f"cc{n}", "object": "chat.completion.chunk", "created": 0,
                     "model": body.get("model", ""),
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse({
        "id": f"cc{n}", "object": "chat.completion", "created": 0, "model": body.get("model", ""),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from mock."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8119)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
