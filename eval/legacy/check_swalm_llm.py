#!/usr/bin/env python3
"""Connectivity / permission check for the swalm LLM pipeline.

Verifies that swalm's LLMCaller can reach + authenticate a model, independent of the
CoSiGen render loop. Use this to debug model wiring before running the full harness
(scripts/cosigen_harness.py --model ...).

Run with the swalm venv (has swalm + anthropic/openai SDKs), NO proxy for internal
served endpoints (the modelhub claude endpoints are reachable directly too):

    .venv-swalm/bin/python scripts/tests/check_swalm_llm.py --model claude-azure
    .venv-swalm/bin/python scripts/tests/check_swalm_llm.py --model served --endpoint http://host:8000
    .venv-swalm/bin/python scripts/tests/check_swalm_llm.py --probe            # try a candidate model list

Exit 0 iff the chosen model answers.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import types

# Reuse the exact LLMConfig presets the harness uses, so this test matches production wiring.
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/scripts")
from cosigen_harness import build_llm_config  # noqa: E402

from swalm.core.types.llm import LLMConfig  # noqa: E402
from swalm.core.llm.llm_caller_factory import LLMCallerFactory  # noqa: E402

# Candidate model names to probe against the AzureOpenAI (modelhub) route when the
# provided key's model access is unknown.
PROBE_MODELS = [
    "es1_orange_o47", "gpt-4o", "gpt-4.1", "gpt-4o-2024-08-06",
    "claude-3-7-sonnet", "gcp-claude4-sonnet",
]

AZURE_CLIENT_ARGS = {
    "azure_endpoint": "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl",
    "api_key": "***REMOVED-SECRET***",
    "api_version": "2024-02-01",
}


async def _call(cfg: LLMConfig, timeout: float) -> tuple[bool, str]:
    caller = LLMCallerFactory.create_llm_caller(cfg)
    t = time.time()
    try:
        r = await asyncio.wait_for(caller._call_llm([{"role": "user", "content": "say PONG"}]), timeout=timeout)
        return True, f"OK in {time.time()-t:.1f}s content={r.content!r}"
    except Exception as e:  # surface the real error, never swallow it
        return False, f"FAIL in {time.time()-t:.1f}s: {str(e)[:200]}"


async def probe(timeout: float) -> int:
    ok_any = False
    for m in PROBE_MODELS:
        cfg = LLMConfig(client_type="AzureOpenAI", client_args=AZURE_CLIENT_ARGS,
                        request_args={"model": m, "max_tokens": 64})
        ok, msg = await _call(cfg, timeout)
        print(f"[{m}] {msg}", flush=True)
        ok_any = ok_any or ok
    return 0 if ok_any else 1


async def check_one(args) -> int:
    fake = types.SimpleNamespace(model=args.model, endpoint=args.endpoint,
                                 temperature=0.7, top_p=0.95, max_tokens=1024)
    cfg = build_llm_config(fake)
    print(f"[check] model={args.model} client_type={cfg.client_type}", flush=True)
    ok, msg = await _call(cfg, args.timeout)
    print(f"[check] {msg}", flush=True)
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["served", "claude-anthropic", "claude-azure"],
                    help="which harness preset to check")
    ap.add_argument("--endpoint", default=None, help="serving URL (for --model served)")
    ap.add_argument("--probe", action="store_true", help="probe a list of candidate model names via Azure route")
    ap.add_argument("--timeout", type=float, default=40.0)
    args = ap.parse_args()

    if args.probe:
        sys.exit(asyncio.run(probe(args.timeout)))
    if not args.model:
        ap.error("pass --model <preset> or --probe")
    sys.exit(asyncio.run(check_one(args)))


if __name__ == "__main__":
    main()
