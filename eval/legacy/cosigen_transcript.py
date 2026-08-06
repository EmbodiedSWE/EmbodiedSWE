#!/usr/bin/env python3
"""Write workspace/.agent/transcript.jsonl while our harness runs.

The eval pipeline reads one file to price a run: eval/grader/collect_spend.py sums
``message.usage`` over the first N lines of transcript.jsonl, where N is the position stamped
into each submission. The claude adapter gets that file for free — the CLI's own
``--output-format stream-json`` IS that format. Our harness drives the model itself, so the
same file has to come from somewhere, and the honest source is the trajectory relay: it holds
the provider's own response for every call, usage included, so the numbers are the provider's
rather than anything reconstructed here.

One relay record with a response becomes one line:

    {"type": "assistant", "message": <the provider's response verbatim>}

Nothing is summarised or dropped, so a submission stamped at line N prices exactly the calls
that had happened by then. Run as root (the relay's directory is root-only) for as long as the
run lasts:

    python3 cosigen_transcript.py --relay-log /opt/relay/trajlog/raw_requests.jsonl \
                                  --transcript /workspace/.agent/transcript.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-log", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--poll-s", type=float, default=2.0)
    args = ap.parse_args()

    relay, transcript = Path(args.relay_log), Path(args.transcript)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    # Resume where a restart left off, counted in relay records rather than bytes: the same
    # record must never be written twice, or the token totals would double.
    done = 0
    if transcript.exists():
        with transcript.open() as fh:
            done = sum(1 for _ in fh)
    offset = 0
    while True:
        if not relay.exists():
            time.sleep(args.poll_s)
            continue
        with relay.open() as fh:
            fh.seek(offset)
            new = []
            for line in fh:
                if not line.endswith("\n"):      # a record still being written
                    break
                offset += len(line.encode())
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                resp = rec.get("response")
                if isinstance(resp, dict) and resp.get("usage"):
                    new.append({"type": "assistant", "message": resp})
        if new:
            with transcript.open("a") as out:
                for ev in new:
                    out.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
                out.flush()
                os.fsync(out.fileno())   # a submission may be stamped against these lines
            done += len(new)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
