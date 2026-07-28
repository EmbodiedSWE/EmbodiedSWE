"""CLI client for a forge server — what construction agents call to test their task.

Usage (FORGE_URL from the orchestrator, in the agent's environment):
  python3 sim_gen/isaac/forge_client.py ping
  python3 sim_gen/isaac/forge_client.py submit <task_name> <local_task_dir>
  python3 sim_gen/isaac/forge_client.py run <task_name> [--module smoke] [--timeout 900] \
      [--args "--flag value"]
  python3 sim_gen/isaac/forge_client.py fetch <remote_rel_path> <local_dest>
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path


def _url() -> str:
    u = os.environ.get("FORGE_URL", "").rstrip("/")
    if not u:
        sys.exit("FORGE_URL is not set")
    return u


def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(_url() + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.loads(opener.open(req, timeout=timeout).read())


def _get(path: str, timeout: float = 30.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return json.loads(opener.open(_url() + path, timeout=timeout).read())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    s = sub.add_parser("submit")
    s.add_argument("task")
    s.add_argument("local_dir")
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--module", default="smoke")
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--args", default="")
    f = sub.add_parser("fetch")
    f.add_argument("remote_rel")
    f.add_argument("local_dest")
    a = ap.parse_args()

    if a.cmd == "ping":
        print(json.dumps(_get("/ping")))
    elif a.cmd == "submit":
        files = {}
        root = Path(a.local_dir)
        for p in root.rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                files[str(p.relative_to(root))] = p.read_text()
        out = _post("/submit", {"task": a.task, "files": files}, 60)
        print(json.dumps(out))
    elif a.cmd == "run":
        out = _post("/run", {"task": a.task, "module": a.module,
                             "args": a.args.split() if a.args else [],
                             "timeout_s": a.timeout}, a.timeout + 60)
        # stdout_tail printed raw so the agent reads the smoke output directly
        tail = out.pop("stdout_tail", "")
        print(json.dumps(out))
        print("---- smoke output tail ----")
        print(tail)
    elif a.cmd == "fetch":
        out = _get(f"/fetch?path={a.remote_rel}", timeout=300)
        if not out.get("ok"):
            sys.exit(f"fetch failed: {out}")
        Path(a.local_dest).parent.mkdir(parents=True, exist_ok=True)
        Path(a.local_dest).write_bytes(base64.b64decode(out["b64"]))
        print(f"wrote {a.local_dest} ({out['size']} bytes)")


if __name__ == "__main__":
    main()
