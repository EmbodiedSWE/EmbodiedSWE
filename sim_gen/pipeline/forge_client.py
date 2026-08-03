"""CLI client for a sim_gen Isaac forge (isaac/forge_server.py).

The construction agent tests its robobench-format task on its assigned forge:

  python sim_gen/pipeline/forge_client.py --forge-url URL submit --task NAME --dir DIR
  python sim_gen/pipeline/forge_client.py --forge-url URL run --task NAME \
      [--module smoke] [--timeout 1200] [--args "--headless --demo"]
  python sim_gen/pipeline/forge_client.py --forge-url URL fetch --path rel/file --out local
  python sim_gen/pipeline/forge_client.py --forge-url URL ping
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path


def _call(url: str, payload: dict | None = None, timeout: float = 1800.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if payload is None:
        return json.loads(opener.open(url, timeout=timeout).read())
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(opener.open(req, timeout=timeout).read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge-url", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    s = sub.add_parser("submit")
    s.add_argument("--task", required=True)
    s.add_argument("--dir", required=True, help="local task package dir (*.py, *.md)")
    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--module", default="smoke")
    r.add_argument("--timeout", type=float, default=1200)
    r.add_argument("--args", default="--headless",
                   help="space-separated args passed to the module")
    f = sub.add_parser("fetch")
    f.add_argument("--path", required=True)
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    base = args.forge_url.rstrip("/")

    if args.cmd == "ping":
        print(json.dumps(_call(base + "/ping", timeout=20)))
        return
    if args.cmd == "submit":
        d = Path(args.dir)
        files = {str(p.relative_to(d)): p.read_text()
                 for p in d.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".txt")}
        res = _call(base + "/submit", {"task": args.task, "files": files}, timeout=60)
        print(json.dumps(res))
        sys.exit(0 if res.get("ok") else 1)
    if args.cmd == "run":
        res = _call(base + "/run", {"task": args.task, "module": args.module,
                                    "args": args.args.split(), "timeout_s": args.timeout},
                    timeout=args.timeout + 120)
        print(res.get("stdout_tail", ""))
        print(f"\n[forge] rc={res.get('rc')} seconds={res.get('seconds')} "
              f"all_pass={res.get('all_pass')}")
        sys.exit(0 if res.get("rc") == 0 and res.get("all_pass") else 1)
    if args.cmd == "fetch":
        res = _call(base + f"/fetch?path={args.path}", timeout=300)
        if not res.get("ok"):
            print(json.dumps(res))
            sys.exit(1)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(base64.b64decode(res["b64"]))
        print(f"fetched {res['path']} ({res['size']} bytes) -> {args.out}")


if __name__ == "__main__":
    main()
