#!/usr/bin/env python3
"""Laptop-side AIDP forwarder: the intranet leg of the pod->laptop->AIDP bridge.

Listens on 127.0.0.1:8898 and forwards every request to https://aidp.bytedance.net with the
same path, query, and body. Pods reach it through a reverse SSH tunnel (`ssh -R
8899:localhost:8898 root@pod`, opened by laptop_tunnel_daemon.py), so the pod-local URL
http://127.0.0.1:8899/api/modelhub/online/v2/crawl?ak=... transparently becomes an AIDP call
made from the corp network.

Stdlib only (runs on the stock macOS python3); threaded so concurrent pods don't serialize.

    python3 eval/scripts/laptop_aidp_forwarder.py            # 127.0.0.1:8898
    curl http://127.0.0.1:8898/_health                       # -> ok
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://aidp.bytedance.net"
# Generations on the big models run minutes; match the relay's read timeout.
READ_TIMEOUT = 600
# Hop-by-hop headers must not be forwarded (RFC 7230); everything else passes through.
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "upgrade", "host",
               "content-length", "accept-encoding"}


class Forwarder(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, body: bytes | None) -> None:
        if self.path == "/_health":
            payload = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        url = UPSTREAM + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_HEADERS}
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as resp:
                data = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in HOP_HEADERS:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:  # upstream 4xx/5xx: pass through verbatim
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type",
                             exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:  # noqa: BLE001 -- network failure: report as 502, never hang
            data = f'{{"error": "laptop forwarder: {type(exc).__name__}: {exc}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self):  # noqa: N802 -- http.server API
        length = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(length) if length else None)

    def do_GET(self):  # noqa: N802
        self._forward(None)

    def log_message(self, fmt, *args):  # one line per request, with the client visible
        print(f"[forwarder] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8898)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Forwarder)
    server.daemon_threads = True
    print(f"[forwarder] listening on {args.host}:{args.port} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
