"""Minimal HTTP server around an openpi trained policy (for BC eval).

Loads a checkpoint via openpi's own policy factory (PyTorch checkpoints are
auto-detected from model.safetensors; norm stats come from the checkpoint's
assets/ dir) and answers:

    GET  /health -> {"ok": true, "ckpt": ...}
    POST /act    <- npz{image u8 (256,256,3), wrist_image u8, state f32 (8,),
                       prompt str}
                 -> npz{actions f32 (horizon, 7)}

Plain stdlib HTTP + npz transport so the forge-side client (eval_batch.py)
needs no extra deps. Single-threaded on purpose: one eval batch per server,
GPU inference serialized.

  .venv/bin/python CoSiGen_vla/policy_http_server.py \
      --config pi05_simgen_bc_v1 --ckpt /opt/tiger/bc_ckpt --port 9100
"""
from __future__ import annotations

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_simgen_bc_v1")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (model.safetensors inside)")
    ap.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()

    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config

    policy = policy_config.create_trained_policy(
        _config.get_config(args.config), args.ckpt)
    print(f"[policy-server] loaded {args.ckpt}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet the per-request spam
            pass

        def do_GET(self):
            body = json.dumps({"ok": True, "ckpt": args.ckpt}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            data = np.load(io.BytesIO(self.rfile.read(n)), allow_pickle=False)
            example = {
                "observation/image": np.asarray(data["image"]),
                "observation/wrist_image": np.asarray(data["wrist_image"]),
                "observation/state": np.asarray(data["state"]),
                "prompt": str(data["prompt"]),
            }
            actions = np.asarray(policy.infer(example)["actions"], dtype=np.float32)
            buf = io.BytesIO()
            np.savez_compressed(buf, actions=actions)
            body = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class V6Server(HTTPServer):
        import socket as _socket
        address_family = _socket.AF_INET6  # stdlib default is AF_INET; '::' needs v6

    srv = V6Server(("::", args.port), H)  # dual-stack -- forge pods reach us over v6
    print(f"[policy-server] listening on :{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
