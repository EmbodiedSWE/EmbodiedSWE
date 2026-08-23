"""Wire format for the eval socket — length-prefixed JSON header + raw numpy buffers.

Shared by both sides of the closed-loop eval: serve.py (CoSiGen venv, py3.11)
and the lerobot_env_cosigen plugin (lerobot venv, py3.12) — which is why this
module must stay stdlib + numpy only, with no pickle (cross-version) and no
base64 (a 640x480x3 frame stays a raw 0.9 MB buffer).

One message = uint32 header length, JSON header, then each array's raw bytes
in header order. The header's "arrays" entry carries [{key, dtype, shape}].
An {"error": ...} header is the server-side failure channel; the client raises.
"""

from __future__ import annotations

import json
import struct

import numpy as np


def send_msg(sock, header: dict, arrays: dict[str, "np.ndarray"] | None = None) -> None:
    arrays = arrays or {}
    h = dict(header)
    h["arrays"] = [{"key": k, "dtype": str(v.dtype), "shape": list(v.shape)}
                   for k, v in arrays.items()]
    hb = json.dumps(h).encode()
    sock.sendall(struct.pack("<I", len(hb)) + hb)
    for v in arrays.values():
        sock.sendall(np.ascontiguousarray(v).tobytes())


def recv_msg(sock) -> tuple[dict, dict[str, "np.ndarray"]]:
    (hlen,) = struct.unpack("<I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, hlen))
    arrays = {}
    for m in header.pop("arrays", []):
        n = int(np.prod(m["shape"])) * np.dtype(m["dtype"]).itemsize
        # .copy(): frombuffer views are read-only; consumers hand these to torch
        arrays[m["key"]] = np.frombuffer(_recv_exact(sock, n),
                                         dtype=m["dtype"]).reshape(m["shape"]).copy()
    return header, arrays


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf += chunk
    return bytes(buf)
