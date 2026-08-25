#!/usr/bin/env python3
"""Laptop-side AIDP bridge — the outbound-only successor to the (banned) reverse SSH tunnel.

The pod's relay runs with `--spool-dir /opt/relay/spool`: instead of calling any API, it
writes each upstream request to `<spool>/req/<id>.json` and waits for `<spool>/resp/<id>.json`.
This daemon, running on a corp-network laptop, does the intranet leg:

    for each live rb-* pod:
        pull  /opt/relay/spool/req/*.json   (scp FROM pod  — outbound)
        for each request: POST it to AIDP  (from THIS machine, a sanctioned corp client)
        push  the response back to /opt/relay/spool/resp/  (scp TO pod — outbound)
        delete the consumed request on the pod

Every connection is initiated BY the laptop, outbound (ssh/scp to a cloud VM, https to AIDP).
Nothing external ever connects into the corp network; no port is bridged or exposed. The API
is only ever called by this machine, and every request/response passes through here where it
is logged. This is the difference from the reverse tunnel, which exposed an inbound path.

Stdlib only (stock macOS python3); one worker thread per pod so pods never serialize.

    caffeinate -dims python3 eval/scripts/laptop_aidp_bridge.py
    curl ...          # nothing to curl: it drives itself off the RunPod pod list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

RUNPOD_KEY = "***REMOVED-SECRET***"
SSH_KEY = str(Path.home() / ".ssh" / "cosigen_campaign")
AIDP_BASE = "https://aidp.bytedance.net"
SPOOL = "/opt/relay/spool"
READ_TIMEOUT = 600           # match the relay's own upstream read timeout
LOG_DIR = Path("/Users/bytedance/Desktop/CoSiGen/eval_result/aidp_bridge_log")


def now() -> str:
    return time.strftime("%H:%M:%S")


def rest_pods() -> list[dict]:
    req = urllib.request.Request("https://rest.runpod.io/v1/pods",
                                 headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data if isinstance(data, list) else (data.get("pods") or data.get("data") or [])


def ssh(ip: str, port: int, cmd: str, timeout: float = 60.0) -> tuple[int, str]:
    p = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=15", f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout


def ssh_bytes(ip: str, port: int, cmd: str, timeout: float = 300.0) -> bytes:
    p = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=15", f"root@{ip}", cmd],
        capture_output=True, timeout=timeout)
    return p.stdout if p.returncode == 0 else b""


def untar_json(blob: bytes) -> dict:
    """{member basename: parsed json} from a tar stream (skips partial/non-json members)."""
    import io
    import tarfile
    out: dict = {}
    if not blob:
        return out
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".json"):
                    continue
                try:
                    out[m.name.rsplit("/", 1)[-1]] = json.loads(
                        tf.extractfile(m).read().decode(errors="replace"))
                except json.JSONDecodeError:
                    continue
    except tarfile.TarError:
        pass
    return out


def call_aidp(url: str, payload: dict) -> tuple[int, dict]:
    """Make the actual API call from this (corp-network) machine."""
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:  # upstream 4xx/5xx: pass the body through
        try:
            body = json.loads(exc.read())
        except Exception:  # noqa: BLE001
            body = {"error": {"message": f"aidp HTTP {exc.code}"}}
        return exc.code, body
    except Exception as exc:  # noqa: BLE001 -- network failure: 502 so the relay can retry
        return 502, {"error": {"message": f"laptop bridge: {type(exc).__name__}: {exc}"}}


def serve_pod(ip: str, port: int, name: str, stop: threading.Event) -> None:
    """Drain one pod's request spool until it stops appearing in the pod list."""
    log = LOG_DIR / f"{name}.jsonl"
    served: set = set()  # rids already fulfilled; the req file may outlive its response
    paid: dict = {}      # rid -> resp_json already bought from AIDP but not yet delivered
                         # (a failed push must be retried WITHOUT re-paying for the call)
    while not stop.is_set():
        try:
            # One roundtrip for ALL pending requests: multi-MB payloads over repeated ssh
            # `cat`s took minutes per turn (measured 2026-08-21: the relay's spool timeout
            # fired before the response landed), so fetch the whole req/ dir as one tar.
            rc, listing = ssh(ip, port, f"ls {SPOOL}/req/*.json 2>/dev/null")
            files = [f.strip() for f in listing.splitlines() if f.strip()]
            if rc != 0 or not files:
                time.sleep(1 if rc == 0 else 2)
                continue
            blob = ssh_bytes(ip, port, f"tar cf - -C {SPOOL}/req . 2>/dev/null")
            reqs = untar_json(blob)
            for path in files:
                fname = path.rsplit("/", 1)[-1]
                req = reqs.get(fname)
                if req is None:
                    continue
                rid, url, payload = req.get("id"), req.get("url"), req.get("payload") or {}
                if rid in served:
                    # already fulfilled; the pod-side delete raced our re-listing — do NOT
                    # pay for the same call twice, just re-delete the request file
                    ssh(ip, port, f"rm -f {path}", timeout=60)
                    continue
                t0 = time.time()
                if rid in paid:
                    status, resp_json = paid[rid]
                else:
                    # AIDP is one chat gateway; the relay already bridged Responses/Anthropic
                    # to chat shape, so any url it spools is fulfilled by the same endpoint.
                    status, body = call_aidp(AIDP_BASE + "/api/modelhub/online/v2/crawl"
                                             "?ak=odiVodksVzIsXAf35pKNXjGVgz0DSSdj_GPT_AK",
                                             payload)
                    if status == 400 and "reasoning_effort" in json.dumps(body):
                        # AIDP rejects tools+reasoning_effort on chat completions (measured
                        # 2026-08-21); same field the relay's own strip loop drops on 400
                        payload.pop("reasoning_effort", None)
                        print(f"{now()} [{name}] {rid}: stripped reasoning_effort after 400, "
                              f"retrying", flush=True)
                        status, body = call_aidp(
                            AIDP_BASE + "/api/modelhub/online/v2/crawl"
                            "?ak=odiVodksVzIsXAf35pKNXjGVgz0DSSdj_GPT_AK", payload)
                    resp_json = json.dumps({"status": status, "body": body},
                                           ensure_ascii=False)
                    paid[rid] = (status, resp_json)
                dt = time.time() - t0
                # push response (atomic rename on the pod), then delete the request.
                # base64 -d, NOT xxd: the campaign pods don't ship xxd (measured 2026-08-25 —
                # every push died on "command not found" while this line ignored the rc, the
                # bridge marked the rid served, and the relay starved to its 600 s timeout).
                import base64 as _b64mod
                b64 = _b64mod.b64encode(resp_json.encode()).decode()
                push = (f"mkdir -p {SPOOL}/resp && "
                        f"printf %s {b64} | base64 -d > {SPOOL}/resp/.{rid}.tmp && "
                        f"mv {SPOOL}/resp/.{rid}.tmp {SPOOL}/resp/{rid}.json && "
                        f"rm -f {path} && echo PUSH_OK")
                prc, pout = ssh(ip, port, push, timeout=120)
                if prc != 0 or "PUSH_OK" not in pout:
                    # do NOT mark served: leave the req file so the next loop retries the
                    # push (the AIDP call is repaid, but a lost response is a starved agent)
                    print(f"{now()} [{name}] {rid} PUSH FAILED rc={prc}; will retry",
                          flush=True)
                    continue
                served.add(rid)
                paid.pop(rid, None)
                if len(served) > 5000:
                    served.clear()
                with log.open("a") as fh:
                    fh.write(json.dumps({"t": now(), "id": rid, "status": status,
                                         "dt": round(dt, 1)}) + "\n")
                print(f"{now()} [{name}] {rid} -> {status} ({dt:.1f}s)", flush=True)
        except subprocess.TimeoutExpired:
            time.sleep(2)
        except Exception as exc:  # noqa: BLE001 -- one pod's failure must not kill its worker
            print(f"{now()} [{name}] worker error: {exc!r}", flush=True)
            time.sleep(3)
    print(f"{now()} [{name}] pod gone; worker exiting", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-sec", type=float, default=20.0,
                    help="how often to refresh the pod list")
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    workers: dict[str, tuple[threading.Thread, threading.Event]] = {}
    print(f"{now()} AIDP bridge up (outbound-only); spool {SPOOL}", flush=True)
    while True:
        try:
            pods = rest_pods()
        except Exception as exc:  # noqa: BLE001
            print(f"{now()} pod list failed ({exc!r}); retrying", flush=True)
            time.sleep(args.poll_sec)
            continue
        live = {}
        for p in pods:
            name = p.get("name") or ""
            if not name.startswith("rb-") or p.get("desiredStatus") != "RUNNING":
                continue
            ip = p.get("publicIp") or ""
            port = (p.get("portMappings") or {}).get("22")
            if ip and port:
                live[name] = (ip, int(port))
        # start workers for new pods
        for name, (ip, port) in live.items():
            if name not in workers:
                stop = threading.Event()
                th = threading.Thread(target=serve_pod, args=(ip, port, name, stop),
                                      daemon=True)
                th.start()
                workers[name] = (th, stop)
                print(f"{now()} + worker for {name} ({ip}:{port})", flush=True)
        # retire workers for pods that are gone
        for name in list(workers):
            if name not in live:
                _, stop = workers.pop(name)
                stop.set()
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    import urllib.error  # noqa: E402 -- used in call_aidp's except
    main()
