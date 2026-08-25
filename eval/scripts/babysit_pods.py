#!/usr/bin/env python3
"""Laptop-side safety net for campaign pods whose Modal launcher has died.

Modal has twice preempted launcher containers mid-campaign (2026-08-16), leaving pods
running with nobody enforcing the budget, mirroring artifacts, or terminating them. This
daemon watches every `rb-*` pod and — ONLY when the pod has clearly outlived its launcher's
care — does the launcher's teardown: stop the agent, mirror artifacts locally, terminate.

A pod is torn down when any of these hold:
  * its agent entry has EXITED but the pod is still up `grace_min` later (a live launcher
    tears down within minutes of the entry exiting);
  * its entry has been running longer than `budget_min` + `slack_min` (a live launcher stops
    the entry at budget; the slack covers the launcher's own stop+mirror window);
  * its entry never started and the pod is older than `setup_max_min` (provisioning wedged).

While a launcher is alive and doing its job, none of these fire.

    caffeinate -dimsu python3 eval/scripts/babysit_pods.py --budget-min 240
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

RUNPOD_KEY = "***REMOVED-SECRET***"
SSH_KEY = str(Path.home() / ".ssh" / "cosigen_campaign")
MIRROR_DIR = Path("/Users/bytedance/Desktop/CoSiGen/eval_result/babysit_mirror")
MIRROR_PATHS = ("workspace submissions opt/relay/trajlog opt/relay/relay.log "
                "opt/relay/container.log home/agent task")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rest(method: str, path: str):
    req = urllib.request.Request(f"https://rest.runpod.io/v1{path}",
                                 headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                                 method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def ssh(ip: str, port: int, cmd: str, timeout: float = 60.0) -> tuple[int, str]:
    proc = subprocess.run(
        ["ssh", "-i", SSH_KEY, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=20", f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout


def probe(ip: str, port: int) -> dict | None:
    """entry_alive + entry start epoch (container.log mtime) — None if SSH unreachable."""
    try:
        rc, out = ssh(ip, port,
                      'echo "alive=$(pgrep -f \'[a]gent-entry\' | wc -l | tr -d \' \')";'
                      'echo "start=$(stat -c %Y /opt/relay/container.log 2>/dev/null || echo 0)"')
    except subprocess.TimeoutExpired:
        return None
    if rc != 0:
        return None
    vals = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    return {"alive": vals.get("alive", "0") != "0", "start": int(vals.get("start") or 0)}


def teardown(pod_id: str, name: str, ip: str, port: int) -> bool:
    print(f"{now()} [{name}] teardown: stopping agent", flush=True)
    try:
        ssh(ip, port, "pkill -TERM -f '[a]gent-entry'; pkill -TERM -f '[c]laude'; "
                      "pkill -TERM -f '[c]odex'; sleep 20; "
                      "pkill -KILL -f '[a]gent-entry'; pkill -KILL -f '[c]laude'; "
                      "pkill -KILL -f '[c]odex'; true", timeout=120)
    except subprocess.TimeoutExpired:
        print(f"{now()} [{name}] stop timed out; mirroring anyway", flush=True)
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    out = MIRROR_DIR / f"{name}.tgz"
    print(f"{now()} [{name}] mirroring -> {out}", flush=True)
    try:
        with out.open("wb") as fh:
            proc = subprocess.run(
                ["ssh", "-i", SSH_KEY, "-p", str(port),
                 "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}",
                 f"tar czf - -C / {MIRROR_PATHS} 2>/dev/null; true"],
                stdout=fh, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"{now()} [{name}] mirror TIMED OUT — pod left up for manual salvage", flush=True)
        return False
    ok = (subprocess.run(["tar", "tzf", str(out)], capture_output=True).returncode == 0
          and out.stat().st_size > 1_000_000)
    if not ok:
        print(f"{now()} [{name}] mirror FAILED verification ({out.stat().st_size} bytes) — "
              f"pod left up for manual salvage", flush=True)
        return False
    rest("DELETE", f"/pods/{pod_id}")
    print(f"{now()} [{name}] mirror OK ({out.stat().st_size / 1e6:.0f} MB); pod terminated",
          flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=240.0)
    ap.add_argument("--slack-min", type=float, default=30.0,
                    help="grace beyond budget for a live launcher's own stop+mirror window")
    ap.add_argument("--grace-min", type=float, default=15.0,
                    help="how long an exited entry may sit before we assume the launcher died")
    ap.add_argument("--setup-max-min", type=float, default=75.0,
                    help="max age for a pod whose entry never started (wedged provisioning)")
    ap.add_argument("--poll-seconds", type=float, default=300.0)
    args = ap.parse_args()

    entry_dead_since: dict[str, float] = {}
    print(f"{now()} babysitter up: budget {args.budget_min}m +{args.slack_min}m slack, "
          f"entry-dead grace {args.grace_min}m, setup cap {args.setup_max_min}m", flush=True)
    while True:
        try:
            pods = rest("GET", "/pods")
        except Exception as exc:  # noqa: BLE001 -- a flaky poll must not kill the daemon
            print(f"{now()} pod list failed ({exc!r}); retrying", flush=True)
            time.sleep(args.poll_seconds)
            continue
        for p in pods:
            name = str(p.get("name", ""))
            if not name.startswith("rb-") or p.get("desiredStatus") != "RUNNING":
                continue
            pod_id, ip = p["id"], p.get("publicIp")
            port = (p.get("portMappings") or {}).get("22")
            if not ip or not port:
                continue
            state = probe(ip, int(port))
            if state is None:
                print(f"{now()} [{name}] ssh unreachable; will retry", flush=True)
                continue
            t = time.time()
            if state["start"] == 0:
                # entry never started: no reliable pod age via REST; track from first sight
                first = entry_dead_since.setdefault(f"boot:{pod_id}", t)
                if t - first > args.setup_max_min * 60:
                    print(f"{now()} [{name}] entry never started after "
                          f"{(t - first) / 60:.0f}m of watching — reaping", flush=True)
                    teardown(pod_id, name, ip, int(port))
                continue
            elapsed_min = (t - state["start"]) / 60
            if state["alive"]:
                entry_dead_since.pop(pod_id, None)
                if elapsed_min > args.budget_min + args.slack_min:
                    print(f"{now()} [{name}] entry {elapsed_min:.0f}m > budget+slack — "
                          f"launcher presumed dead; enforcing budget", flush=True)
                    teardown(pod_id, name, ip, int(port))
                else:
                    print(f"{now()} [{name}] running, {elapsed_min:.0f}m elapsed", flush=True)
            else:
                first = entry_dead_since.setdefault(pod_id, t)
                if t - first > args.grace_min * 60:
                    print(f"{now()} [{name}] entry exited {(t - first) / 60:.0f}m ago, pod "
                          f"still up — launcher presumed dead; tearing down", flush=True)
                    teardown(pod_id, name, ip, int(port))
                else:
                    print(f"{now()} [{name}] entry exited; grace "
                          f"{(t - first) / 60:.1f}/{args.grace_min}m", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
