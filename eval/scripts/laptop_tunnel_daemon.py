#!/usr/bin/env python3
"""Laptop-side tunnel daemon: keeps one reverse SSH tunnel per live campaign pod.

For every RUNNING RunPod pod whose name starts with `rb-`, maintains

    ssh -N -R 8899:localhost:8898 -p <port> root@<ip>

so the pod's 127.0.0.1:8899 reaches the laptop's AIDP forwarder (laptop_aidp_forwarder.py on
127.0.0.1:8898). Dead tunnels are respawned on the next poll; tunnels of terminated pods are
closed. Run it under caffeinate so the laptop never sleeps mid-campaign:

    caffeinate -dimsu python3 eval/scripts/laptop_tunnel_daemon.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request

RUNPOD_KEY = "***REMOVED-SECRET***"
SSH_KEY = "~/.ssh/cosigen_campaign"
REMOTE_PORT = 8899          # pod-local port the relay talks to
LOCAL_PORT = 8898           # the forwarder's port on this laptop


def live_pods() -> dict[str, tuple[str, int]]:
    """pod_id -> (ip, ssh_port) for every RUNNING rb-* pod with an SSH mapping."""
    req = urllib.request.Request("https://rest.runpod.io/v1/pods",
                                 headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        pods = json.load(resp)
    out: dict[str, tuple[str, int]] = {}
    for p in pods:
        if (p.get("desiredStatus") == "RUNNING" and str(p.get("name", "")).startswith("rb-")
                and p.get("publicIp") and (p.get("portMappings") or {}).get("22")):
            out[p["id"]] = (p["publicIp"], int(p["portMappings"]["22"]))
    return out


def spawn_tunnel(pod_id: str, ip: str, port: int) -> subprocess.Popen:
    import os
    cmd = ["ssh", "-N",
           "-R", f"{REMOTE_PORT}:localhost:{LOCAL_PORT}",
           "-i", os.path.expanduser(SSH_KEY), "-p", str(port),
           "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ServerAliveInterval=15",
           "-o", "ServerAliveCountMax=4",
           "-o", "ExitOnForwardFailure=yes",
           "-o", "ConnectTimeout=20",
           f"root@{ip}"]
    # stderr kept per pod: rc=255 alone is undiagnosable (measured 2026-08-17)
    err = open(f"/tmp/tunnel_{pod_id}.err", "ab")  # noqa: SIM115 -- lives with the process
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=float, default=45.0)
    args = ap.parse_args()
    tunnels: dict[str, subprocess.Popen] = {}
    # Adopt no orphans: a tunnel from a previous daemon instance holds the pod-side port and
    # makes every new spawn fail its forward (measured 2026-08-17). One daemon, one owner.
    subprocess.run(["pkill", "-f", f"{REMOTE_PORT}:localhost:{LOCAL_PORT}"], check=False)
    time.sleep(2)
    print(f"[tunnels] daemon up: rb-* pods -> -R {REMOTE_PORT}:localhost:{LOCAL_PORT}",
          flush=True)
    while True:
        try:
            pods = live_pods()
        except Exception as exc:  # noqa: BLE001 -- a flaky poll must not kill the daemon
            print(f"[tunnels] pod list failed ({exc!r}); retrying next cycle", flush=True)
            time.sleep(args.poll_seconds)
            continue
        # close tunnels whose pod is gone
        for pod_id in list(tunnels):
            if pod_id not in pods:
                print(f"[tunnels] {pod_id}: pod gone, closing tunnel", flush=True)
                tunnels.pop(pod_id).terminate()
        # (re)open tunnels for live pods
        for pod_id, (ip, port) in pods.items():
            proc = tunnels.get(pod_id)
            if proc is not None and proc.poll() is None:
                continue
            if proc is not None:
                print(f"[tunnels] {pod_id}: tunnel died (rc={proc.returncode}), respawning",
                      flush=True)
            else:
                print(f"[tunnels] {pod_id}: opening tunnel to {ip}:{port}", flush=True)
            tunnels[pod_id] = spawn_tunnel(pod_id, ip, port)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
