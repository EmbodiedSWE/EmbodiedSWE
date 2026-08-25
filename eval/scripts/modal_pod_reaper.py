#!/usr/bin/env python3
"""Scheduled backstop: no campaign pod may outlive its budget, no matter what.

The launcher terminates its pod at teardown; this reaper exists for the case where the
launcher itself died. Every 15 minutes it lists RunPod pods and force-deletes any rb-*
pod older than MAX_LIFETIME (240 min budget + setup + teardown + margin). The template
pod (cosigen-probe-1) is never touched.

    modal deploy eval/scripts/modal_pod_reaper.py     # runs until `modal app stop`
"""

from __future__ import annotations

import modal

RUNPOD_KEY = "***REMOVED-SECRET***"
MAX_LIFETIME_MIN = 320          # 240 budget + ~15 setup + ~30 teardown/mirrors + margin

app = modal.App("cosigen-pod-reaper")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    schedule=modal.Period(minutes=15),
    timeout=300,
)
def reap() -> None:
    import json
    import urllib.request
    from datetime import datetime, timezone

    req = urllib.request.Request(
        "https://rest.runpod.io/v1/pods",
        headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        pods = json.loads(resp.read())

    now = datetime.now(timezone.utc)
    for p in pods:
        name = p.get("name", "")
        if not name.startswith("rb-") or p.get("desiredStatus") != "RUNNING":
            continue
        created = p.get("createdAt", "")
        try:
            # "2026-08-15 04:26:12.345 +0000 UTC"
            ts = datetime.strptime(created.split(".")[0], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc)
        except ValueError:
            print(f"cannot parse createdAt for {name}: {created!r}; skipping")
            continue
        age_min = (now - ts).total_seconds() / 60
        if age_min <= MAX_LIFETIME_MIN:
            continue
        print(f"REAPING {name} ({p['id']}): {age_min:.0f} min old > {MAX_LIFETIME_MIN}")
        dreq = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{p['id']}", method="DELETE",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"})
        try:
            urllib.request.urlopen(dreq, timeout=60)
            print(f"  deleted {p['id']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  DELETE FAILED for {p['id']}: {exc} — will retry next cycle")
