#!/usr/bin/env python3
"""Audit a CoSiGen session's video lineage end to end.

For EVERY checkpoint in the persisted tree (not just the current path):
  - does it carry native edge_frames lineage?
  - is its frame range fully covered by indexed parts in the frames manifest?
  - do the referenced NPZ parts actually exist on HDFS?

Exit code 0 = every lineage edge is fully renderable; nonzero otherwise.

Usage: python3 scripts/tests/audit_session_video.py <session_id>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

SESSIONS_DIR = "hdfs://haruna/tmp/zeyu.shen/cosigen_sessions"


def main() -> None:
    session = sys.argv[1]
    local = f"/tmp/audit_tree_{re.sub(r'[^A-Za-z0-9_.-]', '_', session)}.pt"
    got = subprocess.run(
        ["hdfs", "dfs", "-get", "-f", f"{SESSIONS_DIR}/{session}/tree_manifest.pt", local],
        capture_output=True, text=True)
    if got.returncode != 0:
        sys.exit(f"no persisted tree for {session}: {got.stderr.strip()[-200:]}")
    import torch
    payload = torch.load(local, map_location="cpu", weights_only=False)
    nodes = payload.get("nodes") or {}

    manifest = subprocess.run(
        ["hdfs", "dfs", "-cat", f"{SESSIONS_DIR}/{session}/frames_manifest.jsonl"],
        capture_output=True, text=True).stdout
    parts_by_epoch: dict[str, list[dict]] = {}
    for line in manifest.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for part in row.get("parts") or []:
            if row.get("epoch") and isinstance(part, dict) and "start" in part:
                parts_by_epoch.setdefault(row["epoch"], []).append(part)

    bad = 0
    checked_paths: set[str] = set()
    for cid in sorted(nodes, key=lambda c: int(c[1:])):
        node = nodes[cid]
        edge = node.get("edge_frames") or {}
        lo, hi = int(edge.get("start", 0)), int(edge.get("end", 0))
        if not edge or hi <= lo:
            status = "no-footage-edge" if not edge else "empty-edge (0 frames)"
            print(f"  {cid}: {status}")
            continue
        cands = parts_by_epoch.get(edge.get("epoch"), [])
        covered = 0
        paths = []
        for part in cands:
            p0, p1 = int(part["start"]), int(part["start"]) + int(part["n"])
            a, b = max(lo, p0), min(hi, p1)
            if b > a:
                covered += b - a
                paths.append(part["path"])
        ok = covered == hi - lo
        exists_fail = []
        for p in paths:
            if p not in checked_paths:
                checked_paths.add(p)
                r = subprocess.run(["hdfs", "dfs", "-test", "-e", p], capture_output=True)
                if r.returncode != 0:
                    exists_fail.append(p)
        if not ok or exists_fail:
            bad += 1
            print(f"  {cid}: FAIL edge {lo}-{hi}: covered {covered}/{hi-lo}"
                  + (f", missing parts {exists_fail}" if exists_fail else ""))
        else:
            print(f"  {cid}: OK edge {lo}-{hi} ({hi-lo} frames, {len(paths)} part(s))")
    print(f"[audit] {session}: {len(nodes)} checkpoints, {bad} broken lineage edges")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
