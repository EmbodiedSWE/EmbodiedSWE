#!/usr/bin/env python3
"""Render THE progress video of a CoSiGen session (2026-07-25 spec, no modes):

  checkpointed lineage (root -> current node)  +  latest attempt from that node

With no checkpoint tree / no checkpoints (base agent, or a tree still at root) the
lineage is empty, so the video is just the latest attempt. This one definition is what
every agent variant publishes, keeping base / ckpt / full apples-to-apples.

The run whose checkpoint created the current node is not shown twice: when the latest
recorded attempt IS that checkpointing run, the video is the lineage alone.

Frames come from the session's durable manifest (cosigen_sessions/<sid>/
frames_manifest.jsonl) and the exact edge ranges stored on tree nodes — recorded
footage only, no replay, no manual stitching.

Usage:
  python3 scripts/cosigen_session_video.py --session <sid> [--fps 8] --out video.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

SESSIONS_DIR = "hdfs://haruna/tmp/zeyu.shen/cosigen_sessions"


def _hdfs_cat(path: str) -> str:
    r = subprocess.run(["hdfs", "dfs", "-cat", path], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def manifest_attempts(session: str) -> list[dict]:
    """Manifest entries that recorded footage, in execution order — one entry per
    executed turn. NOTE: entry['turn'] is the SERVER's episode-local counter, which
    rewinds every reset; ordinal position is the only reliable attempt identifier."""
    text = _hdfs_cat(f"{SESSIONS_DIR}/{session}/frames_manifest.jsonl")
    out = []
    for line in text.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("parts"):
            out.append(d)
    return out


def checkpoint_path_slices(session: str) -> list[tuple[str, int, int]]:
    """Resolve the persisted root->current-checkpoint lineage to NPZ frame slices.

    Nodes store exact edge ranges in a server-local recording epoch; the durable
    frames manifest maps those ranges to uploaded NPZ parts."""
    tree_hdfs = f"{SESSIONS_DIR}/{session}/tree_manifest.pt"
    tree_local = f"/tmp/csv_tree_{re.sub(r'[^A-Za-z0-9_.-]', '_', session)}.pt"
    got = subprocess.run(["hdfs", "dfs", "-get", "-f", tree_hdfs, tree_local],
                         capture_output=True, text=True)
    if got.returncode != 0:
        raise SystemExit(f"cannot load checkpoint tree {tree_hdfs}: {got.stderr.strip()}")
    import torch

    payload = torch.load(tree_local, map_location="cpu", weights_only=False)
    nodes = payload.get("nodes") or {}
    current = payload.get("current")
    if not current or current not in nodes:
        raise SystemExit(f"checkpoint {current!r} not found; have {sorted(nodes)}")
    path = []
    cid = current
    while cid:
        path.append(nodes[cid])
        cid = nodes[cid].get("parent")
    path.reverse()

    manifest = _hdfs_cat(f"{SESSIONS_DIR}/{session}/frames_manifest.jsonl")
    parts_by_epoch: dict[str, list[dict]] = {}
    for line in manifest.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get("epoch")
        for part in row.get("parts") or []:
            if epoch and isinstance(part, dict) and {"path", "start", "n"} <= part.keys():
                parts_by_epoch.setdefault(epoch, []).append(part)
    for parts in parts_by_epoch.values():
        parts.sort(key=lambda p: int(p["start"]))

    slices: list[tuple[str, int, int]] = []
    for node in path[1:]:  # root has no incoming edge
        edge = node.get("edge_frames") or {}
        epoch = edge.get("epoch")
        lo, hi = int(edge.get("start", 0)), int(edge.get("end", 0))
        if hi <= lo:
            continue
        candidates = parts_by_epoch.get(epoch, [])
        if not candidates:
            # Pre-lineage checkpoint (recorded before edge_frames/indexed manifests
            # existed): its footage cannot be resolved natively; skip the edge.
            print(f"[lineage] {node['cid']}: no indexed footage for epoch "
                  f"{epoch!r} (pre-lineage edge, skipped)")
            continue
        covered = 0
        for part in candidates:
            p0, p1 = int(part["start"]), int(part["start"]) + int(part["n"])
            a, b = max(lo, p0), min(hi, p1)
            if b > a:
                slices.append((part["path"], a - p0, b - p0))
                covered += b - a
        if covered != hi - lo:
            raise SystemExit(
                f"incomplete video lineage for {node['cid']}: expected {hi-lo} frames, "
                f"resolved {covered}")
    print(f"[lineage] checkpoint={current} nodes={[n['cid'] for n in path]} "
          f"slices={len(slices)}")
    return slices


def progress_slices(session: str) -> list[tuple[str, int, int | None]]:
    try:
        lineage = checkpoint_path_slices(session)
    except SystemExit as exc:
        print(f"[progress] no checkpoint lineage ({exc}); latest attempt only")
        lineage = []
    slices: list[tuple[str, int, int | None]] = [(p, lo, hi) for p, lo, hi in lineage]

    attempts = manifest_attempts(session)
    if not attempts:
        if not slices:
            raise SystemExit("no recorded footage in the session manifest")
        return slices
    last = attempts[-1]
    parts = [p for p in last["parts"] if isinstance(p, dict)]
    # Skip the latest attempt when its footage is already the tail of the lineage
    # (i.e. it is the very run whose checkpoint created the current node).
    if lineage and parts:
        tail_path, _, tail_hi = lineage[-1]
        att_end = max(int(p["start"]) + int(p["n"]) for p in parts)
        tail_part = next((p for p in parts if p["path"] == tail_path), None)
        if tail_part is not None and tail_hi is not None:
            checkpointed_end = int(tail_part["start"]) + int(tail_hi)
            if att_end <= checkpointed_end:
                print("[progress] latest attempt == checkpointing run; lineage only")
                return slices
    print(f"[progress] lineage {len(lineage)} slice(s) + latest attempt "
          f"({last.get('n_frames')} frames in {len(parts)} part(s))")
    for p in last["parts"]:
        slices.append((p["path"] if isinstance(p, dict) else p, 0, None))
    return slices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session id")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slices = progress_slices(args.session)
    print(f"[encode] {len(slices)} slice(s) -> {args.out}")

    import imageio_ffmpeg
    import numpy as np
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc, total = None, 0
    for p, lo, hi in slices:
        loc = f"/tmp/csv_{os.path.basename(p)}"
        r = subprocess.run(["hdfs", "dfs", "-get", "-f", p, loc],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[encode] MISSING part {p}: {r.stderr.strip()[-160:]}")
            continue
        frames = np.load(loc)["frames"]
        os.remove(loc)
        frames = frames[int(lo):int(hi) if hi is not None else None]
        if not frames.size:
            continue
        if proc is None:
            h, w = frames.shape[1:3]
            # Encode into a temp file and swap it in when complete: encoding directly
            # into the target leaves an unplayable header-less file on disk for the
            # whole render, and live watchers re-render on a loop (2026-07-25).
            proc = subprocess.Popen(
                [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                 "yuv420p", "-movflags", "+faststart", args.out + ".rendering.mp4"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        proc.stdin.write(np.ascontiguousarray(frames[..., :3]).tobytes())
        total += int(frames.shape[0])
        print(f"[encode] +{frames.shape[0]:4d} frames ({os.path.basename(p)}"
              f"[{lo}:{hi if hi is not None else ''}]), total {total}")
    if proc is None:
        raise SystemExit("no frames decoded")
    proc.stdin.close()
    proc.wait()
    os.replace(args.out + ".rendering.mp4", args.out)
    print(f"[done] {args.out}: {total} frames = {total / args.fps:.0f}s at {args.fps} fps")


if __name__ == "__main__":
    main()
