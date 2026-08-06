"""Live checkpoint-tree viewer: renders a session's persisted tree to a text file.

The render server persists the tree manifest to HDFS on every node/annotation change;
this watcher polls it and rewrites a human-readable rendering, so the file can stay
open in an editor as a live view of a running session.

Usage:
  python3 scripts/cosigen_tree_watch.py --session ikea-bifranka-v4 \
      --out CoSiGen/cosigen_eval_artifacts/ikea_bifranka_v4/tree_live.txt [--interval 60]
  python3 scripts/cosigen_tree_watch.py --session ... --once   # single render
"""
from __future__ import annotations

import argparse
import subprocess
import time

HDFS_ROOT = "hdfs://haruna/tmp/zeyu.shen/cosigen_sessions"


def render(session: str) -> tuple[str, str] | None:
    local = f"/tmp/tree_watch_{session}.pt"
    proc = subprocess.run(
        ["hdfs", "dfs", "-get", "-f", f"{HDFS_ROOT}/{session}/tree_manifest.pt", local],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    import torch
    p = torch.load(local, map_location="cpu", weights_only=False)
    nodes = p["nodes"]
    out = [f"session: {session} | {len(nodes)} nodes | current: {p['current']} | "
           f"turn: {p['turn']} | steps: {p['total_steps']} | "
           f"saved {time.strftime('%H:%M:%S', time.localtime(p['saved_at']))}",
           f"(rendered {time.strftime('%H:%M:%S')}; refreshes automatically)", ""]

    def walk(cid: str, ind: int) -> None:
        nd = nodes[cid]
        pad = "  " * ind
        ann = nd.get("ann") or {}
        cur = "   <== current" if cid == p["current"] else ""
        flags = (" [SUCCESS]" if nd.get("success") else "") + (" [CRASHED]" if nd.get("rc") else "")
        action = ann.get("action") or nd.get("label") or nd.get("caption") or ""
        parent = f"parent {nd['parent']}" if nd.get("parent") else "root"
        out.append(f"{pad}{cid} ({parent}, turn {nd['turn']}){flags}{cur}")
        out.append(f"{pad}  action: {action}")
        if ann.get("state_diff"):
            out.append(f"{pad}  state diff: {ann['state_diff']}")
        if ann.get("scene_diff"):
            out.append(f"{pad}  scene diff: {ann['scene_diff']}")
        out.append("")
        for ch in nd["children"]:
            walk(ch, ind + 1)

    for r in [c for c, nd in nodes.items() if nd["parent"] is None]:
        walk(r, 0)
    return "\n".join(out), p["current"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    while True:
        rendered = render(args.session)
        if rendered is not None:
            text, _current = rendered
            with open(args.out, "w") as f:
                f.write(text)
            print(f"[{time.strftime('%H:%M:%S')}] rendered -> {args.out}", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no persisted tree yet for {args.session}",
                  flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
