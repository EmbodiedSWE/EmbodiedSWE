#!/usr/bin/env python3
"""Grade checkpoint-tree node states for tool runs.

Tool-run agents explored a TREE of saved env states (checkpoint_tree): each node's .pt is
the exact `env.get_states()` snapshot the agent reached by executing code along the path
from the root. Grading therefore needs NO re-execution: boot the env once per preset,
and for every node of every run: env.reset() -> env.set_states(node state) -> read the
scene's authoritative scorer. A run's checkpoint grade = max over ALL tree nodes
(internal nodes included — a parent can outscore its descendants).

    python grade_checkpoints.py --preset assembly.nut_thread.franka.osc \
        --manifest jobs.json --out results.jsonl

    jobs.json: [{"label": "...", "ckpt_dir": "/path/to/.checkpoints"}, ...]

Writes one JSON line per node: {label, cid, created, node_label, score, success}.
Node scoring uses the same measure as the replay grader (rubric grader progress ->
scene.score()/100 -> scene.success()), with the grader instantiated on the reset state
(as in a live episode) before the node state is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=1)
    args = ap.parse_args()

    jobs = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add((d.get("label"), d.get("cid")))
            except Exception:  # noqa: BLE001
                continue

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True)
    import importlib
    import torch
    import robobench
    from robobench.core.registries import ENVS
    robobench.discover()

    env = ENVS.get(args.preset)().build(num_envs=args.num_envs)
    scene = env.scene
    suite, scene_name = args.preset.split(".")[0], args.preset.split(".")[1]
    try:
        gmod = importlib.import_module(f"robobench.suites.{suite}.grader")
        gcls = getattr(gmod, "GRADERS", {}).get(scene_name)
    except Exception:  # noqa: BLE001
        gcls = None

    fh = out.open("a")
    for job in jobs:
        label = job["label"]
        ckpt = Path(job["ckpt_dir"])
        tree_file = ckpt / "tree.json"
        if not tree_file.exists():
            print(f"[ckpt] {label}: no tree.json", flush=True)
            continue
        try:
            nodes = json.loads(tree_file.read_text()).get("nodes", {})
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] {label}: tree.json unreadable: {exc!r}", flush=True)
            continue
        for cid, node in sorted(nodes.items()):
            if (label, cid) in done:
                continue
            rec = {"label": label, "cid": cid, "created": node.get("created"),
                   "node_label": (node.get("label") or "")[:120],
                   "tool_success_flag": bool(node.get("success"))}
            sp = ckpt / f"{cid}.pt"
            if not sp.is_file():
                rec.update(score=None, success=None, reason="no state file")
                fh.write(json.dumps(rec) + "\n"); fh.flush(); continue
            try:
                env.reset()
                grader = gcls(env) if gcls is not None else None
                states = torch.load(sp, map_location=env.device, weights_only=False)
                env.set_states(states)
                if grader is not None:
                    v = grader.progress()
                    score = float(v.max() if hasattr(v, "max") else v)
                elif hasattr(scene, "score"):
                    s = scene.score()
                    score = float((s.float().max() if hasattr(s, "max") else float(s))) / 100.0
                else:
                    s = scene.success()
                    score = float(s.max() if hasattr(s, "max") else s)
                succ = scene.success()
                rec.update(score=round(score, 4),
                           success=bool(succ.all() if hasattr(succ, "all") else succ))
            except Exception as exc:  # noqa: BLE001 -- a corrupt state is a real datum
                rec.update(score=None, success=None,
                           reason=f"{type(exc).__name__}: {exc}"[:200])
                print(f"[ckpt] {label}/{cid} failed: {exc!r}", flush=True)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[ckpt] {label}/{cid}: score={rec.get('score')} "
                  f"success={rec.get('success')}", flush=True)
    fh.close()
    print("CKPT BATCH DONE", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
