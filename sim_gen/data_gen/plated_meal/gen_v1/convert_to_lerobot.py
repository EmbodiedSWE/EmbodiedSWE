"""Convert generated plated_meal episodes (Option-B dumps on HDFS) into the LeRobot
dataset (DATA_SPEC Option A), through vla_training.lerobot_dataset.

Run from the openpi repo with its venv (the pinned lerobot) and HF_LEROBOT_HOME set:

    cd relevant_repos/openpi
    HF_LEROBOT_HOME=/home/tiger/cap-x/simgen_bc/hf_lerobot uv run python \
        /home/tiger/cap-x/CoSiGen/sim_gen/data_gen/plated_meal/gen_v1/convert_to_lerobot.py \
        --repo-id simgen/bc_v1 --batch-glob "prod_*"

Batches are staged from HDFS one at a time and deleted after conversion; every episode
adds one line to manifest.jsonl (mode/seed/params lineage) at the dataset root.

Parallel conversion: run N processes with --shard i --num-shards N (each owns the
batches whose sorted index % N == i, writing to its own repo-id), --loop makes a
process poll HDFS for new batches until the stop file exists; merge the shard repos
with merge_lerobot_shards.py afterwards.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zlib
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tiger/cap-x/CoSiGen")
sys.path.insert(0, "/home/tiger/cap-x")

from vla_training import lerobot_dataset as lds  # noqa: E402

HDFS_ROOT = "hdfs://haruna/tmp/zeyu.shen/simgen_bc/plated_meal_gen_v1"


def hdfs_ls(path: str) -> list[str]:
    out = subprocess.run(["hdfs", "dfs", "-ls", path], capture_output=True, text=True)
    return [l.split()[-1] for l in out.stdout.splitlines() if l.strip().endswith(("json", "npz"))
            or (l.startswith("d") and "/" in l)]


def my_batches(args) -> list[str]:
    names = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in hdfs_ls(HDFS_ROOT)
                   if fnmatch(p.rstrip("/").rsplit("/", 1)[-1], args.batch_glob))
    # stable hash: index-based sharding reassigns batches as the listing grows
    return [b for b in names
            if zlib.crc32(b.encode()) % args.num_shards == args.shard]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="simgen/bc_v1")
    ap.add_argument("--batch-glob", default="prod_*")
    ap.add_argument("--staging", default="/tmp/pm_convert_staging")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = no cap")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--loop", action="store_true",
                    help="poll HDFS for new batches until the stop file exists")
    ap.add_argument("--stop-file", default="/tmp/pm_conv_stop")
    args = ap.parse_args()

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    ds = lds.create_dataset(args.repo_id)
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    root = Path(HF_LEROBOT_HOME) / args.repo_id
    manifest = open(root / "manifest.jsonl", "a")

    n_eps = n_frames = 0
    done: set[str] = set()
    while True:
        pending = [b for b in my_batches(args) if b not in done]
        print(f"[conv] shard {args.shard}/{args.num_shards}: {len(pending)} pending "
              f"({len(done)} done)", flush=True)
        for b in pending:
            local = staging / b
            for stage_try in range(2):
                if local.exists():
                    shutil.rmtree(local)
                r = subprocess.run(["hdfs", "dfs", "-get", f"{HDFS_ROOT}/{b}", str(staging)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[conv] SKIP {b}: hdfs get failed: {r.stderr[-200:]}", flush=True)
                    break
                # a successful episode without frames.npz means we staged the batch
                # mid-upload (hdfs put is not atomic) — wait and re-stage once
                short = [d.name for d in sorted(local.glob("ep_*"))
                         if json.loads((d / "meta.json").read_text()).get("success")
                         and not (d / "frames.npz").exists()]
                if not short:
                    break
                print(f"[conv] {b}: {len(short)} episodes missing frames.npz "
                      f"({short[:3]}...) — "
                      f"{'re-staging in 120s' if stage_try == 0 else 'GIVING UP, will retry next loop'}",
                      flush=True)
                if stage_try == 0:
                    time.sleep(120)
            else:
                continue  # both stagings short: leave batch pending for the next loop
            if r.returncode != 0:
                continue
            done.add(b)
            for ep_dir in sorted(local.glob("ep_*")):
                meta = json.loads((ep_dir / "meta.json").read_text())
                if not meta.get("success"):
                    continue
                with np.load(ep_dir / "frames.npz") as d:
                    # materialize once: indexing the lazy NpzFile per frame re-decompresses
                    # the whole member each access (measured: O(T^2), 98% CPU in zipfile)
                    img, wrist = d["image"], d["wrist_image"]
                    state, act = d["state"], d["actions"]
                T = len(state)
                assert img.shape == (T, 256, 256, 3), img.shape
                assert np.isfinite(state).all() and np.isfinite(act).all()
                for t in range(T):
                    lds.add_step(ds, image=img[t], wrist_image=wrist[t],
                                 state=state[t], actions=act[t], task=meta["task"])
                ds.save_episode()
                manifest.write(json.dumps({"batch": b, "ep": ep_dir.name, "frames": T,
                                           **{k: meta[k] for k in meta
                                              if k not in ("task",)}}) + "\n")
                manifest.flush()
                n_eps += 1
                n_frames += T
                if n_eps % 25 == 0:
                    print(f"[conv] {n_eps} episodes, {n_frames} frames", flush=True)
            shutil.rmtree(local)
            if args.max_episodes and n_eps >= args.max_episodes:
                break
        if not args.loop or (args.max_episodes and n_eps >= args.max_episodes):
            break
        if Path(args.stop_file).exists() and not [b for b in my_batches(args)
                                                  if b not in done]:
            break
        time.sleep(120)

    shutil.copy("/home/tiger/cap-x/CoSiGen/sim_gen/data_gen/plated_meal/gen_v1/convention.json",
                root / "convention.json")
    manifest.close()
    print(f"[conv] DONE: {n_eps} episodes, {n_frames} frames -> {root}", flush=True)


if __name__ == "__main__":
    main()
