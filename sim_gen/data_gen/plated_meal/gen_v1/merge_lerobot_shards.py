"""Merge shard LeRobot repos (written by parallel convert_to_lerobot.py --shard i)
into one final dataset. v2.1 layout is per-episode parquet files (images embedded)
plus jsonl meta, so merging = renumber episode_index / global index and concatenate.

    HF_LEROBOT_HOME=/home/tiger/cap-x/simgen_bc/hf_lerobot uv run python \
        merge_lerobot_shards.py --out simgen/bc_v1 --shards simgen/bc_v1_shard0 ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HOME = Path(os.environ["HF_LEROBOT_HOME"])


def renumber(src: Path, dst: Path, new_ep: int, frame_base: int) -> int:
    """Copy an episode parquet with episode_index/index rewritten, PRESERVING the
    huggingface schema metadata (pandas round-trips strip it, which breaks image
    decoding on load)."""
    t = pq.read_table(src)
    meta = t.schema.metadata
    n = t.num_rows
    for name, values in (("episode_index", [new_ep] * n),
                         ("index", list(range(frame_base, frame_base + n)))):
        i = t.schema.get_field_index(name)
        t = t.set_column(i, name, pa.array(values, type=t.schema.field(name).type))
    if t.schema.metadata != meta:
        t = t.replace_schema_metadata(meta)
    pq.write_table(t, dst)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="simgen/bc_v1")
    ap.add_argument("--shards", nargs="+", required=True)
    args = ap.parse_args()

    out = HOME / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    shard_roots = [HOME / s for s in args.shards]
    for r in shard_roots:
        assert (r / "meta/info.json").exists(), f"missing shard {r}"

    # single task across the whole set -- verify, then reuse shard 0's tasks.jsonl
    tasks = {(r / "meta/tasks.jsonl").read_text() for r in shard_roots}
    assert len(tasks) == 1, "shards disagree on tasks.jsonl"
    (out / "meta/tasks.jsonl").write_text(tasks.pop())

    info = json.loads((shard_roots[0] / "meta/info.json").read_text())
    chunk_size = info["chunks_size"]

    ep_out = open(out / "meta/episodes.jsonl", "w")
    stats_out = open(out / "meta/episodes_stats.jsonl", "w")
    manifest_out = open(out / "manifest.jsonl", "w")

    new_ep = frame_base = n_dupes = 0
    seen: set[tuple[str, str]] = set()
    for r in shard_roots:
        eps = [json.loads(l) for l in (r / "meta/episodes.jsonl").read_text().splitlines()]
        stats = [json.loads(l) for l in
                 (r / "meta/episodes_stats.jsonl").read_text().splitlines()]
        man = [json.loads(l) for l in (r / "manifest.jsonl").read_text().splitlines()]
        assert len(eps) == len(stats) == len(man), f"{r}: meta line counts disagree"
        for ep, st, mn in zip(eps, stats, man):
            if (mn["batch"], mn["ep"]) in seen:  # converter re-shard overlap
                n_dupes += 1
                continue
            seen.add((mn["batch"], mn["ep"]))
            old = ep["episode_index"]
            assert st["episode_index"] == old
            src = r / f"data/chunk-{old // chunk_size:03d}/episode_{old:06d}.parquet"
            dst = out / f"data/chunk-{new_ep // chunk_size:03d}/episode_{new_ep:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            n = renumber(src, dst, new_ep, frame_base)
            assert n == ep["length"], (src, n, ep["length"])
            ep["episode_index"] = st["episode_index"] = new_ep
            ep_out.write(json.dumps(ep) + "\n")
            stats_out.write(json.dumps(st) + "\n")
            manifest_out.write(json.dumps(mn) + "\n")
            frame_base += n
            new_ep += 1
        print(f"[merge] {r.name}: +{len(eps)} episodes (total {new_ep})", flush=True)

    for f in (ep_out, stats_out, manifest_out):
        f.close()
    info.update(total_episodes=new_ep, total_frames=frame_base,
                total_chunks=(new_ep + chunk_size - 1) // chunk_size,
                splits={"train": f"0:{new_ep}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    shutil.copy(shard_roots[0] / "convention.json", out / "convention.json")
    print(f"[merge] DONE: {new_ep} episodes, {frame_base} frames "
          f"({n_dupes} shard-overlap dupes skipped) -> {out}", flush=True)


if __name__ == "__main__":
    main()
