#!/usr/bin/env python3
"""Aggregate a sharded BC eval sweep from HDFS and plot score vs checkpoint step.

Per checkpoint the sweep has shard dirs plated_meal_s<step>_k<shard>/ (and the
older unsharded runs plated_meal_s<step>/ are read too). Only summary/error
jsons are downloaded for every shard (tiny); VIDEOS are fetched only for
successful episodes, capped per checkpoint -- a full sync would blow up the
devbox disk (user directive 2026-08-05).

Writes per-step aggregates + the score-vs-step plot, and keeps a csv of every
episode for later slicing.

  python3 CoSiGen/vla_training/sync_bc_eval.py --ts 0805_XXXX
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

CAPX = Path(__file__).resolve().parents[2]
OUT_BASE = "hdfs://haruna/tmp/zeyu.shen/pi05_eval"
SCENE = "plated_meal"
CTRL_HZ = 15.0  # robobench franka OSC control rate; matches gen + eval
MAX_SUCCESS_VIDEOS = 3  # per checkpoint


def sh(cmd: list[str], timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def hdfs_names(path: str) -> list[str]:
    out = sh(["hdfs", "dfs", "-ls", path]).stdout
    return [l.rsplit("/", 1)[-1] for l in out.splitlines() if "/" in l]


def reencode_realtime(paths: list[Path]) -> None:
    """Old pods wrote fps=20 regardless of capture cadence; rewrite at wall-clock
    speed. New pods (video-every=1) already write fps=ctrl_hz and are left alone."""
    import imageio.v2 as imageio
    for v in paths:
        rd = imageio.get_reader(v)
        meta = rd.get_meta_data()
        fps = meta.get("fps", 0)
        if fps <= CTRL_HZ:
            rd.close()
            continue
        frames = [f for f in rd]
        rd.close()
        real = CTRL_HZ / 10.0  # old runs captured every 10th ctrl step
        tmp = v.with_suffix(".tmp.mp4")
        w = imageio.get_writer(str(tmp), fps=real, codec="libx264",
                               pixelformat="yuv420p", macro_block_size=1)
        for f in frames:
            w.append_data(f)
        w.close()
        tmp.replace(v)


def sync(ts: str) -> tuple[dict[int, list[dict]], int, int]:
    """-> ({step: [episode rows]}, shards_done, shards_total). Cached locally."""
    root = f"{OUT_BASE}/{ts}"
    local = CAPX / "simgen_bc/eval_artifacts" / f"sweep_{ts}"
    local.mkdir(parents=True, exist_ok=True)
    shard_dirs = [d for d in hdfs_names(root)
                  if re.match(rf"{SCENE}_s\d+(_k\d+)?$", d)]
    per_step: dict[int, list[dict]] = {}
    done = 0
    for d in sorted(shard_dirs):
        step = int(re.search(r"_s(\d+)", d).group(1))
        cache = local / f"{d}.summary.json"
        if not cache.exists():
            rdir = f"{root}/{d}"
            names = hdfs_names(rdir)
            if f"{SCENE}_summary.json" in names:
                r = sh(["hdfs", "dfs", "-get", "-f", f"{rdir}/{SCENE}_summary.json",
                        str(cache)])
                if r.returncode != 0:
                    continue
            elif f"{SCENE}_error.json" in names:
                r = sh(["hdfs", "dfs", "-get", "-f", f"{rdir}/{SCENE}_error.json",
                        str(cache)])
                if r.returncode != 0:
                    continue
                print(f"  {d}: ERROR run: "
                      f"{json.loads(cache.read_text()).get('error', '?')[:120]}")
            else:
                continue  # still running
        s = json.loads(cache.read_text())
        done += 1
        if "runs" not in s:
            continue  # error shard: counted done, contributes no episodes
        for run in s["runs"]:
            per_step.setdefault(step, []).append({**run, "shard_dir": d})
    return per_step, done, len(shard_dirs)


def fetch_success_videos(ts: str, per_step: dict[int, list[dict]]) -> None:
    root = f"{OUT_BASE}/{ts}"
    local = CAPX / "simgen_bc/eval_artifacts" / f"sweep_{ts}"
    for step, rows in sorted(per_step.items()):
        wins = [r for r in rows if r.get("success")][:MAX_SUCCESS_VIDEOS]
        got = []
        for r in wins:
            name = f"{SCENE}_ep{r['episode']}.mp4"
            dst = local / f"s{step}_success_seed{r.get('seed')}.mp4"
            if dst.exists():
                continue
            res = sh(["hdfs", "dfs", "-get", "-f",
                      f"{root}/{r['shard_dir']}/{name}", str(dst)], timeout=600)
            if res.returncode == 0:
                got.append(dst)
        if got:
            reencode_realtime(got)
            print(f"  s{step}: pulled {len(got)} success video(s)")


def report(ts: str, per_step: dict[int, list[dict]], done: int, total: int) -> None:
    local = CAPX / "simgen_bc/eval_artifacts" / f"sweep_{ts}"
    rows_csv = local / "episodes.csv"
    with open(rows_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "seed", "episode", "success", "score", "steps", "wall_s"])
        for step, rows in sorted(per_step.items()):
            for r in rows:
                w.writerow([step, r.get("seed"), r["episode"], int(r["success"]),
                            r.get("score"), r["steps"], r["wall_s"]])

    print(f"[sweep {ts}] shards done {done}/{total}")
    stats = []
    for step, rows in sorted(per_step.items()):
        n = len(rows)
        sr = sum(r["success"] for r in rows) / n
        sc = [r["score"] for r in rows if r.get("score") is not None]
        ms = sum(sc) / len(sc) if sc else None
        stats.append((step, n, sr, ms))
        print(f"  s{step}: n={n} success={sr:.2f} mean_score="
              f"{ms:.3f}" if ms is not None else f"  s{step}: n={n} success={sr:.2f}")

    if not stats:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random
    steps = [s for s, *_ in stats]
    ns = [n for _, n, *_ in stats]
    srs = [sr for *_, sr, _ in stats]
    mss = [ms for *_, ms in stats]
    fig, ax = plt.subplots(figsize=(10, 5))
    rng = random.Random(0)
    for step, rows in sorted(per_step.items()):
        xs = [step + rng.uniform(-220, 220) for _ in rows]
        ys = [r.get("score") for r in rows]
        ax.scatter(xs, ys, s=12, alpha=0.18, color="tab:gray", zorder=1)
    ax.plot(steps, mss, "o-", lw=2, color="tab:blue", label="mean score (graded)")
    ax.plot(steps, srs, "s--", lw=1.5, color="tab:red", label="success rate")
    for x, n in zip(steps, ns):
        ax.annotate(str(n), (x, -0.03), fontsize=6, ha="center", color="gray")
    ax.set_xlabel("training step (gray = episodes evaluated)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.07, 1.05)
    ax.set_title(f"pi05 BC on {SCENE}: {ts} sweep, {done}/{total} shards in")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = CAPX / "simgen_bc/plots" / f"bc_eval_sweep_{ts}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    print(f"wrote {png} and {rows_csv}")


def reap_finished_jobs() -> None:
    """Stop [pi05] eval jobs that are still RUNNING although their work is done.

    keepMins/orphaned uploaders can hold a pod's GPU after the eval concluded
    (user directive 2026-08-05: no zombie jobs, but never touch other work).
    Deliberately narrow, three independent guards:
      1. only jobs whose job_def_name starts with '[pi05] plated_meal'
      2. the job's own HDFS dir has a summary/error json (work provably done)
      3. that marker is >=3 min old (final uploads finished)
    """
    import time as _t

    def mcli(args: list[str], tries: int = 4) -> subprocess.CompletedProcess:
        """merlin-cli with retry: its subcommands register dynamically from a
        spec service, and when that fetch is throttled the CLI falls back to
        the static root and rejects --json ('unknown flag')."""
        for k in range(tries):
            r = sh(["merlin-cli", *args], timeout=300)
            blob = r.stdout + r.stderr
            if r.returncode == 0 and "unknown flag" not in blob:
                return r
            _t.sleep(15 * (k + 1))
        return r

    r = mcli(["job", "list-run", "--control-plane", "cn-seed",
              "--json", json.dumps({"current": 1, "pageSize": 400})])
    raw = r.stdout
    d = json.loads(raw[raw.find("{"):])

    def find_runs(o):
        if isinstance(o, list) and o and isinstance(o[0], dict) and "job_def_name" in o[0]:
            return o
        if isinstance(o, dict):
            for v in o.values():
                got = find_runs(v)
                if got:
                    return got
        return None

    runs = [x for x in (find_runs(d) or [])
            if x.get("status") == "RUNNING"
            and str(x.get("job_def_name", "")).startswith("[pi05] plated_meal")]
    if not runs:
        print("[reap] no running [pi05] jobs")
        return

    # job id -> out dir, from every launch manifest we ever wrote
    id2out: dict[str, str] = {}
    for mf in (CAPX / "simgen_bc/eval_artifacts").glob("launched_*.json"):
        for row in json.loads(mf.read_text()):
            if row.get("job"):
                id2out[row["job"]] = row["out"]

    sweeps = [l.rsplit("/", 1)[-1] for l in
              sh(["hdfs", "dfs", "-ls", OUT_BASE]).stdout.splitlines() if "/" in l]

    def marker_age_min(out_dir: str) -> float | None:
        out = sh(["hdfs", "dfs", "-ls", out_dir]).stdout
        for line in out.splitlines():
            if line.endswith((f"{SCENE}_summary.json", f"{SCENE}_error.json")):
                m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", line)
                if m:
                    ts = _t.mktime(_t.strptime(m.group(1), "%Y-%m-%d %H:%M"))
                    return (_t.time() - ts) / 60.0
        return None

    stopped = kept = 0
    for x in runs:
        jid = x["id"]
        name = str(x["job_def_name"])          # '[pi05] plated_meal_s25000_k3'
        run_dir = name.split("] ", 1)[1]
        cands = ([id2out[jid]] if jid in id2out
                 else [f"{OUT_BASE}/{ts}/{run_dir}" for ts in sweeps if ts != "_code"])
        age = next((a for a in (marker_age_min(c) for c in cands) if a is not None), None)
        if age is not None and age >= 3.0:
            res = mcli(["--control-plane", "cn-seed", "job", "stop-run",
                        "--json", json.dumps({"job_run_id": jid})])
            ok = res.returncode == 0 and "unknown flag" not in (res.stdout + res.stderr)
            print(f"[reap] {'stopped' if ok else 'STOP FAILED'} {jid} {name} "
                  f"(marker {age:.0f} min old)"
                  + ("" if ok else f": {(res.stdout + res.stderr).strip()[:160]}"))
            stopped += ok
        else:
            kept += 1
    print(f"[reap] stopped {stopped}, left running {kept} (no/fresh marker)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--reap", action="store_true",
                    help="stop RUNNING [pi05] jobs whose HDFS marker shows the "
                         "eval concluded >=3 min ago")
    args = ap.parse_args()
    per_step, done, total = sync(args.ts)
    if not args.no_videos:
        fetch_success_videos(args.ts, per_step)
    report(args.ts, per_step, done, total)
    if args.reap:
        reap_finished_jobs()


if __name__ == "__main__":
    main()
