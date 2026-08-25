#!/usr/bin/env python3
"""Standalone per-run grader for external clusters (e.g. L20 nodes).

Reproduces exactly what the Modal grade_unit_v2 does for one run label:
  1. untar versions/<label>.tgz (reconstructed workspaces, one per code version),
  2. one job per version: overlay data files + the version's own files,
     tiered timeouts (final version gets the official 3600s, intermediates 1800s),
  3. run grade_replay_batch.py in a restart loop (a hung solve is killed by the
     watchdog after writing its partial peak-score result; the loop resumes past it),
  4. assemble grades_v2.json (per-version score/success/state_file joined with the
     version timestamps) plus states_v2/ with the recorded env states.

Usage (one GPU per invocation; parallelize across labels):
    python run_label_standalone.py --label coffee_opus_5_r2 --kit /path/to/grading_kit \
        --out /path/to/grades_out
Requires: the environment in README.md (isaacsim 5.1 / isaaclab 2.3.2 / torch 2.7 cu128,
robobench on PYTHONPATH), and a writable /workspace (or pass --stage-dir).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--kit", required=True, help="grading_kit dir (manifest.json, versions/)")
    ap.add_argument("--out", required=True, help="output root; writes <out>/<label>/")
    ap.add_argument("--stage-dir", default="/workspace",
                    help="where each version is staged; /workspace makes agents' "
                         "hardcoded absolute paths resolve (recommended; needs root)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=1800)
    ap.add_argument("--final-max-seconds", type=float, default=3600)
    args = ap.parse_args()

    kit = Path(args.kit)
    manifest = json.loads((kit / "manifest.json").read_text())
    if args.label not in manifest:
        sys.exit(f"{args.label}: not in manifest")
    preset = manifest[args.label]["preset"]
    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "grades_v2.json").exists():
        print(f"{args.label}: grades_v2.json already exists, skipping")
        return

    work = out_dir / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    with tarfile.open(kit / "versions" / f"{args.label}.tgz") as tf:
        tf.extractall(work)
    vdir = work / "versions"
    versions = json.loads((vdir / "versions.json").read_text())

    overlay = vdir / "overlay"
    jobs = []
    for i, v in enumerate(versions):
        d = vdir / v["version"]
        if not d.is_dir():
            continue
        if overlay.is_dir():
            merged = work / f"m_{v['version']}"
            shutil.copytree(overlay, merged, dirs_exist_ok=True)
            shutil.copytree(d, merged, dirs_exist_ok=True)
            d = merged
        jobs.append({"label": args.label, "submission": v["version"], "dir": str(d),
                     "closure_hash": v.get("closure_hash"),
                     "max_seconds": args.final_max_seconds if i == len(versions) - 1
                     else args.max_seconds})
    if not jobs:
        sys.exit(f"{args.label}: no version workspaces")

    man = work / "jobs.json"
    man.write_text(json.dumps(jobs))
    grades = out_dir / "grades.jsonl"
    states = out_dir / "states_v2"
    states.mkdir(exist_ok=True)
    grader = Path(__file__).resolve().parent / "grade_replay_batch.py"
    for attempt in range(len(jobs) + 2):
        r = subprocess.run(
            [sys.executable, str(grader), "--preset", preset, "--manifest", str(man),
             "--out", str(grades), "--states-dir", str(states),
             "--stage-to", args.stage_dir,
             "--seed", str(args.seed), "--max-seconds", str(args.max_seconds)],
            text=True)
        print(f"[{args.label}] batch attempt {attempt + 1} rc={r.returncode}", flush=True)
        done = False
        for ln in grades.read_text().splitlines() if grades.exists() else []:
            pass
        # grade_replay_batch prints BATCH DONE and exits 0 when every job has a result
        if r.returncode == 0:
            done = True
        if done:
            break

    graded = {}
    for ln in grades.read_text().splitlines() if grades.exists() else []:
        try:
            d = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if not d.get("_start"):
            graded[d["submission"]] = d
    rows = []
    for v in versions:
        g = graded.get(v["version"])
        if g is None:
            continue
        rows.append({"version": v["version"], "ts_ms": v.get("ts_ms"),
                     "wall_min": v.get("wall_min"), "closure_hash": v.get("closure_hash"),
                     "score": g.get("score"), "success": g.get("success"),
                     **({"reason": g["reason"]} if g.get("reason") else {}),
                     **({"dedup_of": g["dedup_of"]} if g.get("dedup_of") else {}),
                     **({"state_file": g["state_file"]} if g.get("state_file") else {})})
    payload = {"label": args.label, "preset": preset, "seed": args.seed,
               "grader": "external-cluster-full-workspace-v2", "grades": rows,
               "peak_score": max((r["score"] for r in rows if r["score"] is not None),
                                 default=0.0),
               "ever_success": any(r["success"] for r in rows)}
    (out_dir / "grades_v2.json").write_text(json.dumps(payload, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    print(f"[{args.label}] DONE: {len(rows)}/{len(versions)} versions, "
          f"peak={payload['peak_score']}", flush=True)


if __name__ == "__main__":
    main()
