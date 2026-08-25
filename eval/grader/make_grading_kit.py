#!/usr/bin/env python3
"""Package everything needed to grade all runs on an external cluster into ONE archive.

Collects, for every reconstructed run under results/:
    versions/<label>.tgz    the reconstructed per-version workspaces (from recon_pipeline)
    manifest.json           label -> {preset, versions, tgz_bytes}
    grader/                 grade_replay_batch.py + run_label_standalone.py
    robobench/              benchmark envs + assets (symlink dereferenced into the tar)
    pyproject.toml, README.md

Presets resolve from results/<label>/run.json, then results/<label>/gt/run.json, then
--inferred-presets (a json of label -> preset for first-campaign runs without run.json).

Usage:
    python3 eval/grader/make_grading_kit.py --out grading_kit.tgz
    # after reconstructing new runs with recon_pipeline.py, just rerun it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
README_SRC = Path(__file__).resolve().parent / "grading_kit_readme.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "grading_kit.tgz"))
    ap.add_argument("--inferred-presets", default=str(Path(__file__).resolve().parent / "inferred_presets.json"))
    ap.add_argument("--labels", default=None,
                    help="optional file of labels to include (default: every run with "
                         "results/<label>/versions.tgz)")
    args = ap.parse_args()

    inferred = {}
    if Path(args.inferred_presets).exists():
        inferred = json.loads(Path(args.inferred_presets).read_text())

    if args.labels:
        labels = [ln.strip() for ln in open(args.labels) if ln.strip()]
    else:
        labels = sorted(d.name for d in RESULTS.iterdir()
                        if (d / "versions.tgz").exists())

    manifest = {}
    skipped = []
    for label in labels:
        run_dir = RESULTS / label
        tgz = run_dir / "versions.tgz"
        vj = run_dir / "versions" / "versions.json"
        if not tgz.exists() or not vj.exists():
            skipped.append((label, "no versions.tgz/versions.json"))
            continue
        n = len(json.loads(vj.read_text()))
        if n == 0:
            skipped.append((label, "0 versions"))
            continue
        preset = None
        for cand in (run_dir / "run.json", run_dir / "gt" / "run.json"):
            if cand.exists():
                try:
                    preset = json.loads(cand.read_text()).get("preset")
                except Exception:  # noqa: BLE001
                    pass
                if preset:
                    break
        preset = preset or inferred.get(label)
        if not preset:
            skipped.append((label, "no preset"))
            continue
        manifest[label] = {"preset": preset, "versions": n,
                           "tgz_bytes": tgz.stat().st_size}

    kit = REPO / "grading_kit"
    if kit.exists():
        shutil.rmtree(kit)
    (kit / "versions").mkdir(parents=True)
    (kit / "grader").mkdir()
    (kit / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for label in manifest:
        shutil.copy2(RESULTS / label / "versions.tgz", kit / "versions" / f"{label}.tgz")
    for f in ("grade_replay_batch.py", "run_label_standalone.py", "rubrics.py"):
        shutil.copy2(Path(__file__).resolve().parent / f, kit / "grader" / f)
    if (REPO / "pyproject.toml").exists():
        shutil.copy2(REPO / "pyproject.toml", kit / "pyproject.toml")
    if README_SRC.exists():
        shutil.copy2(README_SRC, kit / "README.md")
    (kit / "robobench").symlink_to(REPO / "robobench")

    print(f"kit: {len(manifest)} labels, {sum(m['versions'] for m in manifest.values())} "
          f"versions; skipped: {skipped or 'none'}", flush=True)
    print(f"archiving -> {args.out} (robobench dereferenced; takes a few minutes)",
          flush=True)
    r = subprocess.run(["tar", "-czhf", args.out, "-C", str(REPO), "grading_kit"])
    if r.returncode != 0:
        sys.exit("tar failed")
    print(f"DONE: {args.out} ({Path(args.out).stat().st_size / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
