#!/usr/bin/env python3
"""Local pipeline: for every run label, download the GT bundle from the volume, extract
submissions + final workspace, reconstruct the full workspace timeline from the agent
history, verify the reconstruction byte-level against the final workspace, package the
versions (+ a data-file overlay for things text reconstruction cannot produce), and
upload versions.tgz back to the volume for the Modal grading campaign.

    python3 eval/grader/recon_pipeline.py --labels /tmp/bundle_labels.txt --workers 6

Emits one JSON line per label to --report (default /tmp/recon_report.jsonl):
    {label, versions, fidelity{...}, check{exact,diff,missing,gt_total}, uploaded}
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_workspace import check_final, reconstruct  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
MODAL = "/Users/bytedance/Library/Python/3.9/bin/modal"
ENV = {"MODAL_PROFILE": "polaris-lab", "PATH": "/usr/bin:/bin:/usr/local/bin",
       "HOME": str(Path.home())}

OVERLAY_EXCLUDE_SUFFIX = {".log", ".pyc"}
OVERLAY_EXCLUDE_NAMES = {"submitted.json"}


def sh(cmd: list, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)


def process(label: str, skip_upload: bool = False, local_gt: bool = False) -> dict:
    run_dir = RESULTS / label
    run_dir.mkdir(exist_ok=True)
    rep: dict = {"label": label}
    gt = run_dir / "gt"

    if local_gt:
        if not gt.is_dir():
            rep["error"] = "local-gt mode but no gt/ prepared"
            return rep
    else:
        # 1) GT bundle
        bundle = run_dir / "gt_bundle.tgz"
        r = sh([MODAL, "volume", "get", "--force", "cosigen-runs",
                f"{label}/gt_bundle.tgz", str(bundle)])
        if r.returncode != 0 or not bundle.exists():
            rep["error"] = f"bundle download failed: {r.stderr.strip()[:120]}"
            return rep
        if gt.exists():
            shutil.rmtree(gt)
        gt.mkdir()
        with tarfile.open(bundle) as tf:
            tf.extractall(gt)
        bundle.unlink()

    # submissions for the reconstructor's resync + /submissions restores
    subs = run_dir / "submissions"
    if (gt / "submissions").is_dir():
        if subs.exists():
            shutil.rmtree(subs)
        shutil.copytree(gt / "submissions", subs)
    if (gt / "run.json").exists() and not (run_dir / "run.json").exists():
        shutil.copy2(gt / "run.json", run_dir / "run.json")

    # 2) reconstruct + 3) verify
    rep.update(reconstruct(label, RESULTS))
    if rep.get("error"):
        return rep
    if (gt / "workspace").is_dir():
        rep["check"] = check_final(label, RESULTS, gt / "workspace")
        rep["check"].pop("label", None)

    # 4) package: versions/ + overlay of run data files not reconstructed from text
    vdir = run_dir / "versions"
    overlay = vdir / "overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    fin = vdir / "final_state"
    reconstructed = {p.relative_to(fin).as_posix() for p in fin.rglob("*") if p.is_file()} \
        if fin.is_dir() else set()
    ws = gt / "workspace"
    n_overlay = 0
    if ws.is_dir():
        for f in ws.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(ws).as_posix()
            if rel in reconstructed or f.suffix in OVERLAY_EXCLUDE_SUFFIX \
                    or f.name in OVERLAY_EXCLUDE_NAMES or f.suffix == ".py":
                continue
            dst = overlay / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n_overlay += 1
    rep["overlay_files"] = n_overlay

    tgz = run_dir / "versions.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(vdir, arcname="versions")

    # 5) upload (+ run.json for labels that never reached the volume)
    if not skip_upload:
        r = sh([MODAL, "volume", "put", "--force", "cosigen-runs",
                str(tgz), f"{label}/versions.tgz"], timeout=1200)
        rep["uploaded"] = r.returncode == 0
        if r.returncode != 0:
            rep["upload_error"] = r.stderr.strip()[:120]
        if local_gt and (run_dir / "run.json").exists():
            sh([MODAL, "volume", "put", "--force", "cosigen-runs",
                str(run_dir / "run.json"), f"{label}/run.json"], timeout=300)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", default="/tmp/recon_report.jsonl")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--local-gt", action="store_true",
                    help="use pre-prepared results/<label>/gt instead of the volume bundle")
    args = ap.parse_args()
    labels = [ln.strip() for ln in open(args.labels) if ln.strip()]
    done = set()
    rp = Path(args.report)
    if rp.exists():
        for ln in rp.read_text().splitlines():
            try:
                done.add(json.loads(ln)["label"])
            except Exception:  # noqa: BLE001
                continue
    todo = [l for l in labels if l not in done]
    print(f"{len(todo)} to process ({len(done)} already in report)", flush=True)
    with rp.open("a") as out, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process, l, args.skip_upload, args.local_gt): l for l in todo}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                rep = fut.result()
            except Exception as exc:  # noqa: BLE001
                rep = {"label": label, "error": f"{type(exc).__name__}: {exc}"[:200]}
            out.write(json.dumps(rep) + "\n")
            out.flush()
            chk = rep.get("check", {})
            print(f"[{label}] versions={rep.get('versions')} "
                  f"exact={chk.get('exact')}/{chk.get('gt_total')} "
                  f"diff={len(chk.get('diff', []))} miss={len(chk.get('missing', []))} "
                  f"err={rep.get('error', '')}", flush=True)
    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
