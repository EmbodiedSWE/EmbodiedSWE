#!/usr/bin/env python3
"""Replay-grade every delivered solution of every run on the cosigen-runs volume.

Reliable, reproducible scoring to replace the offline text heuristic: for each run, every
stored submission (and the final solution) is replayed from a FRESH reset on a GPU under the
scene's OWN authoritative scorer (rubric grader -> scene.score()/100 -> scene.success()), and
the PEAK is the state's score. Results persist as /<label>/replay_scores.json on the volume
(and mirrored locally), keyed so nothing needs re-grading later.

Substrate = RunPod GPU pods (the same Isaac the campaign used), one pod per PRESET so the
~1 min Isaac boot amortizes across every submission sharing that world. Orchestrated here in
parallel over --max-pods pods. Reuses run_agent_runpod's provisioning primitives and the
validated eval/grader/grade_replay_batch.py driver.

    python eval/scripts/grade_campaign.py --only-preset puzzle.coffee.franka.osc   # test
    python eval/scripts/grade_campaign.py --max-pods 20                            # all runs
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_agent_runpod as R

REPO = Path(__file__).resolve().parents[2]
SSH_KEY = str(Path.home() / ".ssh" / "cosigen_campaign")
MODAL = "/Users/bytedance/Library/Python/3.9/bin/modal"
MENV = {"MODAL_PROFILE": "polaris-lab", "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home())}
RESULTS = REPO / "results"
STAGE = Path("/tmp/grade_stage")
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def mvol(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run([MODAL, "volume", *args], capture_output=True, text=True,
                          timeout=timeout, env=MENV)


def vol_ls(path: str) -> list[str]:
    r = mvol("ls", "cosigen-runs", path)
    return [ln.strip().split("/")[-1] for ln in r.stdout.splitlines() if ln.strip()] \
        if r.returncode == 0 else []


def run_preset(label: str) -> str | None:
    """The run's preset, from run.json (local first, else the volume)."""
    local = RESULTS / label / "run.json"
    if local.exists():
        try:
            p = json.loads(local.read_text()).get("preset")
            if p:
                return p
        except Exception:  # noqa: BLE001
            pass
    dst = STAGE / label / "run.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mvol("get", "--force", "cosigen-runs", f"/{label}/run.json", str(dst)).returncode == 0:
        try:
            return json.loads(dst.read_text()).get("preset")
        except Exception:  # noqa: BLE001
            return None
    return None


def pull_submissions(label: str) -> list[dict]:
    """Pull every submission + the final solution locally. Returns [{name, dir, wall_s}]."""
    base = STAGE / label
    subs = []
    names = [n for n in vol_ls(f"/{label}/submissions") if n and not n.startswith(".")]
    for name in sorted(names):
        d = base / "submissions" / name
        d.mkdir(parents=True, exist_ok=True)
        # pull the whole submission dir (solve.py + sibling modules + submitted.json)
        mvol("get", "--force", "cosigen-runs", f"/{label}/submissions/{name}", str(d.parent))
        if (d / "solve.py").exists():
            wall = None
            sj = d / "submitted.json"
            if sj.exists():
                try:
                    wall = json.loads(sj.read_text()).get("wall_s")
                except Exception:  # noqa: BLE001
                    pass
            subs.append({"name": name, "dir": str(d), "wall_s": wall})
    # final solution
    fd = base / "final"
    fd.mkdir(parents=True, exist_ok=True)
    mvol("get", "--force", "cosigen-runs", f"/{label}/workspace/solution", str(fd.parent))
    fsol = fd.parent / "solution"
    if (fsol / "solve.py").exists():
        subs.append({"name": "final", "dir": str(fsol), "wall_s": None})
    return subs


def grade_group(preset: str, labels: list[str], args) -> str:
    """One pod grades every submission of every run sharing this preset."""
    # skip runs already graded
    todo = [l for l in labels if not (RESULTS / l / "replay_scores.json").exists()]
    if not todo:
        return f"{preset}: all {len(labels)} already graded"
    # gather submissions
    jobs, by_label = [], {}
    for label in todo:
        subs = pull_submissions(label)
        by_label[label] = subs
        for s in subs:
            jobs.append({"label": label, "submission": s["name"], "dir": s["dir"]})
    if not jobs:
        return f"{preset}: no submissions found for {len(todo)} runs"
    log(f"{preset}: {len(jobs)} solutions across {len(todo)} runs -> provisioning pod")

    ssh_pub = Path(SSH_KEY + ".pub").read_text().strip()
    pod = R.create_pod(f"grade-{preset.replace('.', '-')}"[:60], ssh_pub, SSH_KEY, gpu_count=1)
    try:
        R.provision_pod(pod)
        R.sh(pod, "mkdir -p /opt/cosigen/CoSiGen /grade", quiet=True)
        R.upload_dir(pod, REPO / "robobench", "/opt/cosigen/CoSiGen/robobench")
        R.upload_dir(pod, REPO / "eval" / "grader", "/opt/cosigen/CoSiGen/eval/grader")
        # upload all solution dirs under /grade/<label>/<submission>
        remote_jobs = []
        for j in jobs:
            remote = f"/grade/{j['label']}/{j['submission']}"
            R.sh(pod, f"mkdir -p {remote}", quiet=True)
            R.upload_dir(pod, Path(j["dir"]), remote)
            remote_jobs.append({"label": j["label"], "submission": j["submission"], "dir": remote})
        man = STAGE / f"manifest_{preset.replace('.', '_')}.json"
        man.write_text(json.dumps(remote_jobs))
        R.sh(pod, "mkdir -p /grade_manifests", quiet=True)
        # write manifest on the pod (base64 to avoid quoting issues)
        import base64
        b64 = base64.b64encode(man.read_bytes()).decode()
        R.sh(pod, f"echo {b64} | base64 -d > /grade_manifests/jobs.json", quiet=True)
        # run the batch grader
        rc, out = R.sh(
            pod,
            "cd /opt/cosigen/CoSiGen && OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y "
            "OMNI_KIT_ALLOW_ROOT=1 "
            f"{R.VENV}/bin/python eval/grader/grade_replay_batch.py --preset {preset} "
            f"--manifest /grade_manifests/jobs.json --out /grade_out.jsonl --seed {args.seed} "
            f"--max-seconds {args.max_seconds} 2>/grade.err; echo RC=$?; tail -5 /grade.err",
            timeout=args.pod_budget_min * 60)
        # fetch results
        rc2, res = R.sh(pod, "cat /grade_out.jsonl 2>/dev/null", quiet=True)
        graded = defaultdict(dict)
        for ln in res.splitlines():
            try:
                d = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            graded[d["label"]][d["submission"]] = {"score": d.get("score"),
                                                    "success": d.get("success"),
                                                    "reason": d.get("reason")}
        # assemble + persist per run
        done = 0
        for label in todo:
            rows = []
            for s in by_label.get(label, []):
                g = graded.get(label, {}).get(s["name"])
                if g is None:
                    continue
                rows.append({"submission": s["name"], "wall_s": s["wall_s"],
                             "score": g["score"], "success": g["success"],
                             **({"reason": g["reason"]} if g.get("reason") else {})})
            if not rows:
                continue
            out_local = RESULTS / label / "replay_scores.json"
            out_local.parent.mkdir(parents=True, exist_ok=True)
            payload = {"label": label, "preset": preset, "seed": args.seed, "grades": rows,
                       "peak_score": max((r["score"] for r in rows if r["score"] is not None),
                                         default=0.0),
                       "ever_success": any(r["success"] for r in rows)}
            out_local.write_text(json.dumps(payload, indent=2) + "\n")
            mvol("put", "--force", "cosigen-runs", str(out_local), f"/{label}/replay_scores.json")
            done += 1
        return f"{preset}: graded {done}/{len(todo)} runs (peak scores persisted)"
    finally:
        R.terminate_pod(pod)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-preset", default="")
    ap.add_argument("--labels", default="", help="comma-separated subset of run labels")
    ap.add_argument("--max-pods", type=int, default=12)
    ap.add_argument("--runs-per-pod", type=int, default=5,
                    help="pack whole runs into a pod up to this many (keeps a run's "
                         "submissions together; shards heavy presets across pods)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=1200, help="per-solve replay budget")
    ap.add_argument("--pod-budget-min", type=float, default=180, help="per-pod wall budget")
    args = ap.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)

    # target runs: those with a submissions/ dir on the volume
    if args.labels:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    else:
        root = [ln.strip() for ln in Path("/tmp/vol_all.txt").read_text().splitlines()
                if ln.strip()] if Path("/tmp/vol_all.txt").exists() else vol_ls("/")
        labels = root
    # map to presets
    groups = defaultdict(list)
    skipped = []
    for label in labels:
        p = run_preset(label)
        if p is None:
            skipped.append(label); continue
        if args.only_preset and p != args.only_preset:
            continue
        groups[p].append(label)
    # shard each preset's runs into pod-sized chunks (whole runs, never split a run)
    units = []  # (preset, [labels])
    for p, ls in sorted(groups.items()):
        ls = sorted(ls)
        for i in range(0, len(ls), args.runs_per_pod):
            units.append((p, ls[i:i + args.runs_per_pod]))
    log(f"grading {sum(len(v) for v in groups.values())} runs across {len(groups)} presets "
        f"in {len(units)} pod-units (\u2264{args.runs_per_pod} runs each, max {args.max_pods} "
        f"pods in flight); skipped {len(skipped)} without preset")

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.max_pods) as ex:
        futs = {ex.submit(grade_group, p, ls, args): (p, len(ls)) for p, ls in units}
        for fut in cf.as_completed(futs):
            try:
                results.append(fut.result())
                log("DONE " + results[-1])
            except Exception as exc:  # noqa: BLE001
                log(f"UNIT {futs[fut]} FAILED: {exc!r}")
    log("\n==== grade campaign complete ====")
    for r in results:
        log("  " + r)


if __name__ == "__main__":
    main()
