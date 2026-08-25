#!/usr/bin/env python3
"""PhysX Isaac grading ON MODAL GPUs (isaacsim 5.1 / isaaclab 2.3.2 / torch 2.7 cu128).

The Modal-native analogue of the RunPod grading: the exact Dockerfile.l0-isaaclab recipe
baked into a Modal image, robobench + the replay grader added, so grading runs on Modal GPUs
(no RunPod). Isaac's Modal-specific boot fixes are taken verbatim from modal_newton_env.py
(Vulkan ICD, X11 libs, crash-reporter off, file-not-`-c` boot, EULA env).

    modal run eval/scripts/modal_grade_env.py::verify         # prove PhysX Isaac boots on L4
    modal run eval/scripts/modal_grade_env.py::grade --label coffee_opus_5_r2 \
        --preset puzzle.coffee.franka.osc                     # grade one delivered solution
"""
from __future__ import annotations

import modal

REPO = "/repo"
VENV = "/opt/venv"
RUNS = "/runs"

app = modal.App("cosigen-grade")
runs_vol = modal.Volume.from_name("cosigen-runs", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu24.04", add_python="3.11")
    .apt_install("build-essential", "ca-certificates", "curl", "git", "jq",
                 "libglvnd0", "libgl1", "libglx0", "libegl1", "libgles2", "libvulkan1",
                 "vulkan-tools", "libx11-6", "libxt6", "libxrandr2", "libgomp1", "libglu1-mesa",
                 # X11 client libs Kit dlopens at startup (missing -> segfault on boot, per
                 # modal_newton_env's measured Modal fix)
                 "libsm6", "libice6", "libxext6", "libxi6", "libxrender1", "libxfixes3",
                 "libxcursor1", "libxinerama1")
    .pip_install("uv")
    .run_commands(
        f"uv venv --python 3.11 {VENV}",
        # torch pinned to cu128 (Dockerfile.l0-isaaclab)
        f"uv pip install --python {VENV}/bin/python torch==2.7.0 "
        f"--index-url https://download.pytorch.org/whl/cu128",
        # Isaac Sim 5.1 (biggest download, own layer)
        f"uv pip install --python {VENV}/bin/python 'isaacsim[all,extscache]==5.1.0' "
        f"--extra-index-url https://pypi.nvidia.com",
        # Isaac Lab 2.3.2 (flatdict sdist needs setuptools + no build isolation)
        f"uv pip install --python {VENV}/bin/python setuptools wheel",
        f"CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install --python {VENV}/bin/python "
        f"'isaaclab[all]==2.3.2' --extra-index-url https://pypi.nvidia.com "
        f"--no-build-isolation-package flatdict",
        f"uv pip install --python {VENV}/bin/python imageio imageio-ffmpeg",
    )
    .add_local_dir("robobench", remote_path=f"{REPO}/robobench")
    .add_local_dir("eval/grader", remote_path=f"{REPO}/eval/grader")
    .add_local_file("pyproject.toml", remote_path=f"{REPO}/pyproject.toml")
)


def _isaac_env() -> dict:
    import os
    return dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y",
                HOME="/root", NVIDIA_DRIVER_CAPABILITIES="all",
                PATH=f"{VENV}/bin:/usr/bin:/bin:/usr/local/bin",
                PYTHONPATH=REPO,
                OMNI_KIT_CRASH_REPORTER="0", CARB_CRASHREPORTER_ENABLED="0",
                OMNI_KIT_ALLOW_ROOT="1")


def _write_vulkan_icd() -> None:
    import json, os
    os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
    with open("/usr/share/vulkan/icd.d/nvidia_icd.json", "w") as f:
        json.dump({"file_format_version": "1.0.0",
                   "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194"}}, f)
    os.makedirs("/usr/share/glvnd/egl_vendor.d", exist_ok=True)
    with open("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "w") as f:
        json.dump({"file_format_version": "1.0.0",
                   "ICD": {"library_path": "libEGL_nvidia.so.0"}}, f)


@app.function(image=image, gpu="L4", timeout=40 * 60)
def verify() -> str:
    """Prove the PhysX Isaac stack boots + builds a robobench preset on a Modal GPU."""
    import subprocess
    _write_vulkan_icd()
    code = (
        "from isaaclab.app import AppLauncher\n"
        "app = AppLauncher(headless=True).app\n"
        "print('APP_UP', flush=True)\n"
        "import torch, robobench\n"
        "print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)\n"
        "robobench.discover()\n"
        "from robobench.core.registries import ENVS\n"
        "env = ENVS.get('puzzle.coffee.franka.osc')().build(num_envs=1, seed=0)\n"
        "print('BUILT', flush=True)\n"
        "env.reset(seed=0)\n"
        "print('RESET_OK score=%s' % float(env.scene.score().max()), flush=True)\n"
        "import os; os._exit(0)\n"
    )
    with open("/tmp/_verify.py", "w") as f:
        f.write(code)
    r = subprocess.run([f"{VENV}/bin/python", "/tmp/_verify.py"],
                       capture_output=True, text=True, env=_isaac_env())
    print("STDOUT:\n", r.stdout, "\nSTDERR(tail):\n", r.stderr[-4000:], flush=True)
    return "verified" if "RESET_OK" in r.stdout else "FAILED"


@app.function(image=image, gpu="L4", volumes={RUNS: runs_vol}, timeout=60 * 60)
def grade(label: str, preset: str, seed: int = 0) -> str:
    """Replay-grade one run's final solution from the volume on a Modal GPU (single-run proof)."""
    import subprocess, json, shutil
    from pathlib import Path
    _write_vulkan_icd()
    sol_src = Path(RUNS) / label / "workspace" / "solution"
    sol = Path("/tmp/sol")
    if sol.exists():
        shutil.rmtree(sol)
    if not (sol_src / "solve.py").exists():
        return f"{label}: no solve.py on volume"
    shutil.copytree(sol_src, sol)
    r = subprocess.run(
        [f"{VENV}/bin/python", f"{REPO}/eval/grader/grade_replay.py",
         "--preset", preset, "--solution", str(sol), "--seed", str(seed), "--max-seconds", "900"],
        capture_output=True, text=True, env=_isaac_env())
    out = r.stdout + "\n" + r.stderr[-2000:]
    verdict = next((ln for ln in r.stdout.splitlines() if ln.startswith("VERDICT")), None)
    print(out, flush=True)
    return verdict or f"{label}: no verdict\n{r.stderr[-500:]}"


@app.function(image=image, volumes={RUNS: runs_vol}, timeout=20 * 60)
def plan(runs_per_unit: int = 4) -> list:
    """Enumerate gradeable runs from the volume, group by preset, shard into GPU units."""
    import json
    from collections import defaultdict
    from pathlib import Path
    groups = defaultdict(list)
    skipped = graded = 0
    for d in sorted(Path(RUNS).iterdir()):
        if not d.is_dir():
            continue
        if (d / "replay_scores.json").exists():
            graded += 1
            continue
        rj = d / "run.json"
        subs = d / "submissions"
        if not rj.exists() or not subs.is_dir():
            skipped += 1
            continue
        try:
            preset = json.loads(rj.read_text()).get("preset")
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if preset:
            groups[preset].append(d.name)
        else:
            skipped += 1
    units = []
    for p, ls in sorted(groups.items()):
        for i in range(0, len(ls), runs_per_unit):
            units.append((p, ls[i:i + runs_per_unit]))
    print(f"plan: {sum(len(v) for v in groups.values())} runs, {len(units)} units; "
          f"{graded} already graded, {skipped} skipped", flush=True)
    return units


@app.function(image=image, gpu="L4", volumes={RUNS: runs_vol}, timeout=6 * 60 * 60,
              max_containers=25, retries=0)
def grade_unit(preset: str, labels: list, seed: int = 0, max_seconds: float = 600) -> str:
    """Grade every submission of these runs on THIS Modal GPU with the batched replay
    grader (watchdog + restart loop: a hung solve is killed at max_seconds, marked as a
    timeout, and the batch resumes past it)."""
    import json, shutil, subprocess
    from pathlib import Path
    _write_vulkan_icd()
    jobs, meta = [], {}
    for label in labels:
        base = Path(RUNS) / label
        if (base / "replay_scores.json").exists():
            continue
        subs = []
        sdir = base / "submissions"
        if sdir.is_dir():
            for d in sorted(sdir.iterdir()):
                if d.is_dir() and not d.name.startswith(".") and (d / "solve.py").exists():
                    wall = None
                    sj = d / "submitted.json"
                    if sj.exists():
                        try:
                            wall = json.loads(sj.read_text()).get("wall_s")
                        except Exception:  # noqa: BLE001
                            pass
                    stage = Path("/tmp/sols") / label / d.name
                    shutil.copytree(d, stage, dirs_exist_ok=True)
                    subs.append({"name": d.name, "dir": str(stage), "wall_s": wall})
        fsol = base / "workspace" / "solution"
        if (fsol / "solve.py").exists():
            stage = Path("/tmp/sols") / label / "final"
            shutil.copytree(fsol, stage, dirs_exist_ok=True)
            subs.append({"name": "final", "dir": str(stage), "wall_s": None})
        meta[label] = subs
        jobs += [{"label": label, "submission": s["name"], "dir": s["dir"]} for s in subs]
    if not jobs:
        return f"{preset}: nothing to grade in {labels}"
    man = Path("/tmp/jobs.json")
    man.write_text(json.dumps(jobs))
    out = Path("/tmp/grades.jsonl")
    # restart loop: watchdog kills the process on a hung solve; resume marks + skips it
    for attempt in range(len(jobs) + 2):
        r = subprocess.run(
            [f"{VENV}/bin/python", f"{REPO}/eval/grader/grade_replay_batch.py",
             "--preset", preset, "--manifest", str(man), "--out", str(out),
             "--seed", str(seed), "--max-seconds", str(max_seconds)],
            env=_isaac_env(), capture_output=True, text=True)
        print(f"[unit] batch attempt {attempt + 1} rc={r.returncode}\n"
              f"{r.stdout[-1500:]}\n{r.stderr[-800:]}", flush=True)
        if "BATCH DONE" in r.stdout:
            break
    graded = {}
    for ln in out.read_text().splitlines() if out.exists() else []:
        try:
            d = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if d.get("_start"):
            continue
        graded.setdefault(d["label"], {})[d["submission"]] = d
    done = []
    for label in labels:
        rows = []
        for s in meta.get(label, []):
            g = graded.get(label, {}).get(s["name"])
            if g is None:
                continue
            rows.append({"submission": s["name"], "wall_s": s["wall_s"],
                         "score": g.get("score"), "success": g.get("success"),
                         **({"reason": g["reason"]} if g.get("reason") else {})})
        if not rows:
            continue
        payload = {"label": label, "preset": preset, "seed": seed, "grader": "modal-L4",
                   "grades": rows,
                   "peak_score": max((r["score"] for r in rows if r["score"] is not None),
                                     default=0.0),
                   "ever_success": any(r["success"] for r in rows)}
        (Path(RUNS) / label / "replay_scores.json").write_text(
            json.dumps(payload, indent=2) + "\n")
        done.append(label)
    runs_vol.commit()
    return f"{preset}: graded {len(done)}/{len(labels)} runs ({len(jobs)} solutions)"


cpu_image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=cpu_image, volumes={RUNS: runs_vol}, timeout=30 * 60,
              max_containers=20)
def bundle_gt(label: str) -> str:
    """Server-side: tar this run's ground truth (workspace minus heavy/irrelevant dirs,
    submissions, run.json) to <label>/gt_bundle.tgz for a cheap local download."""
    import tarfile
    from pathlib import Path
    base = Path(RUNS) / label
    if not base.is_dir():
        return f"{label}: no run dir"
    out = base / "gt_bundle.tgz"
    skip_dirs = {".agent", ".checkpoints", ".footage", ".assessments", "tmp",
                 "__pycache__", ".git"}
    n = 0
    with tarfile.open(out, "w:gz") as tf:
        for sub in ("workspace", "submissions"):
            root = base / sub
            if not root.is_dir():
                continue
            for f in root.rglob("*"):
                if not f.is_file():
                    continue
                if any(part in skip_dirs for part in f.relative_to(base).parts):
                    continue
                if f.stat().st_size > 50 * 1024 * 1024:
                    continue  # no single GT file this big is a source/data file
                tf.add(f, arcname=str(f.relative_to(base)))
                n += 1
        if (base / "run.json").exists():
            tf.add(base / "run.json", arcname="run.json")
            n += 1
    runs_vol.commit()
    return f"{label}: bundled {n} files, {out.stat().st_size // 1024}KB"


@app.function(image=image, volumes={RUNS: runs_vol}, timeout=20 * 60)
def plan_v2(versions_per_unit: int = 72) -> list:
    """Enumerate runs with reconstructed version workspaces (<label>/versions.tgz) that do
    not yet have full-trajectory grades (<label>/grades_v2.json); group by preset and shard
    by TOTAL VERSION COUNT so every GPU unit carries a similar amount of replay work."""
    import json
    import tarfile
    from collections import defaultdict
    from pathlib import Path
    runs_vol.reload()  # see every commit made after this container was provisioned
    groups = defaultdict(list)  # preset -> [(label, n_versions)]
    graded = skipped = 0
    for d in sorted(Path(RUNS).iterdir()):
        if not d.is_dir() or not (d / "versions.tgz").exists():
            continue
        if (d / "grades_v2.json").exists():
            graded += 1
            continue
        rj = d / "run.json"
        try:
            preset = json.loads(rj.read_text()).get("preset") if rj.exists() else None
        except Exception:  # noqa: BLE001
            preset = None
        if not preset:
            skipped += 1
            continue
        try:
            with tarfile.open(d / "versions.tgz") as tf:
                vj = tf.extractfile("versions/versions.json")
                n = len(json.load(vj))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if n:
            groups[preset].append((d.name, n))
    units = []
    for p, ls in sorted(groups.items()):
        cur, cur_n = [], 0
        for label, n in ls:
            if cur and cur_n + n > versions_per_unit:
                units.append((p, cur))
                cur, cur_n = [], 0
            cur.append(label)
            cur_n += n
        if cur:
            units.append((p, cur))
    tot = sum(n for ls in groups.values() for _l, n in ls)
    print(f"plan_v2: {sum(len(v) for v in groups.values())} runs / {tot} versions "
          f"-> {len(units)} GPU units; {graded} already graded, {skipped} skipped", flush=True)
    return units


@app.function(image=image, gpu="L4", cpu=12, memory=36 * 1024,
              volumes={RUNS: runs_vol}, timeout=23 * 60 * 60,
              max_containers=40, retries=0)
def grade_unit_v2(preset: str, labels: list, seed: int = 0,
                  max_seconds: float = 1800, final_max_seconds: float = 3600,
                  workers: int = 2) -> str:
    """Grade EVERY reconstructed workspace version of these runs on this Modal GPU.

    Per label: untar versions.tgz, one job per version (a full workspace: solution/ +
    helpers + data files). `workers` batch graders run concurrently on the one GPU, each
    with a PRIVATE virtual /workspace (python-level path remap), replaying solve under
    verify_solution semantics, tracking the peak authoritative score, and recording the
    full env state after each run. Output per label -> <label>/grades_v2.json and
    <label>/states_v2/*.pt on the volume."""
    import json, shutil, subprocess, tarfile
    from pathlib import Path
    _write_vulkan_icd()
    jobs_by_label = {}
    vmeta = {}
    for label in labels:
        base = Path(RUNS) / label
        if (base / "grades_v2.json").exists():
            continue
        vroot = Path("/tmp/vers") / label
        if vroot.exists():
            shutil.rmtree(vroot)
        vroot.mkdir(parents=True)
        try:
            with tarfile.open(base / "versions.tgz") as tf:
                tf.extractall(vroot)
        except Exception as exc:  # noqa: BLE001
            print(f"[unit] {label}: versions.tgz unreadable: {exc!r}", flush=True)
            continue
        vdir = vroot / "versions"
        try:
            versions = json.loads((vdir / "versions.json").read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[unit] {label}: versions.json unreadable: {exc!r}", flush=True)
            continue
        vmeta[label] = versions
        # binary/data overlay (calibration .pt/.npy/... from the run, not reconstructable
        # from text): laid under each version, with the version's own files on top
        overlay = vdir / "overlay"
        ljobs = []
        for i, v in enumerate(versions):
            d = vdir / v["version"]
            if not d.is_dir():
                continue
            if overlay.is_dir():
                merged = vroot / f"m_{v['version']}"
                shutil.copytree(overlay, merged, dirs_exist_ok=True)
                shutil.copytree(d, merged, dirs_exist_ok=True)
                d = merged
            ljobs.append({"label": label, "submission": v["version"], "dir": str(d),
                          "closure_hash": v.get("closure_hash"),
                          "max_seconds": final_max_seconds if i == len(versions) - 1
                          else max_seconds})
        if ljobs:
            jobs_by_label[label] = ljobs
    if not jobs_by_label:
        return f"{preset}: nothing to grade in {labels}"

    # split whole labels across workers, balanced by version count (dedup cache is
    # per-label, so a label must not span workers)
    wjobs = [[] for _ in range(workers)]
    for label, ljobs in sorted(jobs_by_label.items(), key=lambda kv: -len(kv[1])):
        min(wjobs, key=lambda w: len(w)).extend(ljobs)
    wjobs = [w for w in wjobs if w]

    outs = [Path(f"/tmp/w{i}/grades.jsonl") for i in range(len(wjobs))]
    states = Path("/tmp/states")
    states.mkdir(exist_ok=True)
    # crash/preemption/timeout insurance: incremental grades + states are mirrored to the
    # volume every 5 minutes and reloaded on a fresh start; a shared cross-campaign seed
    # file recovers results from any earlier (differently-sharded) attempt.
    import hashlib as _hl
    import threading as _th
    unit_id = _hl.sha256((preset + "|".join(sorted(labels))).encode()).hexdigest()[:16]
    vol_tmp = Path(RUNS) / "_gradetmp" / unit_id
    seed_file = Path(RUNS) / "_gradetmp" / "_seed.jsonl"
    for i, out in enumerate(outs):
        out.parent.mkdir(parents=True, exist_ok=True)
        ckpt = vol_tmp / f"grades_w{i}.jsonl"
        if ckpt.exists():
            shutil.copy2(ckpt, out)
    if (vol_tmp / "states").is_dir():
        for f in (vol_tmp / "states").glob("*.pt"):
            shutil.copy2(f, states / f.name)

    def _mirror():
        try:
            vol_tmp.mkdir(parents=True, exist_ok=True)
            (vol_tmp / "states").mkdir(exist_ok=True)
            for i, out in enumerate(outs):
                if out.exists():
                    shutil.copy2(out, vol_tmp / f"grades_w{i}.jsonl")
            for f in states.glob("*.pt"):
                dst = vol_tmp / "states" / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
            runs_vol.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[unit] mirror failed: {exc!r}", flush=True)

    stop_mirror = _th.Event()

    def _mirror_loop():
        while not stop_mirror.wait(300):
            _mirror()

    _th.Thread(target=_mirror_loop, daemon=True).start()

    def run_worker(i: int):
        jobs = wjobs[i]
        wdir = Path(f"/tmp/w{i}")
        man = wdir / "jobs.json"
        man.write_text(json.dumps(jobs))
        out = outs[i]
        # restart loop: the watchdog writes a partial result then kills on a hung solve;
        # a segfault leaves a bare START marker; either way the resume skips past it.
        for attempt in range(len(jobs) + 2):
            cmd = [f"{VENV}/bin/python", f"{REPO}/eval/grader/grade_replay_batch.py",
                   "--preset", preset, "--manifest", str(man), "--out", str(out),
                   "--states-dir", str(states), "--stage-to", str(wdir / "ws"),
                   "--virtual-workspace",
                   "--seed", str(seed), "--max-seconds", str(max_seconds)]
            if seed_file.exists():
                cmd += ["--result-seed", str(seed_file)]
            r = subprocess.run(cmd, env=_isaac_env(), capture_output=True, text=True)
            print(f"[unit w{i}] batch attempt {attempt + 1} rc={r.returncode}\n"
                  f"{r.stdout[-1500:]}\n{r.stderr[-600:]}", flush=True)
            if "BATCH DONE" in r.stdout:
                break

    threads = [_th.Thread(target=run_worker, args=(i,)) for i in range(len(wjobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop_mirror.set()
    _mirror()
    graded = {}
    for out in outs:
        for ln in out.read_text().splitlines() if out.exists() else []:
            try:
                d = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if not d.get("_start"):
                graded.setdefault(d["label"], {})[d["submission"]] = d
    done = []
    for label in labels:
        rows = []
        for v in vmeta.get(label, []):
            g = graded.get(label, {}).get(v["version"])
            if g is None:
                continue
            rows.append({"version": v["version"], "ts_ms": v.get("ts_ms"),
                         "wall_min": v.get("wall_min"),
                         "closure_hash": v.get("closure_hash"),
                         "score": g.get("score"), "success": g.get("success"),
                         **({"reason": g["reason"]} if g.get("reason") else {}),
                         **({"state_file": g["state_file"]} if g.get("state_file") else {})})
        if not rows:
            continue
        base = Path(RUNS) / label
        sdst = base / "states_v2"
        sdst.mkdir(exist_ok=True)
        for row in rows:
            sf = row.get("state_file")
            if sf and (states / sf).exists():
                shutil.copy2(states / sf, sdst / sf)
        payload = {"label": label, "preset": preset, "seed": seed,
                   "grader": "modal-L4-full-workspace-v2",
                   "grades": rows,
                   "peak_score": max((r["score"] for r in rows if r["score"] is not None),
                                     default=0.0),
                   "ever_success": any(r["success"] for r in rows)}
        (base / "grades_v2.json").write_text(json.dumps(payload, indent=2) + "\n")
        done.append(label)
    if len(done) == len([l for l in labels if not (Path(RUNS) / l / "grades_v2.json").exists()]) \
            or vol_tmp.exists():
        shutil.rmtree(vol_tmp, ignore_errors=True)  # checkpoint no longer needed
    runs_vol.commit()
    return f"{preset}: graded {len(done)}/{len(labels)} runs ({len(jobs)} versions)"


@app.function(image=image, gpu="L4", volumes={RUNS: runs_vol}, timeout=4 * 60 * 60,
              max_containers=10, retries=0)
def grade_ckpt_unit(preset: str, labels: list) -> str:
    """Score every checkpoint-tree node state of these tool runs (no re-execution:
    env.reset -> set_states(node.pt) -> authoritative scorer). Output per label ->
    <label>/grades_ckpt.json on the volume."""
    import json, subprocess
    from pathlib import Path
    _write_vulkan_icd()
    runs_vol.reload()
    jobs = [{"label": l, "ckpt_dir": f"{RUNS}/_ckpt_kit/checkpoints/{l}"} for l in labels
            if (Path(RUNS) / "_ckpt_kit/checkpoints" / l / "tree.json").exists()]
    if not jobs:
        return f"{preset}: no checkpoint trees in {labels}"
    man = Path("/tmp/ckpt_jobs.json")
    man.write_text(json.dumps(jobs))
    out = Path("/tmp/ckpt_grades.jsonl")
    for attempt in range(3):
        r = subprocess.run(
            [f"{VENV}/bin/python", f"{REPO}/eval/grader/grade_checkpoints.py",
             "--preset", preset, "--manifest", str(man), "--out", str(out)],
            env=_isaac_env(), capture_output=True, text=True)
        print(f"[ckpt-unit] attempt {attempt + 1} rc={r.returncode}\n"
              f"{r.stdout[-2000:]}\n{r.stderr[-600:]}", flush=True)
        if "CKPT BATCH DONE" in r.stdout:
            break
    rows_by_label: dict = {}
    for ln in out.read_text().splitlines() if out.exists() else []:
        try:
            d = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        rows_by_label.setdefault(d["label"], []).append(d)
    done = []
    for label, rows in rows_by_label.items():
        scores = [r["score"] for r in rows if r.get("score") is not None]
        payload = {"label": label, "preset": preset, "grader": "modal-L4-ckpt-states",
                   "nodes": rows,
                   "peak_score": max(scores, default=0.0),
                   "ever_success": any(r.get("success") for r in rows)}
        base = Path(RUNS) / label
        base.mkdir(exist_ok=True)
        (base / "grades_ckpt.json").write_text(json.dumps(payload, indent=2) + "\n")
        done.append(label)
    runs_vol.commit()
    return f"{preset}: ckpt-graded {len(done)}/{len(labels)} runs " \
           f"({sum(len(v) for v in rows_by_label.values())} nodes)"


@app.local_entrypoint()
def main(step: str = "verify", label: str = "coffee_opus_5_r2",
         preset: str = "puzzle.coffee.franka.osc"):
    if step == "verify":
        print(verify.remote())
    elif step == "grade":
        print(grade.remote(label, preset))
    elif step == "campaign":
        units = plan.remote()
        print(f"spawning {len(units)} grade units on Modal L4s", flush=True)
        calls = [grade_unit.spawn(p, ls) for p, ls in units]
        for (p, ls), c in zip(units, calls):
            try:
                print(c.get(), flush=True)
            except Exception as exc:  # noqa: BLE001 -- one unit must not kill the campaign
                print(f"UNIT {p} {ls} FAILED: {exc!r}", flush=True)
    elif step == "unit_v2":
        # single-unit validation: one label end-to-end
        print(grade_unit_v2.remote(preset, [label]))
    elif step == "bundle":
        import sys
        labels = [ln.strip() for ln in open("/tmp/bundle_labels.txt")
                  if ln.strip()] if len(sys.argv) else []
        for res in bundle_gt.map(labels, return_exceptions=True):
            print(res, flush=True)
    elif step == "plan_only":
        units = plan_v2.remote()
        planned = {l for _p, ls in units for l in ls}
        probe = ["latte_opus_5", "latte_fable_5", "latte_gpt_5_6_sol", "latte_gpt_5_6_terra",
                 "latte_opus_4_8", "tshirt_opus_5", "tshirt_fable_5", "tshirt_gpt_5_6_sol",
                 "tshirt_gpt_5_6_terra", "tshirt_opus_4_8", "pc_gpu_fable_5_rtools2",
                 "pc_gpu_fable_5_rtools3", "pc_ram_fable_5_rtools2", "spatula_fable_5_rtools3",
                 "syringe_fable_5_rtools2", "syringe_fable_5_rtools3"]
        print(f"{len(planned)} labels planned; probe labels missing:")
        for l in probe:
            print(f"  {l}: {'IN PLAN' if l in planned else 'MISSING'}")
    elif step == "ckpt_campaign":
        import json as _json
        from collections import defaultdict
        man = _json.load(open("ckpt_kit/manifest.json"))
        groups = defaultdict(list)
        for l, m in sorted(man.items()):
            if m.get("preset") and not m.get("no_checkpoint_tree") and not m.get("error"):
                groups[m["preset"]].append(l)
        print(f"spawning {len(groups)} ckpt units "
              f"({sum(len(v) for v in groups.values())} runs)", flush=True)
        calls = [(p, grade_ckpt_unit.spawn(p, ls)) for p, ls in sorted(groups.items())]
        for p, c in calls:
            try:
                print(c.get(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"CKPT UNIT {p} FAILED: {exc!r}", flush=True)
    elif step == "campaign_v2":
        units = plan_v2.remote()
        print(f"spawning {len(units)} v2 grade units on Modal L4s", flush=True)
        calls = [grade_unit_v2.spawn(p, ls) for p, ls in units]
        for (p, ls), c in zip(units, calls):
            try:
                print(c.get(), flush=True)
            except Exception as exc:  # noqa: BLE001 -- one unit must not kill the campaign
                print(f"UNIT {p} {ls} FAILED: {exc!r}", flush=True)
