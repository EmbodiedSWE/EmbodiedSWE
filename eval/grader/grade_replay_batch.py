#!/usr/bin/env python3
"""Batch replay-grader: boot Isaac ONCE for a preset, grade many workspace versions.

Amortizes the ~1 min Isaac boot + env build across every graded code version of every run
that shares one preset. Each job is a FULL reconstructed workspace (solution/ plus every
helper module and data file the agent had at that moment, rebuilt from the agent history).
For each job the workspace is staged at --stage-to (default /workspace, so hardcoded
absolute paths inside agent code resolve exactly as they did during the run), the env is
reset(seed) (same initial condition each time, so scores are comparable), solve(env) runs
with reset/set_states blocked exactly like verify_solution.py, and we record:

  * the PEAK of the scene's own authoritative scorer over every step
    (rubric grader -> scene.score()/100 -> scene.success()),
  * scene.success() after the solve returns (the official verdict),
  * the FULL env state after the run (env.get_states() -> torch.save under --states-dir),
    so any later analysis can inspect exactly where every version left the scene.

Results are written incrementally to --out (one JSON line per job), so a crash mid-batch
keeps every grade already done. A solve that raises is a real failed grade (with its peak
score up to the exception) and the batch continues. A solve that HANGS inside Isaac's C
extensions is killed by the watchdog — which first writes the job's partial result with
the peak score reached so far, so even a hang is graded by its best achieved state.

    python grade_replay_batch.py --preset assembly.nut_thread.franka.osc \
        --manifest jobs.json --out results.jsonl --states-dir states/ --seed 0

    jobs.json: [{"label": "...", "submission": "v0007", "dir": "/tmp/vers/<label>/v0007"}, ...]
    each dir is a workspace root: solution/solve.py + helpers (dir/solve.py also accepted).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path


class _StepGraded:
    def __init__(self, env, measure, peak, rubric_fn=None, rubric_peak=None,
                 rubric_every: int = 10):
        self._env, self._measure, self._peak = env, measure, peak
        self._rubric_fn, self._rubric_peak = rubric_fn, rubric_peak
        self._rubric_every, self._n = rubric_every, 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action, render: bool = False):
        out = self._env.step(action, render)
        self._peak[0] = max(self._peak[0], self._measure())
        # fine-grained rubric peak: without this, a solve that makes real progress but
        # errors before returning leaves a near-reset END state and reads 0 partial
        # credit on every binary task (measured 2026-08-21). Sampled every N steps to
        # keep the get_states clone cost ~1% of runtime.
        if self._rubric_fn is not None:
            self._n += 1
            if self._n % self._rubric_every == 0:
                try:
                    r = self._rubric_fn(self._env.get_states())
                    self._rubric_peak[0] = max(self._rubric_peak[0], r)
                except Exception:  # noqa: BLE001 -- rubric must never break the replay
                    pass
        return out

    def reset(self, *a, **k):
        raise PermissionError("blocked during grading — solve may only step forward")

    def set_states(self, *a, **k):
        raise PermissionError("blocked during grading — solve may only step forward")


def _to_plain(obj):
    """Nested dict of tensors -> nested dict of lists (json/pt-agnostic snapshot)."""
    import torch
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _install_workspace_shim(stage: str) -> None:
    """Map literal '/workspace' paths to this worker's private stage dir at the Python
    level (open/io/os/os.path), so several graders can run on one GPU without fighting
    over the real /workspace. Agent code accesses files exclusively through these
    interfaces (open / torch.load / np.load / json.load / pathlib all bottom out here)."""
    import builtins
    import io
    import os

    def rw(p):
        if isinstance(p, bytes):
            s = p.decode(errors="surrogateescape")
            return rw(s).encode(errors="surrogateescape")
        if isinstance(p, str) and (p == "/workspace" or p.startswith("/workspace/")):
            return stage + p[len("/workspace"):]
        if hasattr(p, "__fspath__"):
            fs = os.fspath(p)
            if isinstance(fs, (str, bytes)):
                mapped = rw(fs)
                return mapped if mapped != fs else p
        return p

    _open = builtins.open

    def open_w(file, *a, **k):
        return _open(rw(file), *a, **k)
    builtins.open = open_w
    io.open = open_w

    for mod, names in ((os, ["stat", "lstat", "listdir", "scandir", "chdir", "mkdir",
                             "makedirs", "remove", "unlink", "rmdir", "open"]),
                       (os.path, ["exists", "isfile", "isdir", "getsize", "getmtime"])):
        for name in names:
            orig = getattr(mod, name, None)
            if orig is None:
                continue

            def make(orig):
                def wrapped(path, *a, **k):
                    return orig(rw(path), *a, **k)
                return wrapped
            setattr(mod, name, make(orig))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-dir", default=None,
                    help="record env.get_states() after every job here (label__version.pt)")
    ap.add_argument("--stage-to", default="/workspace",
                    help="stage each job's workspace here so absolute paths resolve")
    ap.add_argument("--virtual-workspace", action="store_true",
                    help="stage-to is private; remap literal /workspace paths to it")
    ap.add_argument("--result-seed", default=None,
                    help="jsonl of already-final results (from checkpoints of earlier "
                         "campaigns); matching jobs are skipped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=1800)
    args = ap.parse_args()

    if args.virtual_workspace:
        _install_workspace_shim(str(Path(args.stage_to).resolve()))

    jobs = json.loads(Path(args.manifest).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    states_dir = Path(args.states_dir) if args.states_dir else None
    if states_dir:
        states_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(args.stage_to)

    # Resume: a bare START marker with no matching result means the previous batch process
    # died mid-job. The watchdog now writes a partial result before killing, so a bare
    # START here means a crash harder than a hang (Isaac segfault); record it and move on.
    done = set()
    started = {}
    if args.result_seed and Path(args.result_seed).exists():
        with out.open("a") as fh:
            for line in Path(args.result_seed).read_text().splitlines():
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                key = (d.get("label"), d.get("submission"))
                if not d.get("_start") and key not in done and "score" in d:
                    fh.write(line + "\n")
                    done.add(key)
        print(f"[batch] seeded {len(done)} results from {args.result_seed}", flush=True)
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            key = (d.get("label"), d.get("submission"))
            if d.get("_start"):
                started[key] = d
            else:
                done.add(key)
        with out.open("a") as fh:
            for key, d in started.items():
                if key not in done:
                    fh.write(json.dumps({"label": key[0], "submission": key[1],
                                         "preset": args.preset, "success": False,
                                         "score": d.get("peak_at_kill", 0.0),
                                         "reason": "crashed on a previous attempt; not retried"}) + "\n")
                    done.add(key)
                    print(f"[batch] {key} marked crashed from previous attempt", flush=True)

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True)
    import importlib
    import importlib.util
    import torch  # noqa: F401  (needed for state serialization)
    import robobench
    from robobench.core.registries import ENVS
    robobench.discover()

    env = ENVS.get(args.preset)().build(num_envs=args.num_envs, seed=args.seed)
    scene = env.scene
    suite, scene_name = args.preset.split(".")[0], args.preset.split(".")[1]
    try:
        gmod = importlib.import_module(f"robobench.suites.{suite}.grader")
        gcls = getattr(gmod, "GRADERS", {}).get(scene_name)
    except Exception:  # noqa: BLE001
        gcls = None
    gr = [None]  # current rubric grader instance (rebuilt per job); None -> score()/success()

    def measure() -> float:
        if gr[0] is not None:
            v = gr[0].progress()
            return float(v.max() if hasattr(v, "max") else v)
        if hasattr(scene, "score"):
            s = scene.score()
            return float((s.float().max() if hasattr(s, "max") else float(s))) / 100.0
        s = scene.success()
        return float(s.max() if hasattr(s, "max") else s)

    def record_state(rec: dict) -> None:
        """env.get_states() after the run -> .pt next to the grades + summary in rec."""
        if states_dir is None:
            return
        try:
            states = env.get_states()
            fn = states_dir / f"{rec['label']}__{rec['submission']}.pt"
            torch.save({"states": states, "label": rec["label"],
                        "version": rec["submission"], "preset": args.preset,
                        "score": rec.get("score"), "success": rec.get("success")}, fn)
            rec["state_file"] = fn.name
        except Exception as exc:  # noqa: BLE001
            rec["state_error"] = f"{type(exc).__name__}: {exc}"[:150]
            print(f"[batch] state capture failed for {rec['label']}/{rec['submission']}: "
                  f"{exc!r}", file=sys.stderr)

    import threading
    fh = out.open("a")
    # identical programs replay identically (same build, same seed): grade each distinct
    # closure once per label and copy the result to its duplicates (A->B->A restores)
    by_closure: dict = {}
    for line in out.read_text().splitlines() if out.exists() else []:
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not d.get("_start") and d.get("closure_hash") and "score" in d:
            by_closure[(d["label"], d["closure_hash"])] = d
    for job in jobs:
        key = (job["label"], job["submission"])
        if key in done:
            continue
        ch = job.get("closure_hash")
        dup = by_closure.get((job["label"], ch)) if ch else None
        if dup is not None:
            rec = dict(dup)
            rec["submission"] = job["submission"]
            rec["dedup_of"] = dup["submission"]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            done.add(key)
            print(f"[batch] {key} = dedup of {dup['submission']} "
                  f"(score={rec.get('score')})", flush=True)
            continue
        rec = {"label": job["label"], "submission": job["submission"],
               "preset": args.preset, **({"closure_hash": ch} if ch else {})}
        job_max = float(job.get("max_seconds") or args.max_seconds)
        peak = [0.0]
        # START marker + per-solve watchdog. A hang inside Isaac's C extensions cannot be
        # interrupted from Python, so the watchdog writes the partial result (peak so far
        # and the timeout reason) itself, then kills the process; the restart loop resumes
        # past this job.
        fh.write(json.dumps({"label": key[0], "submission": key[1], "_start": True}) + "\n")
        fh.flush()

        def _on_hang(key=key, rec=rec, peak=peak, job_max=job_max):
            partial = dict(rec)
            partial.update(success=False, score=round(peak[0], 4),
                           reason=f"hang: exceeded {job_max:.0f}s, killed by watchdog "
                                  f"(score = peak before kill)")
            with out.open("a") as f2:
                f2.write(json.dumps(partial) + "\n")
            print(f"[batch] WATCHDOG: {key} exceeded {job_max:.0f}s — partial "
                  f"result written (peak={peak[0]:.4f}), killing process", flush=True)
            os._exit(3)

        watchdog = threading.Timer(job_max, _on_hang)
        watchdog.daemon = True
        watchdog.start()
        try:
            # ---- stage the full workspace where the agent's paths expect it
            src = Path(job["dir"])
            if stage.resolve() != src.resolve():
                if stage.exists():
                    shutil.rmtree(stage)
                shutil.copytree(src, stage)
            ws = stage if stage.resolve() != src.resolve() else src
            solve_py = ws / "solution" / "solve.py"
            sol_dir = ws / "solution"
            if not solve_py.is_file():
                solve_py, sol_dir = ws / "solve.py", ws
            if not solve_py.is_file():
                rec.update(success=False, score=0.0, reason="no solve.py")
                fh.write(json.dumps(rec) + "\n"); fh.flush(); continue

            env.reset(seed=args.seed)
            gr[0] = gcls(env) if gcls is not None else None  # fresh grader on the reset state
            peak[0] = measure()
            rubric_peak = [0.0]
            rubric_fn = None
            try:
                from rubrics import score_state as _rub
                _task = args.preset.split(".")[1]
                rubric_fn = (lambda st, _t=_task: _rub(_t, st))
                rubric_peak[0] = rubric_fn(env.get_states())
            except Exception:  # noqa: BLE001 -- no rubric for this task: official only
                rubric_fn = None
            g = _StepGraded(env, measure, peak, rubric_fn, rubric_peak)
            # CRITICAL isolation: this process grades many versions, each with its OWN
            # helper modules. Purge every module loaded from a staged workspace and every
            # stale workspace path from sys.path before loading this job's solve fresh.
            for _nm, _m in list(sys.modules.items()):
                _f = getattr(_m, "__file__", None) or ""
                if _f.startswith(str(stage)) or _f.startswith(str(src.parent)) \
                        or _nm.startswith("sol_"):
                    del sys.modules[_nm]
            sys.path[:] = [p for p in sys.path
                           if not p.startswith(str(stage)) and "/tmp/vers" not in p]
            sys.path.insert(0, str(ws))
            sys.path.insert(0, str(sol_dir))
            os.chdir(ws)
            spec = importlib.util.spec_from_file_location(
                f"sol_{job['label']}_{job['submission']}", solve_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "solve"):
                rec.update(success=False, score=round(peak[0], 4), reason="no solve(env)")
                record_state(rec)
                fh.write(json.dumps(rec) + "\n"); fh.flush(); continue
            mod.solve(g)
            succ = scene.success()
            rec.update(success=bool(succ.all() if hasattr(succ, "all") else succ),
                       score=round(max(peak[0], measure()), 4))
            if rubric_fn is not None:
                try:
                    rubric_peak[0] = max(rubric_peak[0], rubric_fn(env.get_states()))
                except Exception:  # noqa: BLE001
                    pass
                rec["rubric_peak"] = round(rubric_peak[0], 4)
        except Exception as exc:  # noqa: BLE001 -- a broken solve is a real failed grade
            rec.update(success=False, score=round(peak[0], 4),
                       reason=f"{type(exc).__name__}: {exc}"[:200])
            if rubric_fn is not None and rubric_peak[0] > 0:
                rec["rubric_peak"] = round(rubric_peak[0], 4)
            print(f"[batch] {key} failed: {exc!r}", file=sys.stderr)
            print(traceback.format_exc(limit=2), file=sys.stderr)
        finally:
            watchdog.cancel()
        record_state(rec)
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        if ch:
            by_closure[(job["label"], ch)] = rec
        print(f"[batch] {job['label']}/{job['submission']}: "
              f"success={rec.get('success')} score={rec.get('score')}", flush=True)
    fh.close()
    print("BATCH DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
