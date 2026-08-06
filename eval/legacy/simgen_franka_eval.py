#!/usr/bin/env python3
"""General FRANKA eval pipeline over ALL sim_gen constructed problems.

Evaluates an agent on every ACCEPTED task from the sim_gen campaign ledger, with the
scene bound to a REAL Franka arm (env '<suite>.<scene>.franka' — synthesized by
cosigen_loop.build_env; the agent must physically manipulate, no null-robot/teleport
solutions). Reuses the proven machinery end to end: launch_cosigen_render_pool.py for
pods, cosigen_harness.py (same invocation shape as the seed-30b sim_gen evals and the
opus IKEA runs) for the agent loop, recording always on (--video).

Stages (run in order; each is idempotent):
  --push-tarball        rebuild + push cosigen.tar.gz (patched cosigen_loop + all tasks)
  --launch [--batch N --offset K]   submit render pods for the next N tasks
  --run    [--batch N --offset K]   wait for pods to boot, then drive one harness agent
                                    per task (parallel), artifacts under
                                    CoSiGen/cosigen_eval_artifacts/simgen_franka_eval/
  --status              one-line-per-task progress/success table
  --list                show the task -> env mapping and exit

Example (first batch of 10):
  python3 scripts/simgen_franka_eval.py --push-tarball
  python3 scripts/simgen_franka_eval.py --launch --batch 10 --offset 0
  python3 scripts/simgen_franka_eval.py --run --batch 10 --offset 0
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO / "CoSiGen" / "sim_gen" / "tasks"
LEDGER = REPO / "CoSiGen" / "sim_gen" / "artifacts" / "campaign" / "ledger.jsonl"
ART_ROOT = REPO / "CoSiGen" / "cosigen_eval_artifacts" / "simgen_franka_eval"
REGISTRY_DIR = "hdfs://haruna/tmp/zeyu.shen/cosigen_render"
TARBALL_HDFS = "hdfs://haruna/tmp/zeyu.shen/cosigen.tar.gz"
MODEL = "seed-30b-relay"  # same model preset as the existing sim_gen null-robot evals

EXTRA_PROMPT = (
    "You control a REAL Franka arm (OSC end-effector control via the toolkit: move_to, "
    "look, checkpoint/goto) — success is graded on PHYSICAL manipulation with "
    "the gripper. Do NOT teleport task objects into goal states; write robot control "
    "code. Checkpoint before risky maneuvers; goto() instead of repeating failures. "
    "Tune contact-critical constants with the optimize tool instead of hand-guessing "
    "values run after run."
)


def load_tasks() -> list[dict]:
    """Accepted campaign tasks -> {task, tier, env (franka-bound), env_base}."""
    if not LEDGER.is_file():
        sys.exit(f"no ledger at {LEDGER}")
    seen, out = set(), []
    for line in LEDGER.read_text().splitlines():
        e = json.loads(line)
        task = e.get("task")
        if not e.get("accepted") or task in seen:
            continue
        seen.add(task)
        scene_py = TASKS_ROOT / task / "scene.py"
        if not scene_py.is_file():
            print(f"[tasks] WARNING: accepted {task} has no scene.py — skipped")
            continue
        m = re.search(r'register_env\(\s*"(sim_gen|simgen)"[^)]*scene\s*=\s*"([^"]+)"',
                      scene_py.read_text())
        if not m:
            print(f"[tasks] WARNING: no register_env match in {task}/scene.py — skipped")
            continue
        suite, scene = m.group(1), m.group(2)
        out.append({"task": task, "tier": e.get("tier"),
                    "env_base": f"{suite}.{scene}", "env": f"{suite}.{scene}.franka"})
    return out


def sel(tasks: list[dict], batch: int, offset: int) -> list[dict]:
    return tasks[offset:offset + batch] if batch else tasks[offset:]


def push_tarball() -> None:
    """Rebuild + push cosigen.tar.gz (same content policy as the existing tarball:
    excludes eval artifacts + campaign artifacts; includes sim_gen tasks + RoboVerse)."""
    tmp = "/tmp/cosigen_rebuild.tar.gz"
    cmd = ["tar", "czf", tmp,
           "--exclude=CoSiGen/cosigen_eval_artifacts",
           "--exclude=CoSiGen/sim_gen/artifacts",
           # Our own runs' output: every turn's program for these very tasks, including
           # the ones that worked. Nothing on a pod reads it, a live run rewrites it
           # while tar reads (which is what made this build fail), and an agent that
           # found it would be reading another agent's answer to its own task.
           "--exclude=CoSiGen/experiments",
           "--exclude=.git", "--exclude=__pycache__",
           # HARD EXCLUSION (user directive, violated once on 2026-07-24 at great cost):
           # worked solutions must NEVER ship to a pod running an agent under
           # evaluation. Whole trees, not per-task filtering — see cosigen_config's
           # leakage guard, which removes the same set again on the pod.
           "--exclude=CoSiGen/eval/examples",
           "--exclude=CoSiGen/references",
           "--exclude=CoSiGen/robobench/suites/*/smokes",
           "--exclude=*solve_ikea*",
           "CoSiGen"]
    print("[tarball] building:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    print("[tarball] pushing to", TARBALL_HDFS)
    subprocess.run(["hdfs", "dfs", "-put", "-f", tmp, TARBALL_HDFS], check=True)
    print("[tarball] done")


def launch(tasks: list[dict], num_envs: int, pool: str) -> None:
    envs = [t["env"] for t in tasks]
    print(f"[launch] submitting {len(envs)} render pods: {envs}")
    subprocess.run([sys.executable, str(REPO / "scripts" / "launch_cosigen_render_pool.py"),
                    "--num-envs", str(num_envs), "--pool", pool, *envs], check=True)


def _server_url(env: str) -> str | None:
    r = subprocess.run(["hdfs", "dfs", "-cat", f"{REGISTRY_DIR}/{env}.txt"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def _ping(url: str) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/ping", timeout=8) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _session_scores(sdir: Path) -> dict:
    """Best/final partial-credit score + success across a session's turn jsons."""
    best, final, success, turns = 0.0, 0.0, False, 0
    for f in sorted(sdir.glob("turn_*.json")):
        turns += 1
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        s = d.get("score")
        if isinstance(s, (int, float)):
            best, final = max(best, float(s)), float(s)
        if d.get("success"):
            success, best, final = True, 1.0, 1.0
    return {"best_score": round(best, 4), "final_score": round(final, 4),
            "success": success, "turns": turns}


def supervise_task(t: dict, max_steps: int, sessions: int, session_wall_s: float,
                   wait_min: float) -> None:
    """score@k protocol: run `sessions` INDEPENDENT full agent sessions on this task's
    pod (serial — a pod hosts one live session), each with a sim-step budget and a
    wall-clock cap; auto --resume within a session on driver crashes; append each
    finished session's scores to <task>/scores.jsonl."""
    task = t["task"]
    adir = ART_ROOT / task
    adir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + wait_min * 60
    url = None
    while time.time() < deadline:
        url = _server_url(t["env"])
        if url and _ping(url).get("booted"):
            break
        url = None
        time.sleep(60)
    if not url:
        print(f"[{task}] pod never booted; giving up", flush=True)
        return
    # make sure the pod runs the CURRENT harness (score payload, view fix, RL fixes)
    try:
        import urllib.request
        req = urllib.request.Request(url + "/reload", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=150).read())
        print(f"[{task}] pod reload -> {r.get('ok')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{task}] pod reload failed ({exc!r}); continuing", flush=True)

    scores_path = adir / "scores.jsonl"
    done_sessions = set()
    if scores_path.is_file():
        for line in scores_path.read_text().splitlines():
            try:
                done_sessions.add(int(json.loads(line)["session"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    for k in range(sessions):
        if k in done_sessions:
            continue
        # session 0 keeps the ORIGINAL session id/artifacts (pre-protocol runs count)
        sid = f"simgen-fr30b-{task}"[:90] if k == 0 else f"simgen-fr30b-{task}"[:90] + f"-s{k:02d}"
        sdir = adir if k == 0 else adir / f"s{k:02d}"
        sdir.mkdir(exist_ok=True)
        t0 = time.time()
        attempt = 0
        while time.time() - t0 < session_wall_s and attempt < 40:
            attempt += 1
            cmd = ["python3.11", str(REPO / "scripts" / "cosigen_harness.py"),
                   "--env", t["env"], "--model", MODEL, "--session-id", sid,
                   "--max-steps", str(max_steps),
                   "--token-budget", "128000",  # eval protocol (user spec)
                   "--transcript", str(sdir / "transcript.json"),
                   "--video", str(sdir / "final.mp4"),
                   "--extra-prompt", EXTRA_PROMPT]
            if list(sdir.glob("turn_*.json")):
                cmd.append("--resume")
            with open(sdir / "run.log", "a") as log:
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=log)
                try:
                    rc = proc.wait(timeout=max(60.0, session_wall_s - (time.time() - t0)))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = -9
                    print(f"[{task}] s{k:02d}: wall cap hit; ending session", flush=True)
            res = _session_scores(sdir)
            if rc == 0 or res["success"] or rc == -9:
                break
            print(f"[{task}] s{k:02d}: driver rc={rc}; resuming (attempt {attempt})",
                  flush=True)
            time.sleep(30)
        res = _session_scores(sdir)
        res.update({"session": k, "wall_s": round(time.time() - t0, 1)})
        with open(scores_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[{task}] s{k:02d} DONE: {res}", flush=True)


def run(tasks: list[dict], max_steps: int, wait_min: float, sessions: int,
        session_wall_s: float) -> None:
    """Spawn one detached per-task session-loop supervisor per task."""
    ART_ROOT.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        adir = ART_ROOT / t["task"]
        adir.mkdir(parents=True, exist_ok=True)
        log = open(adir / "supervisor.log", "a")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--supervise-task", t["task"], "--sessions", str(sessions),
               "--session-wall-s", str(session_wall_s),
               "--max-steps", str(max_steps), "--wait-min", str(wait_min)]
        subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=log,
                         start_new_session=True)
        print(f"[run] {t['task']}: session-loop supervisor spawned "
              f"({sessions} sessions)")


def report(tasks: list[dict]) -> None:
    """score@k aggregation: per task, mean/std of best session scores + success rate."""
    import statistics
    print(f"{'task':<70} n  succ  mean   std    max")
    for t in tasks:
        sp = ART_ROOT / t["task"] / "scores.jsonl"
        rows = []
        if sp.is_file():
            for line in sp.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not rows:
            print(f"{t['task']:<70} 0")
            continue
        best = [r["best_score"] for r in rows]
        succ = sum(1 for r in rows if r["success"])
        std = statistics.pstdev(best) if len(best) > 1 else 0.0
        print(f"{t['task']:<70} {len(best):<2} {succ:<5} "
              f"{statistics.mean(best):.3f}  {std:.3f}  {max(best):.3f}")


def watch_scores(tasks: list[dict], interval_s: float, once: bool = False) -> None:
    """Periodic per-session score logger (user request 2026-07-24): every `interval_s`,
    write a human-readable snapshot listing EVERY finished session's scores for EVERY
    task to <ART_ROOT>/scores_snapshot.txt, and append the same snapshot as one JSON
    line to <ART_ROOT>/scores_history.jsonl (timestamped, for trend analysis)."""
    snap_path = ART_ROOT / "scores_snapshot.txt"
    hist_path = ART_ROOT / "scores_history.jsonl"
    while True:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"seed-30b franka eval — per-session scores (snapshot {stamp})",
                 f"(refreshed every {int(interval_s)}s; history in {hist_path.name})", ""]
        record: dict = {"t": time.time(), "tasks": {}}
        total_sessions, total_succ = 0, 0
        for t in tasks:
            sp = ART_ROOT / t["task"] / "scores.jsonl"
            rows = []
            if sp.is_file():
                for line in sp.read_text().splitlines():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            rows.sort(key=lambda r: r.get("session", 0))
            record["tasks"][t["task"]] = rows
            total_sessions += len(rows)
            total_succ += sum(1 for r in rows if r.get("success"))
            lines.append(f"{t['task']}  ({len(rows)} session(s) done)")
            if not rows:
                lines.append("    (no finished sessions yet)")
            for r in rows:
                lines.append(
                    f"    s{r.get('session', '?'):>02}: best={r.get('best_score', 0):.4f} "
                    f"final={r.get('final_score', 0):.4f} "
                    f"success={r.get('success')} turns={r.get('turns')} "
                    f"wall={r.get('wall_s', 0):.0f}s")
            lines.append("")
        lines.insert(2, f"TOTAL: {total_sessions} sessions finished, {total_succ} successes\n")
        snap_path.write_text("\n".join(lines))
        with open(hist_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{stamp}] snapshot -> {snap_path} ({total_sessions} sessions)", flush=True)
        if once:
            break
        time.sleep(interval_s)


def status(tasks: list[dict]) -> None:
    rows = []
    for t in tasks:
        adir = ART_ROOT / t["task"]
        turns = sorted(adir.glob("turn_*.json"))
        success = any('"success": true' in p.read_text() for p in turns[-8:]) \
            or any('"success": true' in p.read_text() for p in turns)
        url = _server_url(t["env"])
        booted = _ping(url).get("booted") if url else None
        rows.append((t["task"], t["tier"], len(turns),
                     "SUCCESS" if success else ("running" if turns else "-"),
                     "up" if booted else "no-pod"))
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'task':<{w}}  tier    turns  result   pod")
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:<6}  {r[2]:<5}  {r[3]:<7}  {r[4]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-tarball", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--watch-scores", action="store_true",
                    help="loop: write per-task per-session score snapshots to "
                         "scores_snapshot.txt + scores_history.jsonl")
    ap.add_argument("--watch-interval", type=float, default=600.0,
                    help="seconds between --watch-scores snapshots")
    ap.add_argument("--once", action="store_true",
                    help="with --watch-scores: write one snapshot and exit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--supervise-task", default=None,
                    help="INTERNAL: run the session loop for one task")
    ap.add_argument("--sessions", type=int, default=64, help="sessions per task (score@k)")
    ap.add_argument("--session-wall-s", type=float, default=9000,
                    help="wall-clock cap per session (default 2.5h)")
    ap.add_argument("--batch", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="explicit task dir names (overrides --batch/--offset)")
    ap.add_argument("--num-envs", type=int, default=16,
                    help="replica envs per pod (the optimize tool's search batch)")
    ap.add_argument("--pool", default="l20")
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--wait-min", type=float, default=90,
                    help="max minutes to wait for pods to boot in --run")
    args = ap.parse_args()

    tasks = load_tasks()
    print(f"[tasks] {len(tasks)} accepted tasks in ledger")
    if args.tasks:
        by_name = {t["task"]: t for t in tasks}
        missing = [n for n in args.tasks if n not in by_name]
        if missing:
            sys.exit(f"unknown/unaccepted tasks: {missing}")
        chosen = [by_name[n] for n in args.tasks]
    else:
        chosen = sel(tasks, args.batch, args.offset)

    if args.supervise_task:
        by_name = {t["task"]: t for t in tasks}
        supervise_task(by_name[args.supervise_task], args.max_steps, args.sessions,
                       args.session_wall_s, args.wait_min)
        return
    if args.list:
        for t in chosen:
            print(f"  {t['task']:<70} {t['tier']:<6} -> {t['env']}")
        return
    if args.push_tarball:
        push_tarball()
    if args.launch:
        launch(chosen, args.num_envs, args.pool)
    if args.run:
        run(chosen, args.max_steps, args.wait_min, args.sessions, args.session_wall_s)
    if args.status:
        status(chosen)
    if args.report:
        report(chosen)
    if args.watch_scores:
        watch_scores(chosen, args.watch_interval, once=args.once)


if __name__ == "__main__":
    main()
