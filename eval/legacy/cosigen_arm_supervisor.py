#!/usr/bin/env python3
"""Unattended supervisor for a CoSiGen arm pair (ckpt + full), selected by --tag.

Keeps the pieces an overnight run needs alive and writes one status line per cycle:

  * the two drivers -- relaunched with --resume when they exit before the deadline,
    unless the arm already reported success (then the run is over and it stays down),
  * the two super_relay instances -- if a relay dies the arm's LLM calls fail and the
    driver dies with it, so relay health is checked first and restarted in place
    (raw_requests.jsonl is append-only, so a restart never loses logged traffic),
  * the trajectory export -- rebuilt every cycle so training_trajs.jsonl is current
    whenever someone looks at it.

Usage:
  python3 scripts/cosigen_v20_supervisor.py [--interval 120] [--hours 24]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/tiger/cap-x")
ART = REPO / "CoSiGen/cosigen_eval_artifacts"
LOGS = REPO / "eval_result/TAG_trajlogs"   # TAG filled in at startup
MON = REPO / "eval_result/TAG_monitor"     # TAG filled in at startup
VENV = "/home/tiger/cap-x/.venv/bin/python"
RELAY_DIR = REPO / "CoSiGen/sim_gen/super_relay"
REGISTRY = "hdfs://haruna/tmp/zeyu.shen/cosigen_render"

EXTRA_COMMON = (
    "You have raw simulator access (env, api, torch and isaaclab imports are live in "
    "the program namespace — see RAW SIMULATOR ACCESS in the API doc); write Isaac "
    "Lab-flavored code where it helps, alongside the toolkit. Study the solved examples "
    "first: /home/tiger/CoSiGen/eval/examples/ on this simulator host holds complete "
    "verified solvers for related but different tasks — read them from your executed "
    "code before planning. Do not teleport task objects into goal states; success is "
    "graded on physical manipulation.")

def arms_for(tag: str) -> dict:
    return {
        "ckpt": {"session": f"ikea-{tag}-ckpt", "pod": f"ikea_{tag}_ckpt",
                 "opt_pod": None, "port": 8118},
        "full": {"session": f"ikea-{tag}-full", "pod": f"ikea_{tag}_full",
                 "opt_pod": f"ikea_{tag}_full_opt", "port": 8119},
    }
TAG = "v21"
ARMS: dict = {}
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(MON / "supervisor.log", "a") as f:
        f.write(line + "\n")


def sh(cmd: str, timeout: float = 120) -> str:
    out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                         timeout=timeout)
    return out.stdout.strip()


def running(*needles: str) -> bool:
    """True when some process's command line contains every needle.

    Reads /proc rather than shelling out to pgrep: a `pgrep -f <pattern>` launched
    through a shell matches the shell's OWN command line (which contains the pattern),
    so every check came back "already running" and the supervisor restarted nothing.
    """
    mine = {os.getpid(), os.getppid()}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in mine:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:  # the process exited while we were reading it
            continue
        if all(n in cmd for n in needles):
            return True
    return False


def pod_url(name: str) -> str:
    return sh(f"hdfs dfs -cat {REGISTRY}/{name}.txt 2>/dev/null")


def relay_ok(port: int) -> bool:
    try:
        _opener.open(f"http://127.0.0.1:{port}/health", timeout=8).read()
        return True
    except Exception:  # noqa: BLE001 -- unreachable means restart it
        return False


def start_relay(arm: str, port: int) -> None:
    subprocess.Popen(
        ["setsid", "nohup", VENV, "server.py", "--port", str(port),
         "--log-dir", str(LOGS / arm)],
        cwd=RELAY_DIR, stdout=open(f"/tmp/relay_{arm}.log", "a"),
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"{arm}: relay was down on :{port} -> restarted")


def driver_running(session: str) -> bool:
    return running("cosigen_harness.py", f"--session-id {session}")


def arm_succeeded(arm: str) -> bool:
    for f in sorted((ART / f"ikea_{TAG}_{arm}").glob("turn_*.json")):
        try:
            if json.loads(f.read_text()).get("success"):
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


def start_driver(arm: str, cfg: dict) -> None:
    d = ART / f"ikea_{TAG}_{arm}"
    server = pod_url(cfg["pod"])
    if not server:
        log(f"{arm}: pod {cfg['pod']} not in the registry -- cannot start the driver")
        return
    opt = f" --opt-server '{pod_url(cfg['opt_pod'])}'" if cfg["opt_pod"] else ""
    cmd = (
        f"cd {REPO} && nohup timeout --signal=TERM --kill-after=120 36000 python3.11 "
        f"scripts/cosigen_harness.py --server '{server}'{opt} "
        f"--env assembly.ikea_table.bimanual_franka.osc --model claude-opus-48 "
        f"--session-id {cfg['session']} --trial-mode --watchdog on "
        f"--traj-relay http://127.0.0.1:{cfg['port']} --resume "
        f"--transcript {d}/transcript.json --extra-prompt {json.dumps(EXTRA_COMMON)} "
        f">> {d}/run.log 2>&1 &")
    subprocess.Popen(["bash", "-lc", cmd], start_new_session=True)
    log(f"{arm}: driver was down -> relaunched with --resume")


def sidecars(arm: str) -> None:
    """The live tree view and the progress video are separate processes; without them a
    run leaves nothing watchable behind, so they are kept up alongside the driver."""
    d = ART / f"ikea_{TAG}_{arm}"
    session = ARMS[arm]["session"]
    if not running("cosigen_tree_watch.py", f"--session {session}"):
        subprocess.Popen(
            ["setsid", "nohup", "python3.11", "scripts/cosigen_tree_watch.py",
             "--session", session, "--out", str(d / "tree_live.txt"),
             "--interval", "60"],
            cwd=REPO, stdout=open(d / "tree_watch.log", "a"),
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"{arm}: tree watcher was down -> restarted")
    if not running("cosigen_session_video.py", f"--session {session}"):
        loop = (f"while true; do python3.11 {REPO}/scripts/cosigen_session_video.py "
                f"--session {session} --out {d}/best_progress_live.mp4 "
                f">> {d}/video_render.log 2>&1; sleep 300; done")
        subprocess.Popen(["setsid", "nohup", "bash", "-c", loop], cwd=REPO,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"{arm}: video render loop was down -> restarted")


def status(arm: str) -> str:
    d = ART / f"ikea_{TAG}_{arm}"
    turns = sorted(d.glob("turn_*.json"))
    motion = errs = 0
    for f in turns:
        try:
            t = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "steps_used=0/" not in str(t.get("scene_summary", "")):
            motion += 1
        if t.get("rc") not in (0, None):
            errs += 1
    wd = d / "watchdog.jsonl"
    fired = 0
    if wd.exists():
        fired = sum(1 for l in wd.read_text().splitlines()
                    if l.strip() and json.loads(l).get("intervene"))
    tree = "n/a"
    tl = d / "tree_live.txt"
    if tl.exists():
        head = tl.read_text().split("\n")[0]
        tree = head.split("|")[1].strip() if "|" in head else head[:20]
    opt = "none"
    osf = d / "optimize_status.json"
    if osf.exists():
        try:
            o = json.loads(osf.read_text())
            opt = f"{o.get('state', '?')} gen={o.get('generations_done')}"
        except (OSError, json.JSONDecodeError):
            opt = "unreadable"
    raw = LOGS / arm / "raw_requests.jsonl"
    reqs = bad = 0
    if raw.exists():
        for line in raw.read_text().splitlines():
            if not line.strip():
                continue
            reqs += 1
            try:
                if (json.loads(line).get("status") or 200) != 200:
                    bad += 1
            except json.JSONDecodeError:
                continue
    return (f"{arm}: turns={len(turns)} motion={motion} err={errs} nudges={fired} "
            f"tree={tree} optimize={opt} llm_reqs={reqs} non200={bad}")


def export_trajectories(arm: str) -> None:
    raw = LOGS / arm / "raw_requests.jsonl"
    if not raw.exists():
        return
    subprocess.run([VENV, "build_training_trajs.py", "--raw-log", str(raw),
                    "--output", str(LOGS / arm / "training_trajs.jsonl")],
                   cwd=RELAY_DIR, capture_output=True, text=True, timeout=600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v21", help="run tag: arms are "
                    "ikea-<tag>-ckpt / ikea-<tag>-full")
    ap.add_argument("--arms", default="ckpt,full",
                    help="which arms of the pair to supervise (comma separated)")
    ap.add_argument("--interval", type=float, default=120)
    ap.add_argument("--hours", type=float, default=24)
    args = ap.parse_args()
    global TAG, ARMS, LOGS, MON
    TAG = args.tag
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    ARMS = {k: v for k, v in arms_for(TAG).items() if k in want}
    if not ARMS:
        raise SystemExit(f"no known arms in --arms {args.arms!r}")
    LOGS = REPO / f"eval_result/{TAG}_trajlogs"
    MON = REPO / f"eval_result/{TAG}_monitor"
    MON.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    log(f"supervisor up: arms={list(ARMS)} interval={args.interval}s "
        f"deadline in {args.hours}h")
    while time.time() < deadline:
        for arm, cfg in ARMS.items():
            try:
                # A PAUSED file in the arm's artifact dir means "leave this arm alone":
                # its driver, relay and side-cars stay down and its state stays on disk,
                # so deleting the file resumes the arm from its conversation snapshot.
                if (ART / f"ikea_{TAG}_{arm}" / "PAUSED").exists():
                    log(f"{arm}: PAUSED (delete the PAUSED file to resume)")
                    continue
                if not relay_ok(cfg["port"]):
                    start_relay(arm, cfg["port"])
                    time.sleep(10)
                if not driver_running(cfg["session"]):
                    if arm_succeeded(arm):
                        log(f"{arm}: reported success and exited -- leaving it down")
                    else:
                        start_driver(arm, cfg)
                sidecars(arm)
                export_trajectories(arm)
                log(status(arm))
            except Exception as exc:  # noqa: BLE001 -- one bad arm must not stop the loop
                log(f"{arm}: supervisor cycle error {exc!r}")
        time.sleep(args.interval)
    log("supervisor reached its deadline; exiting (drivers left as they are)")


if __name__ == "__main__":
    main()
