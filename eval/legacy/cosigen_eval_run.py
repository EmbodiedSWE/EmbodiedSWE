#!/usr/bin/env python3
"""Run OUR harness against a task defined by the eval harness in CoSiGen/eval.

The eval harness (PR #30) runs a bare coding agent in a container: it writes Python,
builds its own env, steps it, and leaves a `solve(env)` behind. Our harness runs a swalm
agent against a persistent booted env on a render pod, with execute / assess / goto /
optimize. Different agents, same tasks — and the definition of a task, the prompt
condition, and the run layout should be shared so the two are comparable.

This is the bridge. It takes an env preset and one of their condition yamls
(eval/prompts/configs/*.yaml), assembles the task folder with THEIR library
(envbuild.prompts: instructions + task + selected rules + hints), boots or reuses a
render pod for that preset, then runs our driver with those rules and hints as its
extra prompt and its artifacts under <exp>/runs/<run>/ — the same place their runner
writes, so spend accounting and their graders can read our runs too.

    # list what can be run
    python3 scripts/cosigen_eval_run.py --list

    # our harness on the default condition
    python3 scripts/cosigen_eval_run.py --preset assembly.pc_ram.franka.osc

    # a specific condition, an existing pod, a search pod for optimize
    python3 scripts/cosigen_eval_run.py --preset assembly.bulb.franka.osc \\
        --config CoSiGen/eval/prompts/configs/nosnap.yaml \\
        --server http://[...]:9439 --opt-server http://[...]:9710
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/tiger/cap-x")
COSIGEN = REPO / "CoSiGen"
EVAL = COSIGEN / "eval"
REGISTRY = "hdfs://haruna/tmp/zeyu.shen/cosigen_render"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(REPO / "scripts"))


def presets() -> list[str]:
    """Every registered env, from robobench's registry (import-light, no Isaac)."""
    out = subprocess.run([sys.executable, str(EVAL / "scripts" / "build_env.py"), "--list"],
                         capture_output=True, text=True, cwd=COSIGEN, timeout=300)
    return [l.strip() for l in out.stdout.splitlines()
            if l.strip() and l.count(".") >= 2 and " " not in l.strip()]


def pod_url(alias: str) -> str:
    out = subprocess.run(["hdfs", "dfs", "-cat", f"{REGISTRY}/{alias}.txt"],
                         capture_output=True, text=True, timeout=180)
    return out.stdout.strip()


def pod_facilities(url: str) -> dict:
    """Which facilities the pod actually offers, read from the prompt it serves.

    Pods are launched with facilities gated off for ablation arms (CAPX_DISABLE_FEATURES),
    and the gate is invisible in the url: putting a full-arm run on a pod whose optimize
    was disabled silently produced a checkpoint-only run.
    """
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        prompt = json.loads(opener.open(url.rstrip("/") + "/ping",
                                        timeout=30).read()).get("prompt") or ""
    except Exception:  # noqa: BLE001
        return {}
    return {"optimize": "== Tuning the numbers" in prompt,
            "checkpoint": "list_checkpoints()" in prompt}


def pod_idle(url: str, timeout: float = 90) -> bool:
    """Whether the pod's main thread is free to take this run's turns.

    A pod runs turns one at a time, and killing a driver does not stop the turn it had in
    flight: that orphan keeps the thread until the runaway guard fires, and a new session
    silently waits behind it (a 20-minute stall that looked like a hang). A tiny probe turn
    is the only reliable test — /ping answers from a different thread and reports an empty
    queue either way.
    """
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url.rstrip("/") + "/run_policy",
        data=json.dumps({"code": "pass  # idle probe", "reset": False,
                         "max_steps": 1_000_000_000, "num_frames": 0,
                         "meta": {"trial_mode": True}, "session": "idle-probe"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        opener.open(req, timeout=timeout).read()
        return True
    except Exception:  # noqa: BLE001 -- busy, or unreachable
        return False


def pod_ready(url: str) -> bool:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        meta = json.loads(opener.open(url.rstrip("/") + "/ping", timeout=30).read())
        return bool(meta.get("booted"))
    except Exception:  # noqa: BLE001 -- not up yet
        return False


def ensure_pod(preset: str, alias: str, num_envs: int, wait_min: float) -> str:
    """A booted render pod for this preset, launching one if the alias is unregistered."""
    url = pod_url(alias)
    if not url:
        import launch_cosigen_render_pool as pool
        print(f"[bridge] no pod registered as {alias}; launching one", flush=True)
        pool.launch(preset, num_envs=num_envs, alias=alias)
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        url = pod_url(alias)
        if url and pod_ready(url):
            return url
        time.sleep(30)
    raise SystemExit(f"pod {alias} did not boot within {wait_min:g} min (url={url or 'none'})")


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ensure_relay(run_dir: Path, port: int | None) -> str:
    """A super_relay for this run, started here so logging is never left to memory.

    The agent's API traffic goes through it and lands in <run>/trajlog/raw_requests.jsonl;
    build_training_trajs turns that into training_trajs.jsonl in the reference format.
    """
    import urllib.request
    port = port or free_port()
    log_dir = run_dir / "trajlog"
    log_dir.mkdir(parents=True, exist_ok=True)
    relay_dir = COSIGEN / "sim_gen" / "super_relay"
    subprocess.Popen(
        ["setsid", "nohup", str(REPO / ".venv" / "bin" / "python"), "server.py",
         "--port", str(port), "--log-dir", str(log_dir)],
        cwd=relay_dir, stdout=open(run_dir / "relay.log", "a"),
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(30):
        time.sleep(2)
        try:
            opener.open(url + "/health", timeout=5).read()
            print(f"[bridge] trajectory relay on {url} -> {log_dir}", flush=True)
            return url
        except Exception:  # noqa: BLE001 -- still starting
            continue
    raise SystemExit(f"the trajectory relay did not come up on {url}; see {run_dir}/relay.log")


def start_sidecars(session_id: str, run_dir: Path) -> None:
    """The live tree view and the progress video, next to the run.

    A run with no footage rendered and no tree view leaves nothing watchable behind, so
    these come up with the driver rather than being remembered separately.
    """
    subprocess.Popen(
        ["setsid", "nohup", "python3.11", str(REPO / "scripts" / "cosigen_tree_watch.py"),
         "--session", session_id, "--out", str(run_dir / "tree_live.txt"),
         "--interval", "60"],
        cwd=REPO, stdout=open(run_dir / "tree_watch.log", "a"),
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    loop = (f"while true; do python3.11 {REPO}/scripts/cosigen_session_video.py "
            f"--session {session_id} --out {run_dir}/best_progress_live.mp4 "
            f">> {run_dir}/video_render.log 2>&1; sleep 300; done")
    subprocess.Popen(["setsid", "nohup", "bash", "-c", loop], cwd=REPO,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    # Keep the exported trajectory current, so it is there whenever someone looks
    # instead of only after a manual export.
    raw = run_dir / "trajlog" / "raw_requests.jsonl"
    if raw.parent.is_dir():
        export = (f"while true; do sleep 300; [ -s {raw} ] && "
                  f"{REPO}/.venv/bin/python {COSIGEN}/sim_gen/super_relay/"
                  f"build_training_trajs.py --raw-log {raw} --output "
                  f"{run_dir}/trajlog/training_trajs.jsonl "
                  f">> {run_dir}/trajlog/export.log 2>&1; done")
        subprocess.Popen(["setsid", "nohup", "bash", "-c", export], cwd=REPO,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[bridge] tree watcher, video loop and trajectory export writing into "
          f"{run_dir}", flush=True)


def condition(config: Path) -> tuple[list[str], list[str], float]:
    import yaml
    cfg = yaml.safe_load(config.read_text()) or {}
    return (list(cfg.get("hints") or []), list(cfg.get("rules") or []),
            float(cfg.get("budget_min") or 240))


# Library entries that describe THEIR container rather than the world: submit/solution,
# the one-shot session, the workspace that carries across stages. None of that exists in
# our harness, so passing them on would send the agent after commands it does not have.
# Everything else is world-level (what the task allows) and carries over unchanged — as
# `no_set_states` and `frozen_controller` will once they are written; both are empty
# placeholder files in the PR today.
CONTAINER_ONLY = {"autonomous_operation", "interactive_session", "transfer_experience"}


def extra_prompt(task_dir: Path) -> tuple[str, list[str]]:
    """The world-level part of the condition, as one prompt block for our driver.

    Their container agent reads /task as files; ours receives the same content inline.
    task.md is left out: our pod builds the same description from the live scene.
    Returns the block and the names skipped, so a run records what it did not carry.
    """
    parts: list[str] = []
    skipped: list[str] = []
    for sub, lead in (("rules", "Rules in force for this experiment (binding):"),
                      ("hints", "Guidance provided for this run:")):
        d = task_dir / sub
        texts = []
        for f in sorted(d.glob("*.md")) if d.is_dir() else []:
            body = f.read_text().strip()
            if not body:
                skipped.append(f"{f.stem} (the library file is empty)")
            elif f.stem in CONTAINER_ONLY:
                skipped.append(f"{f.stem} (describes their container, not the world)")
            else:
                texts.append(body)
        if texts:
            parts.append(lead + "\n\n" + "\n\n".join(texts))
    return "\n\n".join(parts), skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list registered presets and exit")
    ap.add_argument("--preset", help="env preset, e.g. assembly.pc_ram.franka.osc")
    ap.add_argument("--config", default=str(EVAL / "prompts" / "configs" / "default.yaml"),
                    help="condition yaml from their library (default: %(default)s)")
    ap.add_argument("--exp", default=None,
                    help="experiment dir for run artifacts (default: experiments/<preset>)")
    ap.add_argument("--run", default=None, help="run name (default: ours_<timestamp>)")
    ap.add_argument("--server", default=None, help="render pod url (default: boot one)")
    ap.add_argument("--opt-server", default=None,
                    help="dedicated search pod url for the optimize tool")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--wait-min", type=float, default=25, help="pod boot wait")
    ap.add_argument("--model", default="claude-opus-48")
    ap.add_argument("--traj-relay", default=None,
                    help="an existing super_relay url; by default this launches one for the "
                         "run and logs into <run>/trajlog/")
    ap.add_argument("--relay-port", type=int, default=None,
                    help="port for the run's own relay (default: a free one)")
    ap.add_argument("--require", default="",
                    help="facilities the pod must offer, e.g. 'optimize,checkpoint' — "
                         "refuses a pod whose gating would change the arm")
    ap.add_argument("--skip-idle-check", action="store_true",
                    help="launch even if the pod is mid-turn for another session")
    ap.add_argument("--no-traj-relay", action="store_true",
                    help="run without trajectory logging (not recommended)")
    ap.add_argument("--watchdog", default="on", choices=["", "on"])
    ap.add_argument("--budget-min", type=float, default=None,
                    help="wall-clock kill budget (default: the condition's)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the task folder, print the driver command, and exit")
    args = ap.parse_args()

    if args.list:
        for p in presets():
            print(p)
        return
    if not args.preset:
        ap.error("--preset is required (see --list)")

    from envbuild import prompts  # their library, used verbatim

    hints, rules, cfg_budget = condition(Path(args.config))
    budget_min = args.budget_min if args.budget_min is not None else cfg_budget
    # Our pods keep set_states available and do not freeze the controller; declaring the
    # facts lets their check_condition refuse a condition that would misdescribe our world.
    facts = {"set_states": True, "control_mode_frozen": False}
    prompts.check_condition(hints, rules, facts)

    exp = Path(args.exp) if args.exp else COSIGEN / "experiments" / args.preset.replace(".", "_")
    run = args.run or f"ours_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = exp / "runs" / run
    task_dir = run_dir / "task"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts.render_task_dir(task_dir, scene=args.preset.split(".")[1], preset=args.preset,
                            describe_text="(the description comes from the live env: our "
                                          "harness prints it into the agent's prompt)",
                            facts=facts, hints=hints, rules=rules, budget_min=budget_min)
    print(f"[bridge] condition: rules={rules or '[]'} hints={hints or '[]'} "
          f"budget={budget_min:g}min", flush=True)

    server = args.server or ensure_pod(args.preset, f"evalrun_{run}", args.num_envs,
                                       args.wait_min)
    facilities = pod_facilities(server)
    if facilities:
        have = [k for k, v in facilities.items() if v] or ["none"]
        print(f"[bridge] pod facilities: {', '.join(have)}", flush=True)
    for need in [f.strip() for f in (args.require or "").split(",") if f.strip()]:
        if not facilities.get(need):
            raise SystemExit(
                f"this pod does not offer {need} (it was launched with that facility "
                f"gated off), so the run would silently be a different arm. Use a pod "
                f"without that gate, or drop it from --require.")
    if not args.skip_idle_check and not pod_idle(server):
        raise SystemExit(
            f"the pod at {server} is still executing someone else's turn, so this run's "
            f"turns would queue behind it. Wait for it to finish (its runaway guard is "
            f"CAPX_TURN_WALL_S, an hour by default), use another pod, or pass "
            f"--skip-idle-check.")
    session_id = f"{args.preset.replace('.', '-')}-{run}"[:96]
    relay = args.traj_relay
    if not relay and not args.no_traj_relay:
        relay = ensure_relay(run_dir, args.relay_port)
    cmd = ["timeout", "--signal=TERM", "--kill-after=120", str(int(budget_min * 60)),
           "python3.11", str(REPO / "scripts" / "cosigen_harness.py"),
           "--server", server, "--env", args.preset, "--model", args.model,
           "--session-id", session_id,
           "--trial-mode", "--transcript", str(run_dir / "transcript.json")]
    if args.opt_server:
        cmd += ["--opt-server", args.opt_server]
    if args.watchdog:
        cmd += ["--watchdog", args.watchdog]
    if relay:
        cmd += ["--traj-relay", relay]
    extra, skipped = extra_prompt(task_dir)
    if extra:
        cmd += ["--extra-prompt", extra]
    if skipped:
        print(f"[bridge] not carried into our prompt: {', '.join(skipped)}", flush=True)

    (run_dir / "run.json").write_text(json.dumps({
        "harness": "cosigen (persistent env, tool-driven agent)",
        "preset": args.preset, "server": server, "opt_server": args.opt_server,
        "condition": {"config": args.config, "rules": rules, "hints": hints,
                      "not_carried": skipped},
        "model": args.model, "budget_min": budget_min, "traj_relay": relay,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    if args.dry_run:
        import shlex
        print(f"[bridge] task folder: {task_dir}")
        print(shlex.join(cmd))
        return
    log = open(run_dir / "run.log", "w")
    print(f"[bridge] running our harness -> {run_dir}", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    print(f"[bridge] driver pid {proc.pid}; log {run_dir / 'run.log'}", flush=True)
    start_sidecars(session_id, run_dir)


if __name__ == "__main__":
    main()
