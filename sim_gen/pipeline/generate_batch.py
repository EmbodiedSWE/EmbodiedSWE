"""Campaign orchestrator: N parallel construction agents building Isaac tasks.

One worker per forge. Each worker loops: sample an unused seed -> spawn a Claude Code
agent (1 h budget) that builds sim_gen/tasks/<task>/ — scene, TELEPORT solution,
rubric written after the solution, smoke battery — iterating on its forge ->
orchestrator independently re-runs the smoke (ALL PASS) AND the solve
(SIM_GEN_SOLVE: SUCCESS) on the forge + novelty judge -> ledger + cost accounting ->
respawn with a new seed. Stops when the accepted-task count is met.

Usage:
  python sim_gen/pipeline/generate_batch.py --count 50 \
      --workers 10 [--relay-port 8119] [--agent-timeout 3600]

State/artifacts under sim_gen/artifacts/campaign/:
  ledger.jsonl   one line per finished attempt (accepted or failed) with cost
  state.json     remaining count + totals (rewritten continuously)
  <task>/        agent log + fetched video for each attempt
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent
COSIGEN_ROOT = SIM_GEN_ROOT.parent
# Campaign state dir; point SIM_GEN_CAMP_DIR at a fresh dir to start a new campaign
# (the ledger in the dir is replayed for resume: used seeds + accepted count).
CAMP = Path(os.environ.get("SIM_GEN_CAMP_DIR", SIM_GEN_ROOT / "artifacts" / "campaign"))
# Task packages land here. A campaign with a fresh ledger MUST also use a fresh tasks dir:
# attempt numbering restarts, so generated names collide with an older corpus in-place.
TASKS_DIR = Path(os.environ.get("SIM_GEN_TASKS_DIR", SIM_GEN_ROOT / "tasks"))
FORGE_REG = "hdfs://haruna/tmp/zeyu.shen/simgen_forge"
# One raw_requests.jsonl per campaign (SIM_GEN_RELAY_LOG) keeps cost scans fast and the
# campaign's trajectory set self-contained. Must match the relay's --log-dir.
RELAY_LOG = Path(os.environ.get(
    "SIM_GEN_RELAY_LOG",
    SIM_GEN_ROOT / "artifacts" / "relay_logs_campaign" / "raw_requests.jsonl"))

# Cost AWARENESS estimate at Opus-API-equivalent rates (USD per 1M tokens). Actual
# billing is the Claude OAuth SUBSCRIPTION (claude-fable-5) — no per-token invoice
# exists, so this number is an upper-bound proxy for "how much model was consumed".
PRICE = {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75}

LOCK = threading.Lock()


def _oauth_token() -> str:
    path = Path(os.environ.get("SIM_GEN_OAUTH_ENV", Path.home() / ".claude_oauth_env"))
    for line in path.read_text().splitlines():
        if "CLAUDE_CODE_OAUTH_TOKEN=" in line:
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no CLAUDE_CODE_OAUTH_TOKEN in {path}")


def _call(url: str, payload: dict | None = None, timeout: float = 1800.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if payload is None:
        return json.loads(opener.open(url, timeout=timeout).read())
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(opener.open(req, timeout=timeout).read())


def forge_urls() -> list[str]:
    out = subprocess.run(["hdfs", "dfs", "-ls", FORGE_REG], capture_output=True, text=True)
    files = [l.split()[-1] for l in out.stdout.splitlines() if l.strip().endswith(".txt")]
    urls = []
    for f in sorted(files):
        cat = subprocess.run(["hdfs", "dfs", "-cat", f], capture_output=True, text=True)
        url = cat.stdout.strip().splitlines()[-1] if cat.stdout.strip() else ""
        if url:
            urls.append(url)
    return urls


class Campaign:
    def __init__(self, count: int, relay_port: int, agent_timeout: float,
                 model_cli: str):
        self.remaining = count              # accepted tasks still wanted
        self.relay_port = relay_port
        self.agent_timeout = agent_timeout
        self.model_cli = model_cli
        self.used_seeds: set[str] = set()
        self.attempt_no = 0
        self.totals = {"accepted": 0, "failed": 0, "usd_estimate": 0.0,
                       "tokens_in": 0, "tokens_out": 0}
        CAMP.mkdir(parents=True, exist_ok=True)
        # resume support: replay the ledger
        led = CAMP / "ledger.jsonl"
        if led.exists():
            for line in led.read_text().splitlines():
                rec = json.loads(line)
                self.used_seeds.add(rec["seed"])
                self.attempt_no = max(self.attempt_no, rec.get("attempt", 0))
                if rec["accepted"]:
                    self.remaining = max(0, self.remaining - 1)
                    self.totals["accepted"] += 1
                else:
                    self.totals["failed"] += 1
                self.totals["usd_estimate"] += rec.get("usd_estimate", 0.0)

    def claim(self) -> tuple[str, int] | None:
        """Reserve (seed, attempt#) or None when the count quota is met.

        Seeds are REUSABLE: a seed may host many strategically-different variants
        (novelty is judged per variant). The used-seed set only spreads draws across
        the pool; when every seed has been drawn once, the lap resets and the pool
        cycles — a quota larger than the pool keeps producing."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from seeds import enumerate_seed_pool
        with LOCK:
            if self.remaining <= 0:
                return None
            pool = [s for s in sorted(enumerate_seed_pool()) if s not in self.used_seeds]
            if not pool:
                self.used_seeds.clear()
                pool = sorted(enumerate_seed_pool())
            import random
            # salt by campaign dir: fresh campaigns draw fresh seed sequences (an unsalted
            # draw repeats the same first seeds every campaign, and agents then rediscover
            # and port their own prior constructions of those seeds)
            seed = random.Random(f"{CAMP.name}:{self.attempt_no}").choice(pool)
            self.used_seeds.add(seed)
            self.attempt_no += 1
            return seed, self.attempt_no

    def record(self, rec: dict) -> None:
        with LOCK:
            if rec["accepted"]:
                self.remaining = max(0, self.remaining - 1)
                self.totals["accepted"] += 1
            else:
                self.totals["failed"] += 1
            self.totals["usd_estimate"] = round(self.totals["usd_estimate"]
                                                + rec.get("usd_estimate", 0.0), 2)
            self.totals["tokens_in"] += rec.get("tokens_in", 0)
            self.totals["tokens_out"] += rec.get("tokens_out", 0)
            with open(CAMP / "ledger.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (CAMP / "state.json").write_text(json.dumps(
                {"remaining": self.remaining, **self.totals,
                 "updated": datetime.datetime.now().isoformat(timespec="seconds")},
                indent=2))
            print(f"[campaign] {rec['task']} "
                  f"accepted={rec['accepted']} ${rec.get('usd_estimate', 0):.2f} | "
                  f"remaining {self.remaining} | spent ~${self.totals['usd_estimate']:.2f}",
                  flush=True)


def session_cost(session_id: str | None) -> dict:
    """Sum token usage for one CC session from the campaign relay's raw log."""
    out = {"tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0,
           "usd_estimate": 0.0, "n_requests": 0}
    if not session_id or not RELAY_LOG.exists():
        return out
    # streamed: the log grows tens of MB per session; never load it whole into memory
    with open(RELAY_LOG, errors="replace") as f:
        for line in f:
            if session_id not in line:
                continue
            try:
                rec = json.loads(line)
                u = (rec.get("response") or {}).get("usage") or {}
            except Exception:
                continue
            out["n_requests"] += 1
            out["tokens_in"] += int(u.get("input_tokens") or 0)
            out["tokens_out"] += int(u.get("output_tokens") or 0)
            out["cache_read"] += int(u.get("cache_read_input_tokens") or 0)
            cc = u.get("cache_creation") or {}
            out["cache_write"] += int(u.get("cache_creation_input_tokens") or 0) or \
                int(cc.get("ephemeral_5m_input_tokens") or 0) + int(cc.get("ephemeral_1h_input_tokens") or 0)
    out["usd_estimate"] = round(
        out["tokens_in"] / 1e6 * PRICE["input"] + out["tokens_out"] / 1e6 * PRICE["output"]
        + out["cache_read"] / 1e6 * PRICE["cache_read"]
        + out["cache_write"] / 1e6 * PRICE["cache_write"], 2)
    return out


def run_agent(task: str, prompt: str, log_path: Path, relay_port: int, timeout: float,
              model_cli: str) -> tuple[int, str | None]:
    env = dict(os.environ)
    env.update({"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{relay_port}",
                "ANTHROPIC_API_KEY": ""})
    # Two billing modes (same split novelty.py already has): with the OAuth env file the
    # subscription pays and the relay only logs; without it the relay OWNS the upstream key
    # (platform billing, e.g. super-relay) and the client credential is just a session id.
    oauth = Path(os.environ.get("SIM_GEN_OAUTH_ENV", Path.home() / ".claude_oauth_env"))
    if oauth.exists():
        env["CLAUDE_CODE_OAUTH_TOKEN"] = _oauth_token()
    else:
        env["ANTHROPIC_API_KEY"] = "relay-session"
    for v in ("ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(v, None)
    cmd = ["claude", "-p", prompt, "--model", model_cli,
           "--dangerously-skip-permissions", "--verbose", "--output-format", "stream-json"]
    with open(log_path, "wb") as log:
        try:
            rc = subprocess.call(cmd, cwd=str(COSIGEN_ROOT), env=env,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -1
    sid = None
    m = re.search(rb'"session_id"\s*:\s*"([0-9a-f-]+)"', log_path.read_bytes())
    if m:
        sid = m.group(1).decode()
    return rc, sid


def accept(task: str, forge_url: str) -> tuple[bool, str]:
    """Acceptance: orchestrator-run smoke (ALL PASS) AND solve (SIM_GEN_SOLVE: SUCCESS)
    on the forge, then the novelty judge. Returns (accepted, reason)."""
    task_dir = TASKS_DIR / task
    needed = ("TASK.md", "smoke.py", "solve.py")
    if any(not (task_dir / f).exists() for f in needed):
        return False, "missing package files (need scene.py, solve.py, smoke.py, TASK.md)"
    # re-submit exactly what's on disk, then run — don't trust the agent's last upload
    files = {str(p.relative_to(task_dir)): p.read_text()
             for p in task_dir.rglob("*") if p.is_file() and p.suffix in (".py", ".md")}
    try:
        _call(forge_url + "/submit", {"task": task, "files": files}, timeout=60)
        res = _call(forge_url + "/run",
                    {"task": task, "module": "smoke", "args": ["--headless"],
                     "timeout_s": 1500}, timeout=1700)
    except Exception as exc:  # noqa: BLE001
        return False, f"forge error: {exc!r}"
    if not (res.get("rc") == 0 and res.get("all_pass")):
        return False, f"smoke rc={res.get('rc')} all_pass={res.get('all_pass')}"
    # the teleport solution is the task's legitimacy certificate: re-run it fresh
    try:
        sol = _call(forge_url + "/run",
                    {"task": task, "module": "solve", "args": ["--headless"],
                     "timeout_s": 1500}, timeout=1700)
    except Exception as exc:  # noqa: BLE001
        return False, f"forge error on solve: {exc!r}"
    tail = sol.get("stdout_tail") or ""
    if not (sol.get("rc") == 0 and "SIM_GEN_SOLVE: SUCCESS" in tail):
        return False, f"solve rc={sol.get('rc')} (no SIM_GEN_SOLVE: SUCCESS)"
    # latched credit must never decrease along the solution trajectory
    scores = [float(m) for m in re.findall(r"SIM_GEN_SCORE\s+([0-9.]+)", tail)]
    drops = [(a, b) for a, b in zip(scores, scores[1:]) if b < a - 1e-6]
    if drops:
        return False, f"score not monotonic along solve: {drops[:3]}"
    # fetch the video for review
    try:
        vid = _call(forge_url + "/fetch?path=frames.npz", timeout=300)
        if vid.get("ok"):
            import base64
            (CAMP / task).mkdir(parents=True, exist_ok=True)
            npz = CAMP / task / "frames.npz"
            npz.write_bytes(base64.b64decode(vid["b64"]))
            _encode_mp4(npz, CAMP / task / f"{task}.mp4")
    except Exception as exc:  # noqa: BLE001
        print(f"[accept] video fetch/encode failed for {task} (non-fatal): {exc!r}",
              flush=True)
    return True, f"smoke ALL PASS; solve verified in {sol.get('seconds', '?')}s"


def _encode_mp4(npz_path: Path, out: Path, fps: int = 12) -> None:
    """frames.npz -> H.264 yuv420p mp4 (never mp4v — green screen in the IDE player)."""
    import imageio_ffmpeg
    import numpy as np
    frames = np.load(npz_path)["frames"]
    if frames.size == 0:
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.stdin.write(np.ascontiguousarray(frames[..., :3]).tobytes())
    proc.stdin.close()
    proc.wait()


def worker(idx: int, forge_url: str, camp: Campaign) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompt import build_isaac_prompt
    from seeds import get_seed

    while True:
        claimed = camp.claim()
        if claimed is None:
            print(f"[worker {idx}] quota met — exiting", flush=True)
            return
        seed_id, attempt = claimed
        task = re.sub(r"[^a-z0-9_]", "_", seed_id.split("/")[-1].lower()) + f"_i{attempt}"
        seed_path, _ = get_seed(seed_id)
        task_dir = TASKS_DIR / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (CAMP / task).mkdir(parents=True, exist_ok=True)
        prompt = build_isaac_prompt(seed_id, seed_path, task, str(task_dir), forge_url)
        log_path = CAMP / task / "agent.log"
        t0 = time.time()
        print(f"[worker {idx}] attempt {attempt}: task={task} seed={seed_id}",
              flush=True)
        rc, sid = run_agent(task, prompt, log_path, camp.relay_port,
                            camp.agent_timeout, camp.model_cli)
        ok, reason = accept(task, forge_url)
        novelty = judges = None
        if ok:
            jenv = dict(os.environ,
                        SIMGEN_NOVELTY_BASE_URL=f"http://127.0.0.1:{camp.relay_port}",
                        SIM_GEN_REPORTS_DIR=str(CAMP / "reports"))
            oauth = Path(os.environ.get("SIM_GEN_OAUTH_ENV", Path.home() / ".claude_oauth_env"))
            if oauth.exists():   # platform-billing mode needs no client token (relay owns the key)
                jenv["CLAUDE_CODE_OAUTH_TOKEN"] = _oauth_token()
            nv = subprocess.run(
                [os.environ.get("SIM_GEN_PYTHON", sys.executable),
                 str(SIM_GEN_ROOT / "pipeline" / "novelty.py"),
                 "--task", task, "--seed", seed_id, "--model", camp.model_cli],
                capture_output=True, text=True, cwd=str(COSIGEN_ROOT), env=jenv)
            (CAMP / task / "novelty.log").write_text((nv.stdout or "") + (nv.stderr or ""))
            novelty = nv.returncode == 0
            if not novelty:
                ok, reason = False, "novelty judge rejected"
        if ok:
            jd = subprocess.run(
                [os.environ.get("SIM_GEN_PYTHON", sys.executable),
                 str(SIM_GEN_ROOT / "pipeline" / "judges.py"),
                 "--task", task, "--which", "both", "--model", camp.model_cli],
                capture_output=True, text=True, cwd=str(COSIGEN_ROOT), env=jenv)
            (CAMP / task / "judges.log").write_text((jd.stdout or "") + (jd.stderr or ""))
            judges = jd.returncode == 0
            if not judges:
                tail = (jd.stdout or "").strip().splitlines()
                ok, reason = False, f"judges rejected: {tail[-1] if tail else '(no output)'}"
        cost = session_cost(sid)
        camp.record({"attempt": attempt, "task": task, "seed": seed_id,
                     "accepted": ok, "reason": reason, "novelty": novelty,
                     "judges": judges, "agent_rc": rc, "session_id": sid,
                     "minutes": round((time.time() - t0) / 60, 1), **cost,
                     "finished": datetime.datetime.now().isoformat(timespec="seconds")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=50, help="accepted tasks wanted")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--relay-port", type=int, default=8119)
    ap.add_argument("--agent-timeout", type=float, default=3600,
                    help="1 h wall clock per attempt: design + solve + rubric + checks")
    ap.add_argument("--model-cli", default="claude-fable-5",
                    help="model passed to the claude CLI (the relay decides billing: "
                         "auth passthrough or relay-owned key)")
    ap.add_argument("--forge-offset", type=int, default=0,
                    help="skip the first N live forges (lets two campaigns share a pool)")
    args = ap.parse_args()

    urls = forge_urls()
    print(f"[campaign] {len(urls)} forges registered")
    live = []
    for u in urls:
        try:
            if _call(u + "/ping", timeout=15).get("ok"):
                live.append(u)
        except Exception:
            print(f"[campaign] forge unreachable: {u}", flush=True)
    print(f"[campaign] {len(live)} forges live: {live}")
    live = live[args.forge_offset:]
    n = min(args.workers, len(live))
    if n == 0:
        raise SystemExit("no live forges (after --forge-offset)")

    camp = Campaign(args.count, args.relay_port, args.agent_timeout, args.model_cli)
    threads = [threading.Thread(target=worker, args=(i, live[i], camp), daemon=True)
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # export the construction trajectories (the campaign's data product). The vendored
    # copy is the one that ships with the repo; the sibling checkout is a local override.
    out = CAMP / "training_trajs.jsonl"
    builders = [SIM_GEN_ROOT / "super_relay" / "build_training_trajs.py",
                COSIGEN_ROOT.parent / "relevant_repos" / "super_relay"
                / "build_training_trajs.py"]
    builder = next((p for p in builders if p.is_file()), None)
    if builder is None:
        print(f"[campaign] no build_training_trajs.py found (looked in {builders}); "
              f"raw trajectories remain at {RELAY_LOG}")
    else:
        subprocess.run([os.environ.get("SIM_GEN_PYTHON", "python3"), str(builder),
                        "--raw-log", str(RELAY_LOG), "--output", str(out),
                        "--min-messages", "4"], check=False)
    print(f"[campaign] trajectories -> {out}")
    print(f"[campaign] DONE: {json.dumps(camp.totals)}")


if __name__ == "__main__":
    main()
