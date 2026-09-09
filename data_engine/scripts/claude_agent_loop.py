#!/usr/bin/env python3
"""claude_agent_loop — the orchestrator's bring-your-own agent command, on Claude Code.

AgentRunner invokes its agent as

    <agent-cmd> --prompt-file F --workdir D --cap-min M --transcript T [--env K=V ...]

This loop satisfies that contract with the `claude` CLI pointed DIRECTLY at the
seed-code gateway (Anthropic-native Messages API, publicly reachable — no relay,
no corp network in the loop). Model ids are the platform's robo_orange family:

    opus 5.0   model_hub/robo_orange_o50      (the default)
    opus 4.8   model_hub/robo_orange_o48
    fable 5.0  model_hub/robo_orange_f50

Two container realities this loop owns:
  - claude refuses to run as root: when invoked as root it re-execs itself as the
    `agent` user (created here if missing) with HOME and the campaign env intact;
  - onboarding: a fresh HOME gets a .claude.json with onboarding marked complete,
    or the CLI would sit waiting for a keypress that never comes.

The session is hard-capped at --cap-min wall minutes (SIGTERM, then SIGKILL);
stdout/stderr stream to --transcript live, so a killed session keeps its evidence.

API failures never end a session by themselves: when the CLI gives up on the
gateway (its own retries exhausted — 504 storms, 429s, connection drops), this
loop backs off and RESUMES the same Claude session (`--resume`, full conversation
intact) until the cap. Only a non-API crash propagates its exit code — except a
DEAD route: no real model turn for MODEL_DOWN_S while the API keeps failing exits
75 (EX_TEMPFAIL) so the orchestrator can pause instead of burning its clocks (a
14 h gateway outage once consumed every stage of a 12-task wave).
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SEEDCODE_BASE = "https://seed-code.bytedance.com"
SEEDCODE_KEY = "<seed-code key>"  # paste the gateway key here
DEFAULT_MODEL = "model_hub/robo_orange_o50"
AGENT_USER = "agent"
MODEL_DOWN_S = 600.0  # API failing with no model turn this long = the route is dead
FINALIZE_MIN_LEFT_S = 30 * 60   # the orchestrator's launch floor: less than this is not worth a pass
BACKOFF_MAX_S = 300.0           # = the orchestrator's model re-probe interval
FINALIZE_PROMPT = (
    "You ended your turn with {left_min:.0f} minutes of session budget left. Background "
    "jobs you started are still running (they are killed only when the session ends). "
    "If your deliverable is complete and verified, reply exactly: DONE. Otherwise continue "
    "working: finish or re-verify what is unverified, update your notes, and stop when "
    "everything on disk is final."
)


def ensure_agent_user() -> pwd.struct_passwd:
    try:
        return pwd.getpwnam(AGENT_USER)
    except KeyError:
        subprocess.run(["useradd", "-m", "-s", "/bin/bash", AGENT_USER], check=True)
        return pwd.getpwnam(AGENT_USER)


def ensure_onboarded(home: Path) -> None:
    cfg = home / ".claude.json"
    state = json.loads(cfg.read_text()) if cfg.is_file() else {}
    if not state.get("hasCompletedOnboarding"):
        state.update({"hasCompletedOnboarding": True,
                      "bypassPermissionsModeAccepted": True})
        cfg.write_text(json.dumps(state))


# Set by the orchestrator's SIGTERM ("the stage's goal is met — close the session"): the
# running CLI is stopped exactly like a cap kill, the agent's processes are reaped and the
# loop exits 0 — a closed session is a completed one, graded from what is on disk.
CLOSED = threading.Event()
signal.signal(signal.SIGTERM, lambda signum, frame: CLOSED.set())


def run_claude(cmd: list[str], *, cwd: str, env: dict, demote, transcript: Path,
               deadline: float) -> tuple[int, bool]:
    """One CLI invocation appended to the transcript, killed at the deadline.
    The CLI leads its own process group and the WHOLE group is reaped when it
    ends: background jobs an agent leaves behind (a 4-env Isaac probe once sat
    on a pod's GPU for 4.5 h after its session was over) die with the session.
    Returns (returncode, capped)."""
    def preexec() -> None:
        os.setsid()
        if demote is not None:
            demote()

    with transcript.open("a") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out,
                                stderr=subprocess.STDOUT, preexec_fn=preexec)
        capped = False
        while proc.poll() is None:
            if time.time() > deadline or CLOSED.is_set():
                print("[agent_loop] " + ("session closed by the orchestrator (goal met)"
                                         if CLOSED.is_set() else "cap reached") + " — terminating",
                      flush=True)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                capped = True
                break
            time.sleep(5)
        rc = proc.wait()
    reap(proc.pid, demoted=demote is not None)
    return rc, capped


def reap(pgid: int, demoted: bool) -> None:
    """Kill everything the session left running: its process group, and — when the
    agent ran as the dedicated `agent` user — every process of that user. `setsid`
    launches escape the group (v2 agents used it deliberately once they learned
    `nohup` children died with the turn; leaked Isaac runs then held a pod's GPU for
    9 h and turned a whole visual stage into false "scene does not build" verdicts).
    Isaac processes are sent SIGKILL directly: on SIGTERM they hang on a carb
    "Press ABORT" assertion instead of exiting."""
    def alive() -> list[str]:
        pids = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
        if demoted:
            pids += subprocess.run(["pgrep", "-u", AGENT_USER], capture_output=True, text=True).stdout.split()
        return sorted(set(pids) - {str(os.getpid())})

    left = alive()
    if not left:
        return
    print(f"[agent_loop] {len(left)} process(es) left behind by the session — killing: "
          f"{' '.join(left)}", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in alive():
            try:
                os.kill(int(pid), sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(5)
    if alive():
        print(f"[agent_loop] WARNING: still alive after SIGKILL: {' '.join(alive())}", flush=True)


def real_turns(transcript: Path) -> int:
    """Model turns in the transcript (the CLI's synthetic API-error messages excluded)."""
    with transcript.open(errors="replace") as fh:
        return sum(1 for line in fh
                   if '"type":"assistant"' in line and '"model":"<synthetic>"' not in line)


def last_result(transcript: Path) -> dict | None:
    """The CLI's final `result` event (None when it crashed before emitting one)."""
    result = None
    with transcript.open(errors="replace") as fh:
        for line in fh:
            if line.startswith('{"type":"result"'):
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[agent_loop] unparseable result event: {exc}", flush=True)
    return result


def is_api_failure(result: dict | None) -> bool:
    if not result or not result.get("is_error"):
        return False
    return (result.get("api_error_status") is not None
            or str(result.get("result", "")).startswith("API Error"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cap-min", type=float, required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--model", default=os.environ.get("DGEN_AGENT_MODEL", DEFAULT_MODEL))
    a = ap.parse_args()

    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env.update(
        ANTHROPIC_BASE_URL=SEEDCODE_BASE,
        ANTHROPIC_API_KEY=SEEDCODE_KEY,
        ANTHROPIC_MODEL=a.model,
        # sessions must never inherit an interactive terminal's assumptions
        CI="1", TERM="dumb",
    )

    demote = None
    if os.geteuid() == 0:
        user = ensure_agent_user()
        home = Path(user.pw_dir)
        env.update(HOME=str(home), USER=AGENT_USER, LOGNAME=AGENT_USER)
        # the session's workdir must be writable by the demoted user
        subprocess.run(["chown", "-R", f"{AGENT_USER}:{AGENT_USER}", a.workdir],
                       check=False)
        subprocess.run(["chown", "-R", f"{AGENT_USER}:{AGENT_USER}",
                        str(Path(a.prompt_file).parent)], check=False)
        ensure_onboarded(home)
        os.chmod(home, 0o755)
        # The orchestrator's own Isaac runs (as root) own /tmp/isaaclab and
        # /pkg/sitecustomize.py is root-only: give the demoted agent its own
        # TMPDIR and a readable sitecustomize, or every session burns turns
        # rediscovering PermissionErrors.
        agent_tmp = Path(f"/tmp/{AGENT_USER}_tmp")
        agent_tmp.mkdir(exist_ok=True)
        os.chown(agent_tmp, user.pw_uid, user.pw_gid)
        env["TMPDIR"] = str(agent_tmp)
        for p in (Path("/pkg/sitecustomize.py"), Path("/pkg")):
            if p.exists():
                try:
                    os.chmod(p, 0o755 if p.is_dir() else 0o644)
                except OSError as exc:
                    print(f"[agent_loop] cannot chmod {p}: {exc}", flush=True)
        # Isaac's kit cache/data/logs live in the root-owned venv: every agent-launched
        # Isaac run logged datastore-lock and user.config.json errors, and agents kept
        # investigating them. Make those trees writable for the agent user once.
        isaac_py = env.get("ISAAC_PY", "")
        if isaac_py:
            site = Path(isaac_py).resolve().parent.parent / "lib"
            for kit_dir in site.glob("python*/site-packages/isaacsim/kit"):
                for sub in ("cache", "data", "logs"):
                    d = kit_dir / sub
                    if d.exists():
                        subprocess.run(["chmod", "-R", "a+rwX", str(d)], check=False,
                                       capture_output=True)

        def demote():
            os.setgid(user.pw_gid)
            os.setuid(user.pw_uid)
    else:
        ensure_onboarded(Path(env.get("HOME", str(Path.home()))))

    claude = shutil.which("claude", path=env.get("PATH"))
    if not claude:
        print("[agent_loop] no `claude` on PATH", file=sys.stderr)
        return 127
    base = [claude, "--model", a.model, "--dangerously-skip-permissions",
            "--verbose", "--output-format", "stream-json"]
    transcript = Path(a.transcript)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("")
    prompt = Path(a.prompt_file).read_text()
    deadline = time.time() + a.cap_min * 60
    session_id = None    # set -> next launch resumes this conversation
    finalized = False    # the one finalize pass has been spent
    backoff_s = 60.0   # first retry after a minute, doubling to BACKOFF_MAX_S
    turns_seen, last_progress = 0, time.time()
    while True:
        cmd = base + (["--resume", session_id] if session_id else []) + ["-p", prompt]
        rc, capped = run_claude(cmd, cwd=a.workdir, env=env, demote=demote,
                                transcript=transcript, deadline=deadline)
        print(f"[agent_loop] claude exited {rc} (transcript: {transcript})", flush=True)
        if capped:
            # A budget-expired session is a COMPLETED session, not a launch
            # failure: the orchestrator evaluates whatever the agent delivered.
            return 0
        result = last_result(transcript)
        turns = real_turns(transcript)
        if turns > turns_seen:
            turns_seen, last_progress = turns, time.time()
        if is_api_failure(result) and time.time() - last_progress > MODEL_DOWN_S:
            print(f"[agent_loop] MODEL ROUTE DOWN: API failures and no model turn for "
                  f"{MODEL_DOWN_S / 60:.0f} min — exiting 75", flush=True)
            return 75
        if not is_api_failure(result):
            left_s = deadline - time.time()
            if (rc == 0 and not finalized and result and result.get("session_id")
                    and left_s > FINALIZE_MIN_LEFT_S):
                # Agents end their turn early "waiting on monitors" (two v2 sessions
                # shipped UNTESTED cells that way). One resume tells them how much
                # budget is left and lets them finish; an agent that is done says DONE.
                finalized = True
                session_id = result["session_id"]
                prompt = FINALIZE_PROMPT.format(left_min=left_s / 60)
                print(f"[agent_loop] agent ended its turn with {left_s / 60:.0f} min "
                      "left: one finalize pass", flush=True)
                continue
            return rc
        wait = min(backoff_s, deadline - time.time())
        if wait <= 0:
            return 0
        # Always RESUME the same conversation: a fresh conversation made the agent
        # re-read the whole codebase (first code edit 2 h 20 into a 3 h session in v2).
        # A route that stays dead is caught above (MODEL_DOWN_S -> exit 75).
        if result.get("session_id"):
            session_id = result["session_id"]
        # no session id at all (the CLI died before creating one) -> the original brief
        prompt = "Continue the task." if session_id else Path(a.prompt_file).read_text()
        print(f"[agent_loop] API failure ({result.get('api_error_status')}): "
              f"resuming session {session_id} in {wait:.0f}s", flush=True)
        time.sleep(wait)
        backoff_s = min(backoff_s * 2, BACKOFF_MAX_S)


if __name__ == "__main__":
    sys.exit(main())
