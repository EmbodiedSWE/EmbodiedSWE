#!/usr/bin/env python3
"""Launch ONE agent run against a built experiment stage — in a SANDBOX, not docker.

This file is run_agent.py with ONE substitution: the disposable docker container becomes a
disposable SWALM Env Manager sandbox, because our GPU machines have no CAP_SYS_ADMIN and so
cannot run dockerd. Everything else is run_agent.py's code, unchanged — the condition yaml,
the task folder, the sha256 task manifest, the submission stamping, the budget loop, run.json.
A run that differs anywhere else is a run nobody can reproduce against the docker path, which
is the whole point of keeping them identical.

Every place the substrate forces a difference is marked `SUBSTRATE:` and there are only five:
  1. `docker run` -> Env Manager create_sandbox on a GPU pool (same image role, same entry).
  2. Bind mounts do not exist: /bench and /task are UPLOADED into the sandbox, and
     /workspace + /submissions are MIRRORED back into <exp>/runs/<run>/ every poll, so the
     unchanged auto_submit/scan_submissions/transcript_lines see the same files at the same
     paths they would with a mount.
  3. No baked L1 image yet (that needs a docker host to build), so Isaac + the agent CLI are
     installed into the sandbox at startup instead of coming from the image.
  4. `docker logs` -> the entry script's stdout, pulled back as the same container.log.
  5. The sandbox cannot reach api.anthropic.com, so a run always goes through the trajectory
     relay, which holds the upstream credential instead of passing the operator's through.

    python eval/scripts/run_agent_sandbox.py experiments/<exp> \\
        [--agent claude] [--model <id>] [--budget-min 240] [--gpu 1] \\
        [--run NAME] [--dry-run]

This is the DEFAULT runner: one scene + one task — it only accepts
single-stage experiments (sequence/transfer experiments get their own future
script). The CLI selects INFRASTRUCTURE only: which experiment
(scene+robot+controller were fixed when it was built), which agent/model,
GPU, budget. The CONDITION — rules, skills, tools, features — comes exclusively
from one yaml (--config; a name in eval/configs/, default `default`), so every
condition is an authored, reviewable file, never an ad-hoc flag. One run = one disposable container
(contract: eval/docker/README.md §2). The WORLD comes from the experiment
folder (built once by build_env.py, itself under a condition); the per-run half
of the condition is applied here:
  1. picks the stage dir (default: the first under stages/)
  2. assembles <exp>/runs/<run>/task/ — instructions + task (from the stage's
     describe.md) + the selected rules, skills and granted tool docs. Everything
     except the auto-generated task text is hand-authored library content,
     included only when selected; contradicting the world refuses to launch.
  3. creates <exp>/runs/<run>/workspace (the agent's only writable dir)
  4. SUBSTRATE: creates a GPU sandbox, uploads /bench,/task read-only, runs the same
     /opt/entrypoints/agent-entry.sh
  5. waits up to the budget, then stops the entry script; collects container log
  6. writes <exp>/runs/<run>/run.json (raw record; metrics are post-hoc)

SUBSTRATE: credentials do NOT come from your shell env — the sandbox has no route to
api.anthropic.com, so the trajectory relay inside it holds the upstream credential and the
agent is handed a placeholder. This is run_agent.py's own --traj-relay path, always on.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The vendored swalm lives at the repo root. Resolve it HERE instead of trusting the
# caller's PYTHONPATH: on 2026-08-01 plain `python3` stopped finding swalm, so every
# launcher spawned by a queue crashed at claim time while the same command worked the
# day before. Appended, not inserted, so a properly installed swalm still wins.
sys.path.append("/home/tiger/cap-x")
from envbuild import prompts  # noqa: E402
from envbuild.condition import load as load_condition  # noqa: E402

# The vendored /home/tiger/cap-x/swalm was updated on 2026-07-30 23:02 without its
# `core/configs`. swalm's get_hydra_config() then finds no config path, leaves `cfg` unbound, and
# every `from swalm.core.client...` import dies with UnboundLocalError -- which stopped ALL new
# launches from 23:25 onward (runs already going were unaffected: they had imported it already).
# Point swalm at a configs tree that exists. The routing this script depends on is set explicitly
# (ARNOLD_SANDBOX_ENV_MANAGER_PSM below), not read from this config. Remove once the vendored
# tree carries its own configs again.
os.environ.setdefault(
    "SWALM_CONFIG_PATH",
    "/home/tiger/cap-x/.venv-swalm/lib/python3.11/site-packages/swalm/core/configs",
)

# SUBSTRATE: the sandbox pool pulls from hub.byted.org, and there is no rb-l1-agent image
# there yet (building one needs a docker host). Until there is, this CUDA image plus the
# startup install below stands in for it; swap this constant for the real L1 image and
# provision_sandbox() becomes a no-op.
DEFAULT_IMAGE = "hub.byted.org/arnold/pytorch2.4.1-cuda12.4-cudnn9-devel:1.0.0.91"
# SUBSTRATE: the relay owns the upstream credential (see module docstring), so the operator's
# own credential is not passed through and not required.
CRED_VARS = ()
GATEWAY_KEY = "plat_OEHXT9eJ3Y0HO4Kqx8qkPCzR3rFkzAyue73hbAlcIhk"
GATEWAY_BASE = "https://super-relay.byted.org/v1"
GATEWAY_MODEL = "model_hub/es1_orange_o48"
# --upstream openrouter: same relay, different upstream. A sandbox CAN reach openrouter.ai and
# OpenRouter serves the Anthropic Messages format, so pointing the CLI straight at it looks
# possible — and it is not: with the CLI version this harness installs, every request past the
# first came back "400 Provider returned error", while the identical prompt, tools and
# max_tokens through the relay answered 200 forty-seven times running. The relay normalizes what
# it forwards (it calls upstream non-streaming and re-emits the SSE burst itself), which is what
# the direct path lacks. It also logs the upstream body, so a future 400 is readable instead of
# being a bare CLI message. The model id travels as-is from --model, so no rewrite is needed.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = "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"
PSM_L20 = "seed.sandbox.env_manager_ded016ca96e160db.service.wlby"
CC_VERSION = "2.1.216"       # the version eval/docker pins
NODE = "v22.11.0"
VENV = "/opt/venv"           # the path Dockerfile.l0-isaaclab uses, and the task text names
RELAY_DIR = "/opt/relay"     # ours: root-owned 700, so the agent cannot read the trajectory
RELAY_VENV = "/opt/relay-venv"   # outside RELAY_DIR, so resetting a run does not rebuild it
AGENT_USER = "agent"         # Dockerfile.l1-agent's non-root user
# RETIRED 2026-07-31: `--agent cosigen` staged a second harness into the sandbox (its own driver
# venv, a persistent render server on loopback, and the eval/cosigen_*.py modules) and drove the
# agent through it. Everything it existed to provide the agent — the checkpoint tree and the
# parameter search — is now a python module installed at /task/tools by the condition, importable
# from the agent's own scripts. So there is one harness, and the tools are add-ins to it. The
# retired driver is kept, unused, under eval/legacy/.


def single_stage(exp: Path) -> Path:
    stages = sorted((exp / "stages").iterdir())
    if not stages:
        sys.exit(f"no stages under {exp}")
    if len(stages) > 1:
        sys.exit(f"run_agent runs one scene + one task; this experiment has "
                 f"{len(stages)} stages {[s.name for s in stages]} — sequence "
                 "experiments get their own runner")
    return stages[0]


def stage_record(exp: Path, stage: Path) -> dict:
    """The stage's entry in the world receipt."""
    receipt = json.loads((exp / "resolved.json").read_text())
    hits = [r for r in receipt["stages"] if r["dir"] == stage.name]
    if not hits:
        sys.exit(f"stage {stage.name} not in {exp / 'resolved.json'} — rebuild the experiment")
    record = dict(hits[0])
    record["set_states"] = receipt.get("set_states", True)
    record.setdefault("control_mode_frozen", False)
    return record


# ---------------------------------------------------------------------------------------
# SUBSTRATE layer: everything below replaces exactly what `docker run` / `docker inspect` /
# `docker stop` / `docker logs` did, and nothing else. Kept together so the run logic in
# main() stays readable as run_agent.py's.
# ---------------------------------------------------------------------------------------
import asyncio  # noqa: E402
import base64   # noqa: E402
import time     # noqa: E402


def sbx(coro):
    """Run one sandbox coroutine to completion — main() stays synchronous, like run_agent."""
    return asyncio.get_event_loop().run_until_complete(coro)


async def sh(sandbox, cmd: str, user: str = "root", timeout: float = 600,
             env: dict | None = None, quiet: bool = False) -> tuple[int, str]:
    """A command in the sandbox. `/bin/sh -c` because execute() takes argv, not a shell line."""
    # Announce BEFORE running: the Isaac steps take tens of minutes, and printing only on
    # completion makes a working install look like a hang.
    if not quiet:
        print(f"  [{user}] {cmd[:120]}", flush=True)
    out = await sandbox.execute(["/bin/sh", "-c", cmd], user=user, timeout=timeout,
                                env=env, demux=True)
    text = ((out.stdout or "") + (out.stderr or "")).rstrip()
    if not quiet and text:
        print(text[:800], flush=True)
    return out.return_code or 0, text


async def reset_like_new_container(sandbox) -> None:
    """Make a reused sandbox indistinguishable from a fresh one.

    `docker run` gives a NEW container from the same image every time. Reusing a sandbox is
    only equivalent if everything a previous run wrote is gone — otherwise the next agent
    inherits its predecessor's workspace and, worse, the CLI's own conversation history in
    /home/agent/.claude, which lets it recover the last attempt's findings without paying the
    tokens. What survives is only what the IMAGE would have carried: /opt/venv, /opt/npm, node.
    """
    # The relay must be KILLED, not asked to leave. A surviving relay keeps the port and keeps
    # its log file open; after this function deletes RELAY_DIR that file is an unlinked inode,
    # so the next run's trajectory is written where nobody can ever read it — and the next
    # relay cannot bind, while /health still answers 200 from the ghost. That is exactly how a
    # run completed with an empty trajlog.
    # '/opt/relay/[s]erver.py', NOT 'server.py --port': the relay's real command line is
    # `server.py --host ... --port ...`, so the old substring never matched and relays survived
    # every reset — nut_thread_c2's reused sandbox was found carrying SEVEN of them (2026-07-31),
    # one still squatting on port 8118, the pen_holder trajectory-eater. The pgrep verification
    # below had the same dead pattern, so it never refused anything either. The [s] bracket is
    # LOAD-BEARING: without it the pattern matches the pkill shell's own command line and kills
    # it (self-match) — the un-bracketed first fix refused every reused sandbox (pc_gpu_c2/
    # pc_ram_c6, 2026-07-31).
    await sh(sandbox, "pkill -TERM -f '[a]gent-entry'; pkill -TERM -f '[c]laude'; "
                      "pkill -TERM -f '/opt/relay/[s]erver.py'; sleep 3; "
                      "pkill -KILL -f '[c]laude'; pkill -KILL -f '/opt/relay/[s]erver.py'; "
                      "pkill -KILL -f '[a]gent-entry'; sleep 1; true", quiet=True)
    # Everything the previous agent started, not just its CLI. A fresh container has no such
    # processes, and one that was left behind was found still running 4.8 h into the NEXT run:
    # a stuck Isaac holding 3.5 GB (it had outlived its own `timeout 500`), competing for the
    # GPU, and showing another task's script name in this agent's `ps` output.
    # A PREWARMED sandbox has never run an agent, so the user may not exist yet; user-scoped
    # kill/checks are guarded on it, or pgrep's "invalid user name" error lands in the output
    # and reads as "processes survived" (this refused every fresh 2-GPU box on 2026-08-01).
    have_user = f"id -u {AGENT_USER} >/dev/null 2>&1"
    await sh(sandbox, f"{have_user} && {{ pkill -TERM -u {AGENT_USER}; sleep 3; "
                      f"pkill -KILL -u {AGENT_USER}; sleep 1; }}; true", quiet=True)
    rc, out = await sh(sandbox, "pgrep -c -f '/opt/relay/[s]erver.py' || echo 0", quiet=True)
    if int((out.strip() or "0").splitlines()[-1]) > 0:
        raise SystemExit("a previous relay is still running and would hold the trajectory port; "
                         "refusing to start a run whose trajectory could be lost")
    rc, out = await sh(sandbox, f"{have_user} && pgrep -u {AGENT_USER} -a | head -5; true",
                       quiet=True)
    if out.strip():
        # Unkillable leftovers (a process wedged in a CUDA call ignores signals) mean this
        # sandbox is NOT equivalent to a fresh container, so it must not be handed to a run.
        raise SystemExit(f"processes from the previous run survived being killed, so this "
                         f"sandbox is not container-fresh:\n{out.strip()[:400]}")
    # /tmp holds the CLI's own state (/tmp/claude-<uid>, shell snapshots) and Isaac's logs.
    # /opt/build is the world-builder's tree — the FULL robobench source, which a fresh
    # container never has; reusing a builder sandbox without removing it would hand the agent
    # every suite's scenes instead of the pruned /bench.
    await sh(sandbox, f"rm -rf /workspace /submissions /bench /task {RELAY_DIR} "
                      f"/home/{AGENT_USER} /opt/entrypoints /opt/verify_solution.py "
                      f"/opt/harness /opt/build /tmp/* /tmp/.[!.]* 2>/dev/null; true")
    # Kit writes into its install dir at first run; in docker those start empty in every new
    # container, so empty them here too. /ovcache is NOT removed: run_agent.py mounts it as a
    # persistent named volume (rb-ovcache) shared across runs, so keeping it is the faithful
    # behaviour — it is a compiled-shader cache, not run data.
    await sh(sandbox, f"KIT={VENV}/lib/python3.11/site-packages/isaacsim/kit; "
                      f"rm -rf $KIT/data $KIT/logs; mkdir -p $KIT/cache $KIT/data $KIT/logs; "
                      f"chmod -R 777 $KIT/cache $KIT/data $KIT/logs; "
                      f"mkdir -p /ovcache && chown -R {AGENT_USER}:{AGENT_USER} /ovcache",
             quiet=True)
    await sh(sandbox, f"id -u {AGENT_USER} >/dev/null 2>&1 && "
                      f"mkdir -p /home/{AGENT_USER} && "
                      f"cp -a /etc/skel/. /home/{AGENT_USER}/ 2>/dev/null; "
                      f"chown -R {AGENT_USER}:{AGENT_USER} /home/{AGENT_USER} 2>/dev/null; true",
             quiet=True)
    # prove it: nothing of a previous run may remain
    checks = {
        "/workspace": "test -e /workspace && echo present || echo gone",
        "/submissions": "test -e /submissions && echo present || echo gone",
        f"/home/{AGENT_USER}/.claude": f"test -e /home/{AGENT_USER}/.claude && "
                                       f"echo present || echo gone",
        RELAY_DIR: f"test -e {RELAY_DIR} && echo present || echo gone",
        "/opt/build (robobench source)": "test -e /opt/build && echo present || echo gone",
        "/opt/verify_solution.py": "test -e /opt/verify_solution.py && echo present || echo gone",
        "/opt/harness (the success check)": "test -e /opt/harness && echo present || echo gone",
        "/tmp/claude-* (CLI state)": "ls -d /tmp/claude-* >/dev/null 2>&1 && "
                                     "echo present || echo gone",
    }
    for path, cmd in checks.items():
        rc, out = await sh(sandbox, cmd, quiet=True)
        if "gone" not in out:
            raise SystemExit(f"reuse: {path} survived the reset; refusing to run on a "
                             f"sandbox that still holds a previous run")
    rc, out = await sh(sandbox, f"test -x {VENV}/bin/python && echo venv_kept", quiet=True)
    print(f"  reset to fresh-container state ({out.strip() or 'venv MISSING'})", flush=True)


async def reuse_sandbox(uid: str):
    """SUBSTRATE: attach to a named sandbox that already holds the image contents."""
    emc, sandbox, portal = await attach_sandbox(uid)
    await reset_like_new_container(sandbox)
    pool_set(uid, state="busy", pid=os.getpid(), last_used=time.time())
    return emc, sandbox, portal


# ---------------------------------------------------------------------------------------
# The sandbox pool. `docker run` is free because the image already holds Isaac; here the
# image does not, so a fresh sandbox costs ~70 min of installing. The service has no
# commit/snapshot call (create_sandbox takes only an image name), so the only way to stop
# paying that per run is to keep provisioned sandboxes and hand each run a RESET one.
#
# Reset, not reused-as-is: reset_like_new_container() removes everything a run wrote and
# refuses to continue if any of it survived, so the agent still starts from a machine
# equivalent to a new container off the same image.
# ---------------------------------------------------------------------------------------
POOL = Path("/home/tiger/cap-x/eval_result/sandbox_pool.json")


@contextlib.contextmanager
def pool_lock():
    """Serialise pool edits: parallel launches must never claim the same sandbox."""
    POOL.parent.mkdir(parents=True, exist_ok=True)
    lock = POOL.with_suffix(".lock")
    with lock.open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def pool_read() -> list[dict]:
    if not POOL.exists():
        return []
    try:
        return json.loads(POOL.read_text())
    except json.JSONDecodeError as exc:
        print(f"  pool file is unreadable ({exc}); starting a new one", flush=True)
        return []


def pool_write(entries: list[dict]) -> None:
    tmp = POOL.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(POOL)


def pool_set(uid: str, **fields) -> None:
    with pool_lock():
        entries = pool_read()
        for e in entries:
            if e["uid"] == uid:
                e.update(fields)
                break
        else:
            entries.append({"uid": uid, **fields})
        pool_write(entries)


def pool_drop(uid: str) -> None:
    with pool_lock():
        pool_write([e for e in pool_read() if e["uid"] != uid])


def owner_alive(pid: int | None) -> bool:
    """Whether the run that claimed an entry is still running (all runs are local)."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


async def attach_sandbox(uid: str):
    """Reconstruct a client for an existing sandbox.

    execute() posts to runtime_service_url, which only wait_for_session_ready fills in, so a
    bare reconstruction from the id fails with "/api/v1/sessions/execute".
    """
    os.environ["ARNOLD_SANDBOX_ENV_MANAGER_PSM"] = PSM_L20
    from swalm.core.client.env_manager import EnvManagerClient
    from swalm.core.client.portal import PortalClient, PortalConfig
    from seed.sandbox.terminal_sandbox.impl.async_sandbox import AsyncSession
    from swalm.core.utils.bytedance.zti import get_zti_token

    emc = EnvManagerClient()
    sandbox = AsyncSession(image="reused", base_url=emc.base_url,
                           token=get_zti_token() or "", command="true",
                           session={"uid": uid})
    await sandbox.wait_for_session_ready(timeout=300)
    ip = sandbox.ip
    host = f"[{ip}]" if ":" in ip else ip
    portal = PortalClient(PortalConfig(
        endpoint=f"http://{host}:{sandbox.port_mappings['CONTAINER_PORT_0'].host_port}",
        direct_mode=True))
    return emc, sandbox, portal


def take_one_candidate(tried: set[str], min_gpu: int = 1) -> str:
    """Reserve ONE free pool entry under the lock, or return "" when none is left.

    One at a time on purpose: marking every free entry as mine and releasing the rest later left
    a window where a second launcher starting at the same moment saw nothing free and went off
    to create its own sandbox — which then has no environment and refuses to run.

    `min_gpu`: GPU count is FIXED at sandbox creation, so a 2-GPU condition claiming a 1-GPU
    pool entry silently loses its second GPU for the whole run (2026-08-01: all eight
    parameter_search runs inherited 1 GPU this way, making the documented second-GPU search
    placement impossible). Entries recorded before the gpu field existed count as 1.
    """
    with pool_lock():
        entries = pool_read()
        for e in entries:
            # "provisioning" is prewarm working on it right now; "needs-provision" has no
            # environment. Taking either would put the acquisition back inside this run.
            if e["uid"] in tried or e.get("state") in ("needs-provision", "provisioning"):
                continue
            if int(e.get("gpu") or 1) < min_gpu:
                continue
            if e.get("state") == "idle" or not owner_alive(e.get("pid")):
                e["state"], e["pid"] = "claiming", os.getpid()
                pool_write(entries)
                return e["uid"]
    return ""


async def claim_from_pool(min_gpu: int = 1):
    """Take an idle provisioned sandbox and reset it, or return None to create a new one."""
    tried: set[str] = set()
    while True:
        uid = take_one_candidate(tried, min_gpu)
        if not uid:
            return None
        tried.add(uid)
        try:
            emc, sandbox, portal = await attach_sandbox(uid)
        except Exception as exc:  # noqa: BLE001 -- reclaimed by the service, or never came up
            print(f"  pooled sandbox {uid} is gone ({type(exc).__name__}); dropping it",
                  flush=True)
            pool_drop(uid)
            continue
        # Only a sandbox that already holds the environment is worth claiming: taking one that
        # does not would put the 40-90 min acquisition back inside the run.
        #
        # Probing and resetting are as fallible as attaching: a box can hand out a stale
        # endpoint that accepts the attach and then refuses every exec ("Cannot connect to
        # host ..."). Unguarded, that exception left the loop and killed the whole launcher
        # instead of costing one candidate — so an unusable box could stop a whole campaign
        # from starting. Treat it the same way as a box with no Isaac: not claimable, leave it
        # for prewarm, try the next one.
        try:
            usable = await isaac_ok(sandbox)
        except Exception as exc:  # noqa: BLE001
            print(f"  pooled sandbox {uid} does not answer ({type(exc).__name__}); leaving it "
                  f"for prewarm", flush=True)
            pool_set(uid, state="needs-provision", pid=None)
            continue
        if not usable:
            print(f"  pooled sandbox {uid} has no Isaac; leaving it for prewarm", flush=True)
            pool_set(uid, state="needs-provision", pid=None)
            continue
        print(f"  claimed pooled sandbox {uid}", flush=True)
        try:
            await reset_like_new_container(sandbox)
        except Exception as exc:  # noqa: BLE001
            print(f"  pooled sandbox {uid} could not be reset ({type(exc).__name__}); leaving "
                  f"it for prewarm", flush=True)
            pool_set(uid, state="needs-provision", pid=None)
            continue
        pool_set(uid, state="busy", pid=os.getpid(), last_used=time.time())
        return emc, sandbox, portal


async def return_to_pool(sandbox) -> None:
    """Reset the sandbox and leave it for the next run instead of deleting it."""
    try:
        await reset_like_new_container(sandbox)
    except SystemExit as exc:
        # A sandbox that will not come clean must not be handed to another run.
        print(f"  reset failed ({exc}); dropping this sandbox from the pool", flush=True)
        pool_drop(sandbox.id)
        return
    pool_set(sandbox.id, state="idle", pid=None, last_used=time.time())
    print(f"  sandbox {sandbox.id} reset and returned to the pool", flush=True)


async def create_sandbox(image: str, gpu: int, budget_min: float):
    """SUBSTRATE for `docker run -d`: a disposable GPU container from the sandbox service."""
    os.environ["ARNOLD_SANDBOX_ENV_MANAGER_PSM"] = PSM_L20
    from swalm.core.client.env_manager import EnvManagerClient
    from swalm.core.client.portal import PortalClient, PortalConfig
    from seed.sandbox.terminal_sandbox.types import DeviceType

    emc = EnvManagerClient()
    # The idle timeout has to outlast the gaps BETWEEN runs too, or a pooled sandbox is
    # reclaimed while waiting for the next one and the install is paid again.
    sandbox = await emc.create_sandbox(
        image_name=image, device_type=DeviceType.GPU if gpu else DeviceType.CPU,
        gpu_count=gpu or None, idle_timeout=max(int(budget_min * 60) + 3600, 7 * 86400),
        create_sandbox_timeout=1800)
    ip = sandbox.ip
    host = f"[{ip}]" if ":" in ip else ip
    portal = PortalClient(PortalConfig(
        endpoint=f"http://{host}:{sandbox.port_mappings['CONTAINER_PORT_0'].host_port}",
        direct_mode=True))
    for attempt in range(60):
        try:
            await portal.execute_shell("true", timeout=20)
            break
        except Exception:  # noqa: BLE001 -- the container exists before its portal listens
            if attempt == 59:
                raise
            await asyncio.sleep(5)
    pool_set(sandbox.id, state="busy", pid=os.getpid(), image=image, gpu=gpu,
             created=time.time(), last_used=time.time())
    return emc, sandbox, portal


async def upload_dir(portal, local: Path, remote: str) -> int:
    files = {f"{remote}/{p.relative_to(local)}": p.read_bytes()
             for p in sorted(local.rglob("*")) if p.is_file()}
    if files:
        await portal.upload_files(files)
    return len(files)


async def mount_like_docker(sandbox, portal, stage: Path, task_dir: Path) -> None:
    """SUBSTRATE for the four `-v` mounts, and for Dockerfile.l1-agent's user model.

    /bench and /task are uploaded then made root-owned read-only (the `:ro` mounts);
    /workspace and /submissions are created for the non-root agent (the writable mounts).
    They are chmod 777 only while the upload runs: Portal writes as the image's own user, so
    uploading into a root-owned directory reports success and writes nothing.
    """
    await sh(sandbox, f"id -u {AGENT_USER} >/dev/null 2>&1 || "
                      f"useradd -m -s /bin/bash {AGENT_USER}")
    await sh(sandbox, f"mkdir -p /bench /task /workspace/.agent /submissions /ovcache "
                      f"{RELAY_DIR} && chmod 777 /bench /task {RELAY_DIR} && "
                      f"chown -R {AGENT_USER}:{AGENT_USER} /workspace /submissions /ovcache")
    n_bench = await upload_dir(portal, stage / "bench", "/bench")
    n_task = await upload_dir(portal, task_dir, "/task")
    await sh(sandbox, "chown -R root:root /bench /task && chmod -R a+rX,go-w /bench /task && "
                      "chmod 755 /bench /task")
    for remote, expected in (("/bench", n_bench), ("/task", n_task)):
        rc, out = await sh(sandbox, f"find {remote} -type f | wc -l", quiet=True)
        if int(out.strip() or 0) != expected:
            raise SystemExit(f"{remote}: staged {out.strip()} of {expected} files")
        print(f"  mounted {remote}: {expected} files", flush=True)


# SUBSTRATE: the captured output of the Dockerfile.l0 install steps — i.e. what the image
# would have carried. Restoring it is not a shortcut around the recipe, it IS the recipe's
# result, captured once; every run then gets a byte-identical environment instead of whatever
# pip resolves that hour, which is what a pinned image buys you.
ENV_CACHE = Path("/home/tiger/cap-x/eval_result/image_cache/l1_env.tgz")
CHUNK = 200 * 1024 * 1024   # upload_files base64s the whole body, so big files go in pieces


async def push_file(portal, sandbox, local: Path, remote: str) -> None:
    """Chunked upload; a single 10 GB request would be ~13 GB of base64 in memory."""
    await sh(sandbox, f"rm -f {remote} {remote}.part_*", quiet=True)
    sent = 0
    with local.open("rb") as fh:
        i = 0
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            await portal.upload_files({f"{remote}.part_{i:04d}": chunk})
            sent += len(chunk)
            print(f"    uploaded {sent / 1e9:.1f} GB", flush=True)
            i += 1
    await sh(sandbox, f"cat {remote}.part_* > {remote} && rm -f {remote}.part_*", quiet=True)


async def pull_file(portal, sandbox, remote: str, local: Path) -> None:
    """Chunked download, for the same reason."""
    await sh(sandbox, f"rm -f {remote}.part_*; split -b {CHUNK} -d -a 4 {remote} "
                      f"{remote}.part_", timeout=1800, quiet=True)
    rc, out = await sh(sandbox, f"ls {remote}.part_*", quiet=True)
    parts = sorted(p for p in out.split() if ".part_" in p)
    local.parent.mkdir(parents=True, exist_ok=True)
    got = 0
    with local.open("wb") as fh:
        for part in parts:
            blob = await portal.download_files([part])
            payload = (getattr(blob, "files", None) or {}).get(part)
            if not payload:
                raise SystemExit(f"env cache: {part} came back empty")
            data = base64.b64decode(payload) if isinstance(payload, str) else payload
            fh.write(data)
            got += len(data)
            print(f"    downloaded {got / 1e9:.1f} GB", flush=True)
    await sh(sandbox, f"rm -f {remote}.part_*", quiet=True)


async def isaac_ok(sandbox) -> bool:
    eula = {"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}
    rc, out = await sh(sandbox, f"test -x {VENV}/bin/python && {VENV}/bin/python -c "
                                f"'import isaacsim, isaaclab, h5py; print(\"isaac ok\")'",
                       timeout=900, quiet=True, env=eula)
    return "isaac ok" in out


async def restore_env(sandbox, portal) -> bool:
    """Put the captured environment into a fresh sandbox instead of reinstalling it."""
    print(f"  restoring the environment from {ENV_CACHE} "
          f"({ENV_CACHE.stat().st_size / 1e9:.1f} GB)", flush=True)
    t0 = time.time()
    await push_file(portal, sandbox, ENV_CACHE, "/tmp/l1_env.tgz")
    # Remove any half-finished venv first. We only get here because isaac_ok failed, so
    # whatever is there is broken; unpacking on top of it would leave that run's stale files
    # beside the cached ones instead of reproducing the captured environment exactly.
    rc, _ = await sh(sandbox, f"rm -rf {VENV} && tar xzf /tmp/l1_env.tgz -C / && "
                              f"rm -f /tmp/l1_env.tgz", timeout=2400)
    if rc != 0 or not await isaac_ok(sandbox):
        print("  restore did not yield a working environment; installing instead", flush=True)
        return False
    print(f"  environment restored in {time.time() - t0:.0f}s", flush=True)
    return True


async def capture_env(sandbox, portal) -> None:
    """Save this sandbox's installed environment so no later run has to install it again."""
    print("  capturing the environment for reuse", flush=True)
    t0 = time.time()
    # Kit's data/logs/cache are the only things under the venv a running agent mutates, and
    # reset_like_new_container empties them anyway; excluding them keeps tar from reading files
    # that change underneath it, so the capture can run on a sandbox that is working.
    kit = "opt/venv/lib/python3.11/site-packages/isaacsim/kit"
    rc, _ = await sh(sandbox, f"tar czf /tmp/l1_env.tgz --exclude='{kit}/data/*' "
                              f"--exclude='{kit}/logs/*' --exclude='{kit}/cache/*' "
                              f"-C / opt/venv", timeout=3600)
    if rc != 0:
        print("  capture failed; later runs will install", flush=True)
        return
    # A truncated tarball that restores "successfully" would silently give every later run a
    # broken interpreter, so prove the archive reads end to end before it becomes the cache.
    rc, _ = await sh(sandbox, "tar tzf /tmp/l1_env.tgz > /dev/null", timeout=1800, quiet=True)
    if rc != 0:
        print("  the captured archive does not read back; not caching it", flush=True)
        return
    tmp = ENV_CACHE.with_suffix(".partial")
    await pull_file(portal, sandbox, "/tmp/l1_env.tgz", tmp)
    tmp.replace(ENV_CACHE)   # atomic: a half-written cache must never be restored
    print(f"  environment cached -> {ENV_CACHE} "
          f"({ENV_CACHE.stat().st_size / 1e9:.1f} GB in {time.time() - t0:.0f}s)", flush=True)


async def ensure_pinocchio(sandbox) -> None:
    """pinocchio into /opt/venv, Isaac's numpy untouched. REQUIRES the venv to exist — on a
    fresh sandbox this must run AFTER the environment is installed/restored, not before
    (2026-08-01: it sat ahead of the venv install and killed every fresh provision with
    'pinocchio did not install', which no pool sandbox ever hit because they all already
    carried the venv).

    robobench's pink_ik controller requires `import pinocchio` before Isaac's AppLauncher
    (it registers eigenpy converters). The numpy pin is the point: plain `pin` resolves to
    4.1.0 and drags numpy 1.26 -> 2.4.6 under Isaac; constraining numpy makes uv pick pin
    2.7.0, which imports with Isaac's own numpy untouched. Both tried live before this was
    written.
    """
    rc, out = await sh(sandbox, f"{VENV}/bin/python -c 'import pinocchio' 2>/dev/null && "
                                f"echo have_pinocchio || echo need_pinocchio", quiet=True)
    if "need_pinocchio" in out:
        # UV_HTTP_TIMEOUT: uv's default 30 s cuts off the big cmeel wheels (boost/assimp are
        # hundreds of MB) and retries forever — pc_ram_c4 spun 26+ min on exactly this on
        # 2026-07-31. Few, patient downloads beat many racing ones here.
        # timeout 3600: the cmeel wheels are hundreds of MB and this download blew a 1800 s
        # cap in 10 of 13 provision attempts on 2026-08-01 (the cache now carries pinocchio,
        # so this step only runs when restoring an older cache or installing from scratch)
        inst_rc, inst_out = await sh(
            sandbox, f"UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 "
                     f"uv pip install --python {VENV}/bin/python pin 'numpy==1.26.0'",
            timeout=3600)
        rc, out = await sh(sandbox, f"{VENV}/bin/python -c 'import numpy, pinocchio; "
                                    f"print(\"pinocchio\", pinocchio.__version__, "
                                    f"\"numpy\", numpy.__version__)'")
        if "pinocchio" not in out or "1.26" not in out:
            # sh() prints only the head of a command's output, and uv's download chatter
            # fills it — on 2026-08-01 five boxes died here and the log carried no cause.
            # The END of uv's output is where the actual error is; say it before dying.
            raise SystemExit(f"pinocchio did not install without disturbing Isaac's numpy "
                             f"(uv rc={inst_rc}; end of uv output: "
                             f"...{inst_out[-2000:]!r})")


async def provision_sandbox(sandbox, portal, allow_acquire: bool = False) -> None:
    """SUBSTRATE for the image: Dockerfile.l0/l1's contents, installed at startup.

    Nothing here changes the run — it only puts the same interpreter, the same pinned agent
    CLI and the same Kit scratch dirs where the image would have had them. Idempotent, so a
    reused sandbox skips it in seconds.
    """
    # One `command -v` per tool: dash's builtin reports only its FIRST argument, so the
    # combined form always looked like npm was missing and re-downloaded node on every run,
    # including reused sandboxes that already had it.
    rc, out = await sh(sandbox, "for t in node npm uv claude; do command -v $t; done",
                       quiet=True)
    have = {Path(line).name for line in out.split()}
    node_dir = f"/opt/node-{NODE}-linux-x64"
    if "npm" not in have:
        await sh(sandbox, f"curl -fsSL https://nodejs.org/dist/{NODE}/"
                          f"node-{NODE}-linux-x64.tar.xz -o /tmp/node.tar.xz && "
                          f"tar xf /tmp/node.tar.xz -C /opt && chmod -R a+rX {node_dir} && "
                          f"ln -sfn {node_dir}/bin/node /usr/local/bin/node && "
                          f"ln -sfn {node_dir}/bin/npm /usr/local/bin/npm", timeout=900)
    if "uv" not in have:
        await sh(sandbox, "curl -fsSL https://astral.sh/uv/install.sh | "
                          "env UV_INSTALL_DIR=/usr/local/bin sh", timeout=600)
    if "claude" not in have:
        await sh(sandbox, f"export PATH={node_dir}/bin:$PATH && "
                          f"npm install -g --prefix /opt/npm "
                          f"@anthropic-ai/claude-code@{CC_VERSION} && "
                          f"chmod -R a+rX /opt/npm && "
                          f"ln -sfn /opt/npm/bin/claude /usr/local/bin/claude",
                 timeout=900)
    eula = {"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}
    # Dockerfile.l0's apt layer and Vulkan ICD, which I omitted on the first pass. Skipping
    # them looked harmless because torch imports and allocates fine — but Isaac's own GPU
    # pipeline does not: the first world build died in omni.physx.tensors with "CUDA error: an
    # illegal memory access" on a 16-byte allocation. The Dockerfile says why: USD/OpenSubdiv
    # and the iray renderer dlopen libgomp.so.1 / libGLU.so.1, absent from cuda images, and
    # the graphics capability injects libGLX_nvidia without its ICD manifest.
    #
    # This runs before the isaac_ok short-circuit on purpose: isaac_ok only imports the
    # packages, so a sandbox missing these libs passes it and then dies at scene boot. apt is a
    # no-op in seconds once they are installed, so a reused or cache-restored sandbox pays
    # nothing and can never be missing them.
    await sh(sandbox, "apt-get update && apt-get install -y --no-install-recommends "
                      "libglvnd0 libgl1 libglx0 libegl1 libgles2 libvulkan1 vulkan-tools "
                      "libx11-6 libxt6 libxrandr2 libgomp1 libglu1-mesa "
                      "build-essential ca-certificates curl git jq tini", timeout=1800)
    # NVIDIA GRAPHICS USERSPACE: the sandbox service injects the driver's COMPUTE libs only
    # (libcuda/libnvidia-ml/...), no libGLX_nvidia/glcore — so Vulkan fell back to llvmpipe and
    # Kit could create no GPU device: every rendering boot on a sandbox failed (found 2026-07-31
    # when scene_view captures died; nothing before had ever rendered here). The cached tree
    # carries the KERNEL-MATCHED userspace — 535.183.06, assembled from Debian snapshot debs —
    # plus the vulkan/EGL loader jsons and ld.so.conf entry. The handwritten ICD stub that used
    # to be written here pointed at a library that did not exist and claimed Vulkan 1.4.312;
    # the tarball's json is the real one. Probe after fix: NVIDIA L20 enumerated, viewport
    # annotator returns pixels in 70 s including first boot.
    rc, out = await sh(sandbox, "test -e /usr/lib/x86_64-linux-gnu/nvidia/current/"
                                "libGLX_nvidia.so.0 && echo HAVE_GFX || echo NEED_GFX", quiet=True)
    if "NEED_GFX" in out:
        gfx = Path("/home/tiger/cap-x/eval_result/image_cache/nvidia_gfx_535.183.06.tgz")
        rc, drv = await sh(sandbox, "nvidia-smi --query-gpu=driver_version "
                                    "--format=csv,noheader | head -1", quiet=True)
        if not gfx.exists():
            print("  graphics userspace missing and no cached tree; rendering will not work",
                  flush=True)
        elif "535.183.06" not in drv:
            print(f"  driver {drv.strip()} != 535.183.06; NOT installing the cached graphics "
                  f"tree (userspace must match the kernel module) — rendering will not work",
                  flush=True)
        else:
            await sh(sandbox, "mkdir -p /tmp/gfxpush && chmod 777 /tmp/gfxpush", quiet=True)
            await push_file(portal, sandbox, gfx, "/tmp/gfxpush/gfx.tgz")
            rc, out = await sh(sandbox, "tar xzf /tmp/gfxpush/gfx.tgz -C / && rm -rf /tmp/gfxpush "
                                        "&& ldconfig 2>/dev/null; timeout 30 vulkaninfo --summary "
                                        "2>&1 | grep -m1 'deviceName.*NVIDIA' || echo NO_GPU_DEVICE")
            if "NVIDIA" not in out:
                raise SystemExit("the graphics tree installed but Vulkan still cannot see the "
                                 "GPU; rendering would silently fail — refusing the sandbox")
    rc, out = await sh(sandbox, "ldconfig -p | grep -cE 'libgomp|libGLU|libvulkan'", quiet=True)
    if int(out.strip() or 0) < 3:
        raise SystemExit("the GPU libraries Isaac dlopens are missing; scenes would fail to boot")
    if await isaac_ok(sandbox):
        await ensure_pinocchio(sandbox)
        print("  image contents already present (isaac ok)", flush=True)
        return
    # An agent run must never acquire the environment itself. Acquiring it means either a ~9 GB
    # transfer or a ~40 min download of ~17 GB from pypi.nvidia.com, and doing that inside runs
    # is what killed four of them: six sandboxes downloading at once turned a ~2200 s step into
    # one past its 5400 s limit. Provisioning belongs in prewarm_sandbox_pool.py, which does it
    # ahead of time with bounded concurrency; a run then claims a ready sandbox and starts in
    # ~90 s with nothing to wait for and nothing to contend over.
    if not allow_acquire:
        raise SystemExit(
            "this sandbox has no Isaac, and an agent run will not install one: that is a "
            "40-90 min step that contends with every other run doing it.\n"
            "  fill the pool first:  python3 scripts/prewarm_sandbox_pool.py --count N\n"
            "  then launch: runs claim a ready sandbox and start in about 90 s.")
    if ENV_CACHE.exists() and await restore_env(sandbox, portal):
        await ensure_pinocchio(sandbox)
        return
    print("  installing Dockerfile.l0 contents (Isaac Sim 5.1.0 + Isaac Lab 2.3.2)",
          flush=True)
    for cmd, timeout in (
        (f"test -x {VENV}/bin/python || uv venv --python 3.11 {VENV}", 900),
        (f"uv pip install --python {VENV}/bin/python torch==2.7.0 "
         f"--index-url https://download.pytorch.org/whl/cu128", 5400),
        (f"uv pip install --python {VENV}/bin/python 'isaacsim[all,extscache]==5.1.0' "
         f"--extra-index-url https://pypi.nvidia.com", 5400),
        (f"uv pip install --python {VENV}/bin/python setuptools wheel", 600),
        (f"uv pip install --python {VENV}/bin/python 'isaaclab[all]==2.3.2' "
         f"--extra-index-url https://pypi.nvidia.com "
         f"--no-build-isolation-package flatdict", 5400),
        (f"KIT={VENV}/lib/python3.11/site-packages/isaacsim/kit && mkdir -p $KIT/cache "
         f"$KIT/data $KIT/logs && chmod -R 777 $KIT/cache $KIT/data $KIT/logs", 300),
        (f"uv pip install --python {VENV}/bin/python h5py", 600),
    ):
        t0 = time.time()
        rc, _ = await sh(sandbox, cmd, timeout=timeout, env=eula)
        print(f"    ({time.time() - t0:.0f}s rc={rc})", flush=True)
        if rc != 0:
            raise SystemExit(f"image provisioning failed: {cmd[:80]}")
    if not await isaac_ok(sandbox):
        raise SystemExit("the sandbox has no working Isaac; the task cannot be attempted")
    await ensure_pinocchio(sandbox)
    if not ENV_CACHE.exists():
        await capture_env(sandbox, portal)


async def start_relay(sandbox, portal, port: int, upstream_base: str = GATEWAY_BASE,
                      api_key: str = GATEWAY_KEY, force_model: str | None = GATEWAY_MODEL) -> str:
    """SUBSTRATE for --traj-relay: the relay runs INSIDE, because the sandbox cannot reach us.

    Root-owned 700 with its own interpreter, so the agent (a different user) cannot read the
    trajectory. The credential goes through execute()'s env, never a command line.

    `upstream_base`/`api_key`/`force_model` default to the byted gateway. A different upstream
    (--upstream) only changes these three: everything about how the relay is started, secured
    and health-checked is the same, so there is one relay path to keep working. force_model=None
    forwards whatever model the CLI asked for, which is what an upstream whose ids the CLI
    accepts (OpenRouter) needs.
    """
    src = Path(__file__).resolve().parents[2] / "sim_gen" / "super_relay"
    await portal.upload_files({f"{RELAY_DIR}/{p.name}": p.read_bytes()
                               for p in sorted(src.glob("*.py"))})
    await sh(sandbox, f"chown -R root:root {RELAY_DIR} && chmod -R go-rwx {RELAY_DIR} && "
                      f"chmod 700 {RELAY_DIR} && mkdir -p {RELAY_DIR}/trajlog")
    # The interpreter lives OUTSIDE RELAY_DIR so a reset keeps it: it is image content, like
    # /opt/venv, whereas RELAY_DIR holds the trajectory and must be wiped between runs. Keeping
    # it inside meant rebuilding the venv and re-installing fastapi on every reused sandbox.
    relay_py = f"{RELAY_VENV}/bin/python"
    await sh(sandbox, f"test -x {relay_py} || uv venv --python 3.11 {RELAY_VENV}", timeout=600)
    await sh(sandbox, f"chown -R root:root {RELAY_VENV} && chmod 700 {RELAY_VENV}", quiet=True)
    rc, out = await sh(sandbox, f"{relay_py} -c 'import fastapi, uvicorn, httpx' 2>&1 || "
                                f"echo NEED_INSTALL", quiet=True)
    if "NEED_INSTALL" in out:
        await sh(sandbox, f"uv pip install --python {relay_py} httpx uvicorn fastapi",
                 timeout=900)
    # Pick a port nothing is listening on, rather than insisting on one.
    #
    # A relay left over from an earlier run can hold the port while being impossible to kill: a
    # process backgrounded by one execute() call can end up in a PID namespace that later calls
    # cannot see or signal (only 8 pids were visible, and no pid owned the listening socket's
    # inode), yet it still holds the port in the shared network namespace. Insisting on 8118
    # therefore fails permanently for that sandbox, and — before the checks below existed —
    # silently handed the agent the ghost, whose log file the reset had already deleted.
    port = await free_port(sandbox, port)
    # --host 127.0.0.1: the relay holds the upstream credential, and its default 0.0.0.0 bind
    # makes it an open proxy to our gateway for anything that can reach the sandbox. The agent
    # talks to it over loopback, so nothing needs the wider bind.
    relay_env = {"SUPER_RELAY_API_KEY": api_key}
    if force_model:
        relay_env["SUPER_RELAY_FORCE_MODEL"] = force_model
    await sh(sandbox, f"cd {RELAY_DIR} && nohup {relay_py} {RELAY_DIR}/server.py "
                      f"--host 127.0.0.1 --port {port} --log-dir {RELAY_DIR}/trajlog "
                      f"--upstream-base {upstream_base} > {RELAY_DIR}/relay.log 2>&1 & "
                      f"echo $! > {RELAY_DIR}/relay.pid; sleep 8",
             env=relay_env)
    # Three things, because "something answered 200" is not the same as "our relay is up":
    # the process we started is alive, it did not fail to bind, and the endpoint reports the log
    # file WE gave it.
    rc, out = await sh(sandbox,
                       f"kill -0 $(cat {RELAY_DIR}/relay.pid) 2>/dev/null && echo RELAY_ALIVE; "
                       f"grep -q 'address already in use' {RELAY_DIR}/relay.log && echo BIND_FAILED; "
                       f"curl -s --max-time 10 http://127.0.0.1:{port}/health")
    if "RELAY_ALIVE" not in out or "BIND_FAILED" in out:
        rc, tail = await sh(sandbox, f"tail -20 {RELAY_DIR}/relay.log")
        raise SystemExit("the trajectory relay did not come up; a run we cannot replay is one "
                         "that did not happen")
    if f"{RELAY_DIR}/trajlog/raw_requests.jsonl" not in out:
        raise SystemExit(f"the listener on port {port} is not our relay (it reports a different "
                         f"log file); refusing to run untraced")
    rc, _ = await sh(sandbox, f"ls {RELAY_DIR}", user=AGENT_USER, quiet=True)
    if rc == 0:
        raise SystemExit(f"{RELAY_DIR} is readable by the agent")
    return f"http://127.0.0.1:{port}"


AGENT_STATE = "agent_home.tgz"   # the CLI's own session state, kept beside the run's artifacts


async def mirror_agent_state(sandbox, portal, run_dir: Path) -> bool:
    """Copy the agent CLI's session state (/home/<agent>) out, so the run can be RESUMED.

    Not part of the per-poll mirror: this is the conversation history, which grows to the size
    of the whole session, and copying it every 60 s would cost more than it saves. It is taken
    once, at teardown — the moment before reset_like_new_container deletes it.

    Without it a finished run cannot be continued at all: the workspace survives in the mirror,
    but an agent restarted against those files starts a NEW conversation and has to re-derive
    everything it had learned. This is the difference between resuming a session and rerunning
    one.
    """
    rc, out = await sh(sandbox, f"test -d /home/{AGENT_USER} && "
                                f"tar czf /tmp/{AGENT_STATE} -C /home {AGENT_USER} && "
                                f"echo STATE_OK; true", timeout=900, quiet=True)
    if "STATE_OK" not in out:
        print("  no agent session state to preserve (the CLI never started?)", flush=True)
        return False
    blob = await portal.download_files([f"/tmp/{AGENT_STATE}"])
    payload = (getattr(blob, "files", None) or {}).get(f"/tmp/{AGENT_STATE}")
    if not payload:
        print(f"  agent session state did not come back ({getattr(blob, 'errors', None)}); "
              f"this run will not be resumable", flush=True)
        return False
    data = base64.b64decode(payload) if isinstance(payload, str) else payload
    (run_dir / AGENT_STATE).write_bytes(data)
    print(f"  agent session state preserved: {len(data) / 1e6:.1f} MB "
          f"({run_dir / AGENT_STATE})", flush=True)
    return True


async def restore_run_state(sandbox, portal, run_dir: Path) -> bool:
    """Put a previous run's workspace, submissions and session state back into a fresh box.

    Returns whether the CLI's own state came back too, which is what decides between resuming
    the conversation and merely inheriting the files.
    """
    ws, subs = run_dir / "workspace", run_dir / "submissions"
    if not ws.is_dir():
        raise SystemExit(f"{ws} does not exist — nothing to resume from")
    tgz = run_dir / ".restore.tgz"
    parts = ["workspace"] + (["submissions"] if subs.is_dir() else [])
    subprocess.run(["tar", "czf", str(tgz), "-C", str(run_dir), *parts], check=True)
    await portal.upload_files({"/tmp/restore.tgz": tgz.read_bytes()})
    tgz.unlink(missing_ok=True)
    await sh(sandbox, "rm -rf /workspace /submissions && tar xzf /tmp/restore.tgz -C / && "
                      "rm -f /tmp/restore.tgz && ls -d /workspace /submissions 2>/dev/null",
             timeout=900)
    state = run_dir / AGENT_STATE
    if not state.exists():
        print(f"  no {AGENT_STATE} beside this run: its files are restored but the conversation "
              f"is not — the agent will start fresh against its own previous work", flush=True)
        return False
    await portal.upload_files({f"/tmp/{AGENT_STATE}": state.read_bytes()})
    rc, out = await sh(sandbox, f"rm -rf /home/{AGENT_USER} && "
                                f"tar xzf /tmp/{AGENT_STATE} -C /home && "
                                f"rm -f /tmp/{AGENT_STATE} && "
                                f"chown -R {AGENT_USER}:{AGENT_USER} /home/{AGENT_USER} && "
                                f"echo STATE_RESTORED", timeout=900, quiet=True)
    if "STATE_RESTORED" not in out:
        raise SystemExit("the agent session state did not restore; refusing to resume as if it "
                         "had (the run would silently start a new conversation)")
    print("  agent session state restored — the CLI resumes its own conversation", flush=True)
    return True


async def start_entry(sandbox, portal, env: dict) -> None:
    """SUBSTRATE for the container's ENTRYPOINT: the same agent-entry.sh, run as root."""
    entry = Path(__file__).resolve().parents[1] / "docker" / "entrypoints"
    await sh(sandbox, "mkdir -p /opt/entrypoints && chmod 777 /opt/entrypoints")
    await portal.upload_files({
        "/opt/entrypoints/agent-entry.sh": (entry / "agent-entry.sh").read_bytes(),
        "/opt/entrypoints/submit": (entry / "submit").read_bytes(),
        "/opt/entrypoints/verify_solution.py": (
            Path(__file__).resolve().parent / "verify_solution.py").read_bytes(),
    })
    # The verifier goes at a REPO-SHAPED path, because its first statement is
    # `REPO = Path(__file__).resolve().parents[2]`. Staged flat at /opt/verify_solution.py that
    # raises IndexError on import, so every success check died instantly with rc=1 and the
    # keep-working-until-solved signal never worked at all. Keeping the file byte-identical and
    # giving it the depth the repo has is the fix; /opt/verify_solution.py stays as a symlink so
    # SUCCESS_CHECK's command line is unchanged (resolve() follows it to the real depth).
    rc, out = await sh(sandbox,
                       "chown root:root /opt/entrypoints/* && "
                       "chmod 755 /opt/entrypoints/agent-entry.sh && "
                       "install -m 755 /opt/entrypoints/submit /usr/local/bin/submit && "
                       "mkdir -p /opt/harness/eval/scripts && "
                       "mv /opt/entrypoints/verify_solution.py "
                       "/opt/harness/eval/scripts/verify_solution.py && "
                       "chmod 700 /opt/harness/eval/scripts/verify_solution.py && "
                       "chmod -R go-rwx /opt/harness && "
                       "ln -sfn /opt/harness/eval/scripts/verify_solution.py "
                       "/opt/verify_solution.py && "
                       "ls -l /opt/verify_solution.py /opt/harness/eval/scripts/verify_solution.py")
    if rc != 0:
        raise SystemExit("could not install the entry script / success check")
    # Prove it runs before the agent starts: an unusable success check means an agent that can
    # never be told it is done, which is what happened for a full 3.4 h run.
    rc, out = await sh(sandbox, f"{VENV}/bin/python /opt/verify_solution.py --help > /dev/null "
                                f"2>&1; echo rc=$?", quiet=True)
    if "rc=0" not in out:
        rc, detail = await sh(sandbox, f"{VENV}/bin/python /opt/verify_solution.py --help 2>&1 "
                                       f"| tail -5")
        raise SystemExit("the success check cannot even start; the agent could never be told "
                         "the task is solved")
    await sh(sandbox, "nohup /opt/entrypoints/agent-entry.sh "
                      f"> {RELAY_DIR}/container.log 2>&1 & sleep 5; "
                      "pgrep -f '[a]gent-entry' | wc -l", env=env, timeout=120)


async def keepalive_once(sandbox) -> None:
    """SUBSTRATE: a docker container runs until stopped; a sandbox is reclaimed when idle.

    Called from the poll loop rather than a background task: main() is synchronous (as
    run_agent.py's is), so there is no running event loop to hold a task on.
    """
    try:
        await sandbox.keep_alive()
    except Exception as exc:  # noqa: BLE001 -- one missed ping is not fatal
        print(f"  keepalive: {type(exc).__name__}: {exc}", flush=True)


async def free_port(sandbox, first: int, tries: int = 100) -> int:
    """A port inside the sandbox that nothing is listening on."""
    rc, out = await sh(sandbox, "ss -tln 2>/dev/null | awk '{print $4}'", quiet=True)
    taken = {line.rsplit(":", 1)[-1] for line in out.split()}
    for port in range(first, first + tries):
        if str(port) not in taken:
            if port != first:
                print(f"  port {first} is held by an unkillable leftover listener; "
                      f"using {port}", flush=True)
            return port
    raise SystemExit(f"no free port in {first}..{first + tries} inside the sandbox")


async def release(emc, sandbox) -> None:
    """SUBSTRATE for `docker rm -f`: v2 sandboxes are released through the SDK session;
    swalm's delete_session is the v1 route and 404s, silently leaving the GPU running."""
    from seed.sandbox.terminal_sandbox.impl.async_sandbox import AsyncSession
    from swalm.core.utils.bytedance.zti import get_zti_token
    try:
        await emc.delete_sandbox(sandbox)
        return
    except Exception:  # noqa: BLE001 -- fall through to the REST route
        pass
    session = AsyncSession(image="x", base_url=emc.base_url,
                           token=get_zti_token() or "", command="true")
    await session.delete_session(sandbox.id)


async def running(sandbox) -> bool:
    """SUBSTRATE for `docker inspect .State.Running`: the entry script IS the container.

    Not the CLI: the keep-going loop legitimately has no claude process between legs while the
    success check boots Isaac, and treating that as "stopped" ends the run mid-flight.
    """
    rc, out = await sh(sandbox, "pgrep -f '[a]gent-entry' | wc -l", quiet=True)
    return (out.strip() or "0") != "0"


async def stop_entry(sandbox) -> None:
    """SUBSTRATE for `docker stop -t 30`."""
    await sh(sandbox, "pkill -TERM -f '[a]gent-entry'; pkill -TERM -f '[c]laude'; sleep 30; "
                      "pkill -KILL -f '[a]gent-entry'; pkill -KILL -f '[c]laude'; true",
             timeout=120, quiet=True)


async def mirror_back(sandbox, portal, run_dir: Path) -> None:
    """SUBSTRATE for the bind mounts being two-way.

    /workspace and /submissions live in the sandbox, so they are copied out to
    <exp>/runs/<run>/ on every poll. That is what lets run_agent.py's own
    transcript_lines/auto_submit/scan_submissions run unchanged against the same paths, and
    it leaves the same on-disk layout a docker run would have left.
    """
    rc, _ = await sh(sandbox, "tar czf /tmp/back.tgz -C / workspace submissions 2>/dev/null; true",
                     timeout=900, quiet=True)
    blob = await portal.download_files(["/tmp/back.tgz"])
    payload = (getattr(blob, "files", None) or {}).get("/tmp/back.tgz")
    if not payload:
        # Never silent: an empty download is the agent's work not arriving, and treating it as
        # "nothing changed" is how a stalled mirror looked like a stalled relay.
        print(f"  mirror: /tmp/back.tgz came back empty ({getattr(blob, 'errors', None)})",
              flush=True)
    else:
        # base64: DownloadFileResponse leaves content encoded unless plain_text=True.
        data = base64.b64decode(payload) if isinstance(payload, str) else payload
        tgz = run_dir / ".back.tgz"
        tgz.write_bytes(data)
        # extracts workspace/ and submissions/ straight into <exp>/runs/<run>/ — the same paths
        # the docker path's mounts occupy
        subprocess.run(["tar", "xzf", str(tgz), "-C", str(run_dir)], check=False)
        tgz.unlink(missing_ok=True)
    await mirror_trajectory(sandbox, portal, run_dir)


async def relay_pulse(sandbox) -> tuple[bool, int, int, int]:
    """(relay alive, requests COMPLETED with 200, bytes recorded, relay log bytes).

    All three read from inside the sandbox, and the first two come from the relay itself rather
    than from the agent's transcript. The transcript is a bad witness: when the gateway answered
    500, the CLI wrote an error row that counts as a model reply, so "a reply arrived but nothing
    was recorded" was true and yet the relay was working perfectly — that killed a healthy run.
    One completed 200 should produce one record, so those two are what to compare.
    """
    # Labelled, and keyed on OUR pid rather than a pattern: `pgrep -c` and `grep -c` both print
    # 0 AND exit non-zero, so a positional read of three bare numbers silently shifted (it
    # reported a 59 MB trajectory as 236 bytes). A pattern would also match a leftover relay
    # from an earlier run — those survive in a PID namespace we cannot signal — so liveness has
    # to mean the process in relay.pid.
    # A missing relay.pid must not read as death — runs started before it was written would be
    # killed on their first poll — so fall back to the pattern in that case only.
    rc, out = await sh(sandbox,
                       f"echo ALIVE=$(if test -s {RELAY_DIR}/relay.pid; then "
                       f"kill -0 $(cat {RELAY_DIR}/relay.pid) 2>/dev/null && echo 1 || echo 0; "
                       f"else pgrep -c -f '[s]erver.py --port' 2>/dev/null || echo 0; fi); "
                       f"echo SERVED=$(grep -c '200 OK' {RELAY_DIR}/relay.log 2>/dev/null "
                       f"|| echo 0); "
                       f"echo RECORDED=$(stat -c %s {RELAY_DIR}/trajlog/raw_requests.jsonl "
                       f"2>/dev/null || echo 0); "
                       f"echo LOGB=$(stat -c %s {RELAY_DIR}/relay.log 2>/dev/null || echo 0)",
                       quiet=True)
    vals = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            vals[key.strip()] = val.strip()
    def num(key: str) -> int:
        raw = vals.get(key, "0").split()
        return int(raw[0]) if raw and raw[0].isdigit() else 0
    return num("ALIVE") > 0, num("SERVED"), num("RECORDED"), num("LOGB")


async def mirror_trajectory(sandbox, portal, run_dir: Path) -> int:
    """Append only the new bytes of the relay's log, and return its true size in the sandbox.

    The trajectory is the deliverable and it only grows — tens of MB within an hour. Copying all
    of it every poll (it used to ride along in the tarball) means re-sending the whole file every
    60 s for 24 h, and a download that big is exactly what starts coming back empty. Tailing from
    the last byte keeps each poll small no matter how long the run gets.
    """
    remote = f"{RELAY_DIR}/trajlog/raw_requests.jsonl"
    local = run_dir / "opt" / "relay" / "trajlog" / "raw_requests.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    rc, out = await sh(sandbox, f"stat -c %s {remote} 2>/dev/null || echo 0", quiet=True)
    try:
        remote_size = int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        remote_size = 0
    have = local.stat().st_size if local.exists() else 0
    if remote_size <= have:
        return remote_size
    await sh(sandbox, f"tail -c +{have + 1} {remote} > /tmp/traj.part", timeout=600, quiet=True)
    blob = await portal.download_files(["/tmp/traj.part"])
    payload = (getattr(blob, "files", None) or {}).get("/tmp/traj.part")
    if not payload:
        print(f"  mirror: trajectory tail came back empty; sandbox has {remote_size} bytes, "
              f"we have {have}", flush=True)
        return remote_size
    data = base64.b64decode(payload) if isinstance(payload, str) else payload
    with local.open("ab") as fh:
        fh.write(data)
    return remote_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", help="built experiment dir (from build_env.py)")
    ap.add_argument("--config", default="default",
                    help="experimental CONDITION: a name in eval/configs/ or a path to a yaml "
                         "(default: %(default)s). It declares rules / skills / tools / features "
                         "and nothing else — infra options are CLI flags, so a condition file "
                         "stays one citable description of what the agent was granted")
    ap.add_argument("--agent", default=None, choices=["claude", "codex"],
                    help="which agent CLI runs in the container. What the agent is GRANTED is "
                         "the condition's business (--config), not the harness's: tools arrive "
                         "as python modules at /task/tools, so any CLI gets the same toolset")
    ap.add_argument("--model", default=None, help="model override (MODEL env for the agent CLI)")
    ap.add_argument("--upstream", default="gateway", choices=["gateway", "openrouter"],
                    help="which upstream the in-sandbox relay forwards to: 'gateway' (default) "
                         "the byted super-relay, whose model id it rewrites; 'openrouter' "
                         "OpenRouter, which takes --model as written (e.g. qwen/qwen3.6-27b)")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="cap on the response length the agent CLI asks for "
                         "(CLAUDE_CODE_MAX_OUTPUT_TOKENS); default: the CLI's own")
    ap.add_argument("--budget-min", type=float, default=None, help="wall-clock kill budget (minutes)")
    ap.add_argument("--auto-submit-min", type=float, default=None,
                    help="also snapshot solution/ as a submission every N minutes (skipped when "
                         "unchanged) — uniform curve sampling even if the agent never submits")
    # SUBSTRATE: in the docker path this is a device index (nvidia.com/gpu=0); a sandbox is
    # allocated a COUNT instead, so the default is 1 rather than 0. Same flag, same intent
    # ("give the run a GPU"), different unit — spelled out here so nobody reads 0 as "GPU 0".
    ap.add_argument("--gpu", default=None, help="gpu_count for the sandbox (default 1)")
    ap.add_argument("--relay-port", type=int, default=8118,
                    help="SUBSTRATE: port the in-sandbox trajectory relay listens on")
    ap.add_argument("--reuse", default="",
                    help="SUBSTRATE: run in an existing sandbox that already holds the image "
                         "contents, instead of creating one. It is reset to fresh-container "
                         "state first (workspace, submissions, agent home, bench, task and "
                         "relay all removed and verified gone), so the run is equivalent to a "
                         "new container from the same image — only the ~25 min install is "
                         "skipped. Refuses to start if anything from a previous run survives.")
    ap.add_argument("--release", action="store_true",
                    help="SUBSTRATE: delete the sandbox when the run ends (run_agent.py's "
                         "`docker rm -f`). Off by default: a sandbox costs ~70 min to provision "
                         "and the service cannot snapshot one, so the default is to RESET it and "
                         "return it to the pool for the next run.")
    ap.add_argument("--fresh-sandbox", action="store_true",
                    help="SUBSTRATE: provision a new sandbox even if an idle pooled one exists")
    ap.add_argument("--resume-from", default="",
                    help="continue a previous run: its mirrored workspace, submissions and (if "
                         "preserved) the agent CLI's own session state are restored into this "
                         "run's sandbox, and the CLI picks its conversation up where it stopped "
                         "instead of starting over. Pass that run's directory "
                         "(<exp>/runs/<name>).")
    ap.add_argument("--allow-acquire", action="store_true",
                    help="SUBSTRATE: let THIS run acquire the environment if its sandbox lacks "
                         "one (a 40-90 min step that contends with other runs doing the same). "
                         "Off by default; prewarm_sandbox_pool.py does this ahead of time.")
    ap.add_argument("--keep-going", action="store_true",
                    help="when the agent stops before the budget, continue the same "
                         "conversation instead of ending the run (long budgets)")
    ap.add_argument("--traj-relay", default=None,
                    help="route the agent's API traffic through a super_relay so the run's "
                         "trajectory is logged, e.g. http://host.docker.internal:8118 "
                         "(start it with CoSiGen/sim_gen/super_relay/server.py; export with "
                         "build_training_trajs.py)")
    ap.add_argument("--run", default=None, help="run name (default: <agent>_<timestamp>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the task folder, print the docker command, and exit")
    args = ap.parse_args()

    # The condition declares WHAT the agent is granted; every infra option is a CLI flag. Keeping
    # them apart is what makes a condition file one citable description of an arm — a config that
    # also carried budgets and agent names could not be compared across runs.
    condition = load_condition(args.config)

    if not args.exp:
        ap.error("an experiment dir is required")
    agent = args.agent or "claude"
    budget_min = float(args.budget_min or 240)
    auto_submit_min = args.auto_submit_min

    exp = Path(args.exp).resolve()
    stage = single_stage(exp)
    record = stage_record(exp, stage)
    facts = {"set_states": record["set_states"],
             "control_mode_frozen": record["control_mode_frozen"]}
    # The world was built under a condition too; a run that re-declares a different one would
    # ship prompt text the patched tree contradicts, so say so rather than discover it mid-run.
    built = (record.get("condition") or {}).get("config")
    if built and Path(built).name != condition.path.name:
        print(f"note: this world was built under '{Path(built).name}' and this run declares "
              f"'{condition.path.name}' — check the features still match")
    if not record["set_states"] and not condition.rules:
        print("note: this world restricts set_states and the run discloses nothing (no rules selected)")

    describe_file = stage / "describe.md"
    if not describe_file.exists():
        sys.exit(f"missing {describe_file} — rebuild the experiment with the current build_env.py")

    run_name = args.run or f"{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = exp / "runs" / run_name
    if run_dir.exists():
        sys.exit(f"refusing to overwrite existing {run_dir}")

    task_dir = prompts.render_task_dir(
        run_dir / "task",
        scene=record["preset"].split(".")[1],
        preset=record["preset"],
        describe_text=describe_file.read_text(),
        facts=facts,
        condition=condition,
        budget_min=budget_min,
    )
    task_files = {
        str(p.relative_to(task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(task_dir.rglob("*")) if p.is_file()
    }
    workspace = run_dir / "workspace"
    submissions = run_dir / "submissions"

    model = args.model
    # SUBSTRATE: a count, not a device index (see above). When the condition grants the
    # parameter search, an unset --gpu asks for TWO: the search evaluates its population on a
    # WIDE env (64+ candidates in parallel), and a second GPU lets that env live beside the
    # agent's working env instead of competing with it (user-approved 2026-07-31). Pooled
    # sandboxes were created with whatever they have; the bump applies to newly created ones.
    gpu = args.gpu or ("2" if "parameter_search" in condition.tools else "1")
    traj_relay = args.traj_relay
    if traj_relay:
        # SUBSTRATE: the sandbox has no route back to us, so a relay URL on the operator's
        # machine is unreachable. The relay always runs inside the sandbox instead.
        print(f"note: --traj-relay {traj_relay} ignored; the relay runs inside the sandbox "
              f"on port {args.relay_port}")
    keep_going = bool(args.keep_going)
    image = DEFAULT_IMAGE

    # SUBSTRATE: the container's environment, assembled exactly as the `-e` flags above did.
    # The agent's PATH/PYTHONPATH/EULA come from the image in the docker path, so they are set
    # here instead; SUCCESS_CHECK is agent-entry.sh's own keep-working-until-solved hook.
    node_bin = f"/opt/node-{NODE}-linux-x64/bin"
    container_env = {
        "AGENT": agent,
        # /bench is Dockerfile.l1-agent's ENV PYTHONPATH; /task/tools is where the condition's
        # granted tools were installed, so the agent's scripts import them by module name. A
        # withheld tool is not on this path because its file was never installed.
        "PYTHONPATH": "/bench:/task/tools",
        "XDG_CACHE_HOME": "/ovcache",                 # Dockerfile.l0's ENV XDG_CACHE_HOME
        "PATH": f"{VENV}/bin:/opt/npm/bin:{node_bin}:/usr/local/sbin:/usr/local/bin:"
                f"/usr/sbin:/usr/bin:/sbin:/bin",
        "TMPDIR": "/workspace/tmp",
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
        # Dockerfile.l0's ENV: without it the graphics capability is not requested and Isaac's
        # GPU pipeline fails even though torch works.
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        # `timeout` is the OUTER bound; verify_solution.py enforces --max-seconds itself and
        # prints a verdict first. Two bounds because the inner watchdog cannot fire if the
        # process wedges before it starts (Isaac's import/boot), and a success check that never
        # returns silently freezes the keep-going loop for the rest of the budget.
        "SUCCESS_CHECK": f"timeout -k 30 2100 {VENV}/bin/python /opt/verify_solution.py "
                         f"--preset {record['preset']} --solution /workspace/solution",
    }
    if model:
        container_env["MODEL"] = model
    if args.max_output_tokens:
        container_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    if keep_going:
        # the entrypoint loops instead of ending when the CLI returns
        container_env["KEEP_GOING"] = "1"

    if args.dry_run:
        print(f"task folder assembled: {task_dir}")
        print(f"sandbox: image={image} gpu_count={gpu} pool={PSM_L20}")
        print(f"env: {json.dumps(container_env, indent=2)}")
        return

    workspace.mkdir(parents=True)
    submissions.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    # SUBSTRATE: `docker run -d` -> create the sandbox, put the mounts in it, install what the
    # image would have carried, start the relay, then run the same entry script.
    if args.reuse:
        emc, sandbox, portal = sbx(reuse_sandbox(args.reuse))
        cname = sandbox.id
        print(f"sandbox {cname} reused", flush=True)
    else:
        claimed = None if args.fresh_sandbox else sbx(claim_from_pool(min_gpu=int(gpu)))
        if claimed:
            emc, sandbox, portal = claimed
            cname = sandbox.id
            print(f"sandbox {cname} taken from the pool and reset", flush=True)
        else:
            emc, sandbox, portal = sbx(create_sandbox(image, int(gpu), budget_min))
            cname = sandbox.id
            print(f"sandbox {cname} created", flush=True)
    sbx(mount_like_docker(sandbox, portal, stage, task_dir))
    sbx(provision_sandbox(sandbox, portal, allow_acquire=args.allow_acquire))
    if args.resume_from:
        # After the mounts, before the entry: the restore overwrites /workspace, which
        # mount_like_docker has just created empty.
        resumed = sbx(restore_run_state(sandbox, portal, Path(args.resume_from).resolve()))
        if resumed:
            container_env["RESUME"] = "1"   # agent-entry.sh continues rather than starts over
    # SUBSTRATE: --traj-relay is always on and always in-sandbox (no route to the API).
    if args.upstream == "openrouter":
        if not model:
            sys.exit("--upstream openrouter needs --model (an OpenRouter id, e.g. "
                     "qwen/qwen3.6-27b): it is what every request is pinned to")
        # Pinned, not passed through: --model reaches the agent CLI's own turns, but the CLI
        # spawns SUBAGENTS on its built-in default id (seen as cc_is_subagent=true requests for
        # claude-opus-4-8, refused by OpenRouter), so a run would quietly mix models — or lose
        # every delegated turn. Rewriting at the relay is what makes the whole run one model.
        relay_url = sbx(start_relay(sandbox, portal, args.relay_port,
                                    upstream_base=OPENROUTER_BASE, api_key=OPENROUTER_KEY,
                                    force_model=model))
        print(f"upstream: {OPENROUTER_BASE}, every request pinned to {model}", flush=True)
    else:
        relay_url = sbx(start_relay(sandbox, portal, args.relay_port))
    container_env["ANTHROPIC_BASE_URL"] = relay_url
    container_env["ANTHROPIC_API_KEY"] = "relay"   # substituted upstream by the relay
    sbx(start_entry(sandbox, portal, container_env))
    print(f"running: sandbox {cname}\n  watch:  tail -f {workspace}/.agent/transcript.jsonl"
          f"\n  budget: {budget_min} min")

    import shutil
    import time

    t0 = time.time()

    def transcript_lines() -> int:
        transcript = workspace / ".agent" / "transcript.jsonl"
        return sum(1 for _ in transcript.open()) if transcript.exists() else 0

    auto_state = {"last": t0, "hash": None}

    def auto_submit() -> None:
        """Snapshot solution/ on the harness's clock — one curve point every
        --auto-submit-min even if the agent never submits. Skips unchanged
        solutions; stamps submitted.json itself (with auto: true)."""
        sol = workspace / "solution"
        if not (sol / "solve.py").exists():
            return
        h = hashlib.sha256()
        for p in sorted(sol.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(sol)).encode())
                h.update(p.read_bytes())
        digest = h.hexdigest()
        if digest == auto_state["hash"]:
            return
        nums = [int(d.name) for d in submissions.iterdir() if d.is_dir() and d.name.isdigit()]
        name = f"{max(nums, default=0) + 1:02d}"
        tmp = submissions / f".tmp_{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(sol, tmp)
        (tmp / "note.txt").write_text("auto snapshot\n")
        (tmp / "submitted.json").write_text(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_s": round(time.time() - t0, 1),
            "transcript_lines": transcript_lines(),
            "auto": True,
        }, indent=2) + "\n")
        tmp.rename(submissions / name)
        auto_state["hash"] = digest
        print(f"auto-submission: {name}  (wall {round(time.time() - t0)}s)", flush=True)

    def scan_submissions() -> None:
        """Stamp new submission snapshots with wall clock + transcript position
        — the pointers that later price each one in tokens (submissions are
        atomic: `submit` mv's completed copies into place)."""
        if not submissions.is_dir():
            return
        for d in sorted(submissions.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or (d / "submitted.json").exists():
                continue
            lines = transcript_lines()
            (d / "submitted.json").write_text(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_s": round(time.time() - t0, 1),
                "transcript_lines": lines,
            }, indent=2) + "\n")
            print(f"submission: {d.name}  (wall {round(time.time() - t0)}s, "
                  f"{lines} transcript lines)", flush=True)

    traj_state = {"bytes": 0, "served": 0, "logb": 0, "stalled_polls": 0}

    def require_trajectory() -> None:
        """Stop a run whose relay is dead or has stopped recording what it serves.

        An untraced run is worthless — nothing to hand over, nothing to replay — so it must not
        burn a budget. Getting the signal right took FOUR tries, and each wrong one killed or
        would have killed a healthy run:

        - the copy on THIS machine is not evidence: a mirror that fell behind looked exactly like
          a dead relay, and a run recording fine at 5.4 MB was stopped while its relay logged 200s.
        - the agent's transcript is not evidence either: tool_progress rows grow it through a
          ten-minute Isaac call, and when the gateway answered 500 the CLI wrote an error row
          that looks like a model reply with nothing to record.
        - so both numbers come from the relay: requests it finished with 200, and bytes it wrote.
          One 200 owes one record. Flat-and-flat is an agent inside a long tool call, which is
          normal and must be left alone.
        - a record lands when its stream COMPLETES, so on a slow-gateway day "served grew,
          recorded flat" describes a healthy relay waiting out long streams (2026-08-02: a run
          was killed exactly so while its relay was alive and retrying upstream 500s). The
          relay's own log tells those apart: a wedged recorder goes silent everywhere, a slow
          one keeps logging — so the kill also requires the relay LOG to have gone flat.
        """
        alive, served, recorded, logb = sbx(relay_pulse(sandbox))
        if not alive:
            print("the trajectory relay is no longer running — stopping the run rather than "
                  "spending a budget on something nobody can replay", flush=True)
            sbx(sh(sandbox, f"tail -20 {RELAY_DIR}/relay.log; ls -la {RELAY_DIR}/trajlog"))
            sbx(stop_entry(sandbox))
            raise SystemExit("run stopped: the relay died")
        if (recorded > traj_state["bytes"] or served <= traj_state["served"]
                or logb > traj_state["logb"]):
            traj_state.update(bytes=max(recorded, traj_state["bytes"]),
                              served=max(served, traj_state["served"]),
                              logb=max(logb, traj_state["logb"]), stalled_polls=0)
            return
        traj_state["stalled_polls"] += 1
        if traj_state["stalled_polls"] < 10:
            return
        print(f"the relay finished {served - traj_state['served']} requests with 200 over "
              f"{traj_state['stalled_polls']} polls while what it recorded stayed at "
              f"{recorded} bytes and its own log stayed at {logb} bytes — stopping the run "
              f"rather than spending a budget on something nobody can replay", flush=True)
        sbx(sh(sandbox, f"tail -20 {RELAY_DIR}/relay.log; ls -la {RELAY_DIR}/trajlog"))
        sbx(stop_entry(sandbox))
        raise SystemExit("run stopped: the relay is not recording what it serves")

    status, exit_code = "completed", None
    cycle_failures = 0
    while True:
        # SUBSTRATE: the mounts are not two-way, so copy the sandbox's /workspace and
        # /submissions here first, and tell the service the sandbox is still wanted;
        # everything after this line is run_agent.py's.
        # One flaky exec must not kill a 24 h run's babysitter (2026-08-01: a single
        # transient 500 — "container process is not running" — killed pc_gpu_ram_c7's
        # launcher while the run inside was perfectly healthy, silently ending keepalives,
        # mirroring and budget enforcement). Tolerate isolated cycle failures; only a box
        # that stays unreachable for many consecutive cycles ends the run.
        try:
            sbx(keepalive_once(sandbox))
            sbx(mirror_back(sandbox, portal, run_dir))
            sbx(mirror_trajectory(sandbox, portal, run_dir))
            require_trajectory()
            scan_submissions()
            still_running = sbx(running(sandbox))
            cycle_failures = 0
        except SystemExit:
            raise                       # require_trajectory's verdicts stay fatal
        except Exception as exc:  # noqa: BLE001
            cycle_failures += 1
            print(f"poll cycle failed ({cycle_failures}/10, retrying in 60s): {exc!r}",
                  flush=True)
            if cycle_failures >= 10:
                status = "unreachable"
                print("the sandbox has been unreachable for 10 consecutive cycles — "
                      "ending the run", flush=True)
                break
            time.sleep(60)
            continue
        if not still_running:
            exit_code = 0
            break
        if auto_submit_min and time.time() - auto_state["last"] >= float(auto_submit_min) * 60:
            auto_state["last"] = time.time()
            auto_submit()
        if time.time() - t0 > budget_min * 60:
            status = "timeout"
            print(f"budget reached — stopping {cname}")
            sbx(stop_entry(sandbox))
            break
        # SUBSTRATE: 60 s, not 5 s. Each poll copies the workspace out of the sandbox over
        # HTTP; at 5 s that is continuous transfer for a 24 h run. Submission stamping stays
        # exact because `submit` writes atomically and auto_submit hashes the tree.
        time.sleep(60)
    # Final copy-out + pool handling: best-effort per step, so a box that turned flaky at
    # the very end still gets its record written and its pool entry resolved instead of
    # leaving a stranded "busy" claim.
    try:
        scan_submissions()  # catch a submission from the final seconds
        if auto_submit_min:
            auto_submit()  # the final state, cost-pinned — the curve's last point
    except Exception as exc:  # noqa: BLE001
        print(f"final submission scan failed: {exc!r}", flush=True)

    # SUBSTRATE: `docker logs` -> the entry script's stdout, and one last copy-out so the
    # final workspace/submissions/trajectory are here before the sandbox goes away.
    try:
        sbx(mirror_back(sandbox, portal, run_dir))
        sbx(mirror_trajectory(sandbox, portal, run_dir))
        # Last thing off the box before it is reset: without this the run can never be resumed.
        sbx(mirror_agent_state(sandbox, portal, run_dir))
        logs = sbx(sh(sandbox, f"cat {RELAY_DIR}/container.log 2>/dev/null", quiet=True))
        (run_dir / "container.log").write_text(logs[1])
    except Exception as exc:  # noqa: BLE001
        print(f"final mirror failed (artifacts are as of the last good cycle): {exc!r}",
              flush=True)
    try:
        if args.release:
            pool_drop(cname)
            sbx(release(emc, sandbox))
        else:
            sbx(return_to_pool(sandbox))
    except Exception as exc:  # noqa: BLE001
        print(f"pool handoff failed — dropping the entry so nothing reclaims a box in an "
              f"unknown state: {exc!r}", flush=True)
        pool_drop(cname)

    record_out = {
        "exp": str(exp), "stage": stage.name, "preset": record["preset"],
        "set_states": record["set_states"],
        "control_mode_frozen": record["control_mode_frozen"],
        "condition": condition.as_record(),
        "task_files": task_files,
        "config": args.config, "agent": agent, "model": model,
        "image": image, "gpu": gpu, "budget_min": budget_min,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status, "container_exit_code": exit_code, "argv": sys.argv,
        # SUBSTRATE: what replaced the container, so a reader knows which path produced this
        "harness": "run_agent_sandbox (env manager sandbox, no docker)",
        "sandbox_id": cname, "sandbox_pool": PSM_L20,
        "upstream": args.upstream,
        "gateway": GATEWAY_BASE if args.upstream == "gateway" else OPENROUTER_BASE,
        "gateway_model": GATEWAY_MODEL if args.upstream == "gateway" else model,
        "max_output_tokens": args.max_output_tokens,
    }
    (run_dir / "run.json").write_text(json.dumps(record_out, indent=2) + "\n")
    print(f"{status}: raw artifacts in {run_dir}  (workspace/, task/, container.log, run.json)")


if __name__ == "__main__":
    main()
