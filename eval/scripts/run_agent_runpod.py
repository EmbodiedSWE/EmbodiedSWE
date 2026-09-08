#!/usr/bin/env python3
"""Launch ONE agent run against a built experiment stage — on a RUNPOD POD, not docker.

This file is run_agent_sandbox.py with ONE substitution: the SWALM Env Manager sandbox
becomes a disposable RunPod GPU pod (RTX 4090) reached over SSH, because this campaign runs
outside the ByteDance network. Everything else is run_agent_sandbox.py's code, unchanged
where the substrate allows — the condition yaml, the task folder, the sha256 task manifest,
the submission stamping, the budget loop, the in-pod trajectory relay, the resume path,
run.json. A run that differs anywhere else is a run nobody can reproduce against the sandbox
path, which is the whole point of keeping them identical.

Every place the substrate forces a difference is marked `SUBSTRATE:`:
  1. create_sandbox -> RunPod REST create pod (same image role, same entry). No pool: a pod
     provisions from the network-volume env snapshot in ~2 min, so each run creates a fresh
     pod and (by default) terminates it at teardown — the sandbox pool existed because a
     sandbox cost 40-90 min to provision, and that reason does not exist here.
  2. Portal upload/download -> tar streams over SSH; /bench and /task are uploaded and made
     root-owned read-only, /workspace + /submissions are mirrored back every poll.
  3. No baked L1 image: the pod restores /opt/cosigen (Isaac venv + repo), node, the agent
     CLIs and the relay venv from ENV_SNAPSHOT on the attached network volume (mounted at
     /snapshot — NOT /workspace, which the harness contract owns).
  4. The relay runs in-pod as in the sandbox path, but its upstream is the real API
     (api.anthropic.com for claude, api.openai.com for codex) with a relay-owned key; the
     agent only ever sees a placeholder credential.
  5. `docker logs` -> the entry script's stdout, pulled back as the same container.log.

    python eval/scripts/run_agent_runpod.py experiments/<exp> \\
        [--agent claude|codex] [--model <id>] [--budget-min 240] \\
        [--config no_tools] [--keep-going] [--run NAME] [--resume-from DIR] [--dry-run]

Codex runs speak the relay's OpenAI chat-completions endpoint: the launcher writes
/home/agent/.codex/config.toml pinning a `relay` model provider (wire_api "chat") before the
entry starts, so the same agent-entry.sh codex adapter works untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envbuild import prompts  # noqa: E402
from envbuild.condition import load as load_condition  # noqa: E402

# Hardcoded on purpose (user directive: no env-var fallbacks for keys).
RUNPOD_KEY = "<runpod api key>"
ANTHROPIC_KEY = "<anthropic api key>"
# OpenAI direct key for the gpt-6-astra base campaign (2026-09-08, user-provided).
OPENAI_KEY = "<openai api key>"

REST = "https://rest.runpod.io/v1"
POD_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
# RB_GPU_TYPE override added 2026-08-25: EU-RO-1 ran dry of 4090s mid-IK-campaign; 2x5090
# (same DC, same snapshot volume) was the only capacity. Default unchanged.
GPU_TYPE = os.environ.get("RB_GPU_TYPE", "NVIDIA GeForce RTX 4090")
DATACENTER = os.environ.get("RB_DATACENTER", "EU-RO-1")
# Empty RB_NETWORK_VOLUME_ID means a datacenter without network volumes. In that case the
# exact same snapshot is copied from an explicit source pod before provisioning.
NETWORK_VOLUME_ID = os.environ.get("RB_NETWORK_VOLUME_ID", "sv0rdbpy30")
SNAPSHOT_MOUNT = "/snapshot"              # SUBSTRATE: NOT /workspace — the harness owns that
ENV_SNAPSHOT = f"{SNAPSHOT_MOUNT}/cosigen_env2.tar.zst"
SNAPSHOT_SOURCE_HOST = os.environ.get("RB_SNAPSHOT_SOURCE_HOST", "")
SNAPSHOT_SOURCE_PORT = int(os.environ.get("RB_SNAPSHOT_SOURCE_PORT", "22"))
SNAPSHOT_SOURCE_PATH = os.environ.get("RB_SNAPSHOT_SOURCE_PATH", ENV_SNAPSHOT)
SNAPSHOT_TRANSFER_KEY = os.environ.get("RB_SNAPSHOT_TRANSFER_KEY", "")
SNAPSHOT_SHA256 = os.environ.get("RB_SNAPSHOT_SHA256", "")

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
OPENAI_BASE = "https://api.openai.com/v1"
# seed-code gateway (--via-seed-code, added 2026-09-06): the upstream every September campaign
# run reached opus-5 / the gpt-5.6 models through (run.json `upstream` of the base_v1 and
# stage_rehearsal2 runs; key + alias recovered from the live relay of pod uoxtbcal614loi). It
# speaks native Anthropic Messages AND native OpenAI Responses, so the relay needs no bridge;
# it only needs the model id rewritten to the gateway's alias (SUPER_RELAY_FORCE_MODEL).
SEED_CODE_BASE = "https://seed-code.bytedance.com/v1"
SEED_CODE_KEY = "<seed-code key>"
SEED_CODE_ALIASES = {
    "claude-opus-5": "model_hub/robo_orange_o50",
    "gpt-5.6-sol": "model_hub/robo_g56_sol",
    "gpt-5.6-terra": "model_hub/robo_g56_terra",
}
# AIDP modelhub gateway (chat-completions only; the relay can bridge codex's Responses API
# onto it — see sim_gen/super_relay/chat_bridge.py). NOT USABLE FROM CLOUD VMs: it resolves
# to a ByteDance-internal 10.x address (measured from Modal 2026-08-15, ConnectTimeout), so
# cloud runs use the direct OpenAI API (user directive 2026-08-15) and this stays for
# corp-network use only.
AIDP_CHAT_URL = ("https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
                 "?ak=<aidp ak>")
# The same gateway through the laptop bridge (--gpt-via-tunnel): the pod's 127.0.0.1:8899 is
# a reverse SSH tunnel to the laptop's AIDP forwarder (eval/scripts/laptop_aidp_forwarder.py
# + laptop_tunnel_daemon.py), which makes the call from inside the corp network. Plain http
# on the loopback leg; the tunnel itself is SSH-encrypted, laptop->AIDP is https.
TUNNEL_AIDP_URL = ("http://127.0.0.1:8899/api/modelhub/online/v2/crawl"
                   "?ak=<aidp ak>")

CC_VERSION = "2.1.216"       # the version eval/docker pins (baked into ENV_SNAPSHOT)
VENV = "/opt/cosigen/.venv"  # the Isaac venv the snapshot restores
AGENT_USER = "agent"         # Dockerfile.l1-agent's non-root user
RELAY_DIR = "/opt/relay"     # root-owned 700, so the agent cannot read the trajectory
RELAY_VENV = "/opt/relay-venv"


# ---------------------------------------------------------------------------------------
# SUBSTRATE layer: everything below replaces exactly what the Env Manager sandbox calls
# (create/attach/execute/upload/download/delete) did, and nothing else.
# ---------------------------------------------------------------------------------------

import urllib.request  # noqa: E402
import urllib.error    # noqa: E402


def rest(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{REST}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        # explicit User-Agent: RunPod's REST edge started answering 403 Forbidden to the
        # default "Python-urllib/3.x" agent on 2026-09-06 ~22:00 UTC (curl and any other UA
        # pass); without this every pod create/list/DELETE from python fails.
        headers={"Authorization": f"Bearer {RUNPOD_KEY}", "content-type": "application/json",
                 "User-Agent": "cosigen-run-agent-runpod/1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"runpod {method} {path} -> {exc.code}: {detail}") from exc
    data = json.loads(raw) if raw.strip() else {}
    return data if isinstance(data, dict) else {"items": data}


@dataclass
class Pod:
    id: str
    ip: str
    port: int
    ssh_key: str


def sh(pod: Pod, cmd: str, user: str = "root", timeout: float = 600,
       env: dict | None = None, quiet: bool = False) -> tuple[int, str]:
    """A command on the pod. SSH as root; agent-user commands go through runuser, exactly
    as agent-entry.sh itself drops privileges."""
    if not quiet:
        print(f"  [{user}] {cmd[:120]}", flush=True)
    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    inner = f"/bin/sh -c {shlex.quote(cmd)}"
    if user == "root":
        full = f"env {env_prefix} {inner}" if env_prefix else inner
    else:
        full = (f"runuser -u {user} -- env HOME=/home/{user} "
                f"{env_prefix} {inner}")
    proc = subprocess.run(
        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port),
         "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
         f"root@{pod.ip}", full],
        capture_output=True, text=True, timeout=timeout)
    text = (proc.stdout + proc.stderr).rstrip()
    if not quiet and text:
        print(text[:800], flush=True)
    return proc.returncode, text


# Bulk SSH transfers (upload_dir / download_tar / trajectory tail) must FAIL FAST when the TCP
# path dies: a single long-lived stream from a Modal container to a RunPod pod was observed
# hanging mid-transfer with both ends asleep (2026-09-06, /bench upload stalled at 78 of 255 MB
# for 10+ min; pod-side Send-Q never drained), and without keepalives nothing notices until the
# subprocess timeout (30 min) — which is how 3 launchers "got stuck" on 2026-09-05 as well.
BULK_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
                 "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]
UPLOAD_CHUNK = 16 * 1024 * 1024


def upload_dir(pod: Pod, local: Path, remote: str) -> int:
    """SUBSTRATE for portal.upload_files: one tarball, shipped in verified chunks.

    Each chunk is its own short SSH session with keepalives and a bounded timeout, retried
    on failure and verified by sha256 on the pod, so a dead connection costs one retry rather
    than the run (see BULK_SSH_OPTS). The assembled tarball is verified once more before it
    is unpacked."""
    import tempfile
    files = [p for p in sorted(local.rglob("*")) if p.is_file()]
    tmpdir = Path(tempfile.mkdtemp(prefix="_upload_"))
    tgz = tmpdir / "up.tgz"
    # COPYFILE_DISABLE: macOS bsdtar otherwise adds AppleDouble ._* entries (154 of them
    # for a 123-file bench — measured 2026-08-21), which materialize as real files on the
    # Linux pod and fail the staging integrity count
    subprocess.run(["tar", "czf", str(tgz), "-C", str(local), "."], check=True,
                   env={**os.environ, "COPYFILE_DISABLE": "1"})
    total = tgz.stat().st_size
    whole = hashlib.sha256(tgz.read_bytes()).hexdigest()
    stage_dir = f"/tmp/_up_{hashlib.sha1(remote.encode()).hexdigest()[:8]}"
    rc, _ = sh(pod, f"rm -rf {stage_dir} && mkdir -p {stage_dir}", quiet=True)
    if rc != 0:
        raise SystemExit(f"upload of {local} -> {remote}: cannot create {stage_dir} on the pod")
    n_chunks = max(1, (total + UPLOAD_CHUNK - 1) // UPLOAD_CHUNK)
    print(f"  upload {local.name}: {total / 1e6:.0f} MB in {n_chunks} chunks -> {remote}",
          flush=True)
    with tgz.open("rb") as fh:
        for idx in range(n_chunks):
            data = fh.read(UPLOAD_CHUNK)
            csum = hashlib.sha256(data).hexdigest()
            target = f"{stage_dir}/c{idx:05d}"
            for attempt in range(6):
                try:
                    proc = subprocess.run(
                        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port), *BULK_SSH_OPTS,
                         f"root@{pod.ip}", f"cat > {target} && sha256sum {target}"],
                        input=data, capture_output=True, timeout=240)
                    if proc.returncode == 0 and csum in proc.stdout.decode(errors="replace"):
                        break
                    why = f"rc={proc.returncode} {proc.stderr.decode(errors='replace')[-200:]}"
                except subprocess.TimeoutExpired:
                    why = "timed out (connection stalled)"
                print(f"  chunk {idx + 1}/{n_chunks} attempt {attempt + 1}/6 failed: {why}",
                      flush=True)
                time.sleep(5)
            else:
                raise SystemExit(f"upload of {local} -> {remote} failed: chunk {idx} never "
                                 f"arrived intact after 6 attempts")
    rc, out = sh(pod, f"cat {stage_dir}/c* > {stage_dir}/all.tgz && "
                      f"sha256sum {stage_dir}/all.tgz && mkdir -p {shlex.quote(remote)} && "
                      f"tar xzf {stage_dir}/all.tgz -C {shlex.quote(remote)} && "
                      f"rm -rf {stage_dir} && echo UPLOAD_OK", timeout=900, quiet=True)
    tgz.unlink(missing_ok=True)
    tmpdir.rmdir()
    if whole not in out or "UPLOAD_OK" not in out:
        raise SystemExit(f"upload of {local} -> {remote} failed at assembly/unpack: {out[-400:]}")
    return len(files)


def download_tar(pod: Pod, remote_cmd: str, timeout: float = 1800) -> bytes:
    """SUBSTRATE for portal.download_files: `remote_cmd` must write a tar stream to stdout."""
    proc = subprocess.run(
        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port), *BULK_SSH_OPTS,
         f"root@{pod.ip}", remote_cmd],
        capture_output=True, timeout=timeout)
    # rc 1 accepted (2026-08-16): GNU tar exits 1 for "file changed as we read it" — routine
    # while the agent is writing files mid-mirror — and the stream is still valid; rejecting
    # it silently discarded every periodic mirror of an active run.
    return proc.stdout if proc.returncode in (0, 1) else b""


def create_pod(name: str, ssh_pubkey: str, ssh_key: str, gpu_count: int = 1) -> Pod:
    """SUBSTRATE for `docker run -d` / create_sandbox: a disposable RTX 4090 pod.

    Capacity errors are retried (not forever: a campaign must hear about a dry datacenter),
    and the pod is polled until its SSH endpoint answers. `gpu_count=2` is the tools-arm
    topology: parameter_search.launch() pins detached searches to the freest LOCAL GPU
    (device 1, away from the agent's interactive Isaac on device 0) — its design assumes a
    second GPU on the same box, not a second pod.
    """
    body = {
        "name": name,
        "imageName": POD_IMAGE,
        "cloudType": "SECURE",
        "gpuTypeIds": [GPU_TYPE],
        "gpuCount": gpu_count,
        "containerDiskInGb": 100,
        "dataCenterIds": [DATACENTER],
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": ssh_pubkey},
    }
    if NETWORK_VOLUME_ID:
        body.update({
            "networkVolumeId": NETWORK_VOLUME_ID,
            "volumeMountPath": SNAPSHOT_MOUNT,
        })
    pod_id = ""
    for attempt in range(30):
        try:
            created = rest("POST", "/pods", body)
            pod_id = created["id"]
            break
        except RuntimeError as exc:
            print(f"  pod create attempt {attempt + 1}/30 failed: {exc}", flush=True)
            if attempt == 29:
                raise SystemExit(f"could not create a pod after 30 attempts: {exc}")
            time.sleep(30)
    print(f"  pod {pod_id} created; waiting for SSH", flush=True)
    for attempt in range(120):            # up to ~10 min for allocation + boot
        time.sleep(5)
        info = rest("GET", f"/pods/{pod_id}")
        ip = info.get("publicIp") or ""
        port = (info.get("portMappings") or {}).get("22")
        if ip and port:
            pod = Pod(id=pod_id, ip=ip, port=int(port), ssh_key=ssh_key)
            rc, _ = sh(pod, "true", quiet=True)
            if rc == 0:
                print(f"  pod {pod_id} up at {ip}:{port}", flush=True)
                return pod
    rest("DELETE", f"/pods/{pod_id}")
    raise SystemExit(f"pod {pod_id} never became reachable; deleted it")


def terminate_pod(pod: Pod) -> None:
    """SUBSTRATE for `docker rm -f`."""
    try:
        rest("DELETE", f"/pods/{pod.id}")
        print(f"  pod {pod.id} terminated", flush=True)
    except RuntimeError as exc:
        print(f"  POD DELETE FAILED — {pod.id} may still be billing, delete it by hand: {exc}",
              flush=True)


def isaac_ok(pod: Pod) -> bool:
    eula = {"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}
    rc, out = sh(pod, f"test -x {VENV}/bin/python && {VENV}/bin/python -c "
                      f"'import isaacsim, isaaclab, h5py; print(\"isaac ok\")'",
                 timeout=900, quiet=True, env=eula)
    return "isaac ok" in out


def provision_pod(pod: Pod) -> None:
    """SUBSTRATE for the image: restore ENV_SNAPSHOT (Isaac venv + repo + node + agent CLIs +
    relay venv + warmed shader caches) captured from the template pod. Idempotent."""
    # Dockerfile.l0's apt layer FIRST, unconditionally: without libGLU/libgomp/vulkan the
    # RunPod pytorch image boots Isaac on CPU-fallback PhysX ("Unable to get IGpuFoundation")
    # — sims still run, 100-700x slower (pc_ram boot: 40+ min vs 3.3 s, measured 2026-08-15).
    # Idempotent and ~30 s; the NVIDIA GLX userspace itself is already in RunPod's image.
    rc, out = sh(pod, "apt-get update -qq >/dev/null 2>&1; "
                      "apt-get install -y -qq zstd libglvnd0 libgl1 libglx0 libegl1 libgles2 "
                      "libvulkan1 vulkan-tools libx11-6 libxt6 libxrandr2 libgomp1 libglu1-mesa "
                      ">/dev/null 2>&1; ldconfig; ldconfig -p | grep -cE 'libGLU|libvulkan|libgomp'",
                 timeout=900, quiet=True)
    if int((out.strip() or "0").splitlines()[-1]) < 3:
        raise SystemExit("the GPU libraries Isaac dlopens did not install; scenes would run "
                         "on CPU-fallback PhysX (measured 100-700x slower)")
    rc, out = sh(pod, f"test -f {ENV_SNAPSHOT} && echo SNAP_OK", timeout=900, quiet=True)
    if "SNAP_OK" not in out and SNAPSHOT_SOURCE_HOST:
        if not SNAPSHOT_TRANSFER_KEY:
            raise SystemExit("RB_SNAPSHOT_SOURCE_HOST requires RB_SNAPSHOT_TRANSFER_KEY")
        print(f"  copying environment snapshot from {SNAPSHOT_SOURCE_HOST}:"
              f"{SNAPSHOT_SOURCE_PORT}", flush=True)
        remote_key = "/root/.ssh/cosigen_snapshot_transfer"
        copy = subprocess.run(
            ["scp", "-i", pod.ssh_key, "-P", str(pod.port),
             "-o", "StrictHostKeyChecking=accept-new", SNAPSHOT_TRANSFER_KEY,
             f"root@{pod.ip}:{remote_key}"],
            capture_output=True, text=True, timeout=300)
        if copy.returncode != 0:
            raise SystemExit(f"snapshot transfer key upload failed: {copy.stderr[-500:]}")
        sh(pod, f"chmod 600 {remote_key}; mkdir -p {SNAPSHOT_MOUNT}", quiet=True)
        rc, out = sh(
            pod,
            f"scp -i {remote_key} -P {SNAPSHOT_SOURCE_PORT} "
            f"-o StrictHostKeyChecking=accept-new "
            f"root@{SNAPSHOT_SOURCE_HOST}:{SNAPSHOT_SOURCE_PATH} {ENV_SNAPSHOT}",
            timeout=7200)
        sh(pod, f"rm -f {remote_key}", quiet=True)
        if rc != 0:
            raise SystemExit(f"snapshot copy failed from {SNAPSHOT_SOURCE_HOST}:"
                             f"{SNAPSHOT_SOURCE_PORT}")
        if SNAPSHOT_SHA256:
            rc, out = sh(pod, f"sha256sum {ENV_SNAPSHOT}", timeout=1800, quiet=True)
            got = out.strip().split()[0] if out.strip() else ""
            if got != SNAPSHOT_SHA256:
                raise SystemExit(f"snapshot hash mismatch: got {got}, "
                                 f"expected {SNAPSHOT_SHA256}")
        print("  environment snapshot copied and verified", flush=True)
    if not isaac_ok(pod):
        print(f"  restoring environment from {ENV_SNAPSHOT}", flush=True)
        t0 = time.time()
        rc, out = sh(pod, f"test -f {ENV_SNAPSHOT} && echo SNAP_OK", timeout=900, quiet=True)
        if "SNAP_OK" not in out:
            raise SystemExit(f"{ENV_SNAPSHOT} is missing on the network volume; "
                             "capture it first (see eval/scripts/snapshot_pod_env.sh)")
        rc, out = sh(pod, f"tar -I 'zstd -T0' -xf {ENV_SNAPSHOT} -C /", timeout=1800)
        if rc != 0 or not isaac_ok(pod):
            raise SystemExit("the snapshot restored but Isaac does not import; "
                             "the task cannot be attempted")
        print(f"  environment restored in {time.time() - t0:.0f}s", flush=True)
    # What the L1 image would also have carried: node + the agent CLIs + the relay venv.
    # Symlinks FIRST, check second — the CLIs live under /opt/npm/bin (snapshot content),
    # which is not on a fresh pod's PATH until these links exist.
    sh(pod, f"ln -sfn /opt/node-v22.11.0-linux-x64/bin/node /usr/local/bin/node && "
            f"ln -sfn /opt/node-v22.11.0-linux-x64/bin/npm /usr/local/bin/npm && "
            f"ln -sfn /opt/npm/bin/claude /usr/local/bin/claude && "
            f"test -x /opt/npm/bin/codex && ln -sfn /opt/npm/bin/codex /usr/local/bin/codex; "
            f"true", quiet=True)
    rc, out = sh(pod, "for t in node claude codex; do command -v $t >/dev/null || "
                      "echo MISSING_$t; done; "
                      "test -x /opt/relay-venv/bin/python || echo MISSING_relayvenv",
                 quiet=True)
    if "MISSING" in out:
        raise SystemExit(f"the snapshot lacks baked tools ({out.strip()}); "
                         "re-capture it with eval/scripts/snapshot_pod_env.sh")
    sh(pod, f"id -u {AGENT_USER} >/dev/null 2>&1 || useradd -m -s /bin/bash {AGENT_USER}",
       quiet=True)
    # The warmed caches were captured under /root; the agent is a different user, so stage
    # them where the entry's XDG_CACHE_HOME points (the rb-ovcache volume's role).
    sh(pod, f"mkdir -p /ovcache && cp -a /root/.cache/. /ovcache/ 2>/dev/null; "
            f"test -d /root/.nv && mkdir -p /home/{AGENT_USER} && "
            f"cp -a /root/.nv /home/{AGENT_USER}/.nv 2>/dev/null; "
            f"chown -R {AGENT_USER}:{AGENT_USER} /ovcache /home/{AGENT_USER}; true",
       timeout=1800, quiet=True)
    # The agent (non-root) must be able to run the venv — prove it, fix perms only if needed.
    rc, out = sh(pod, f"{VENV}/bin/python -c 'import sys; print(\"venv-as-agent ok\")'",
                 user=AGENT_USER, quiet=True)
    if "venv-as-agent ok" not in out:
        sh(pod, "chmod -R a+rX /opt/cosigen", timeout=1800, quiet=True)
        rc, out = sh(pod, f"{VENV}/bin/python -c 'print(\"venv-as-agent ok\")'",
                     user=AGENT_USER, quiet=True)
        if "venv-as-agent ok" not in out:
            raise SystemExit("the agent user cannot execute the Isaac venv")


def mount_like_docker(pod: Pod, stage: Path, task_dir: Path) -> None:
    """SUBSTRATE for the four `-v` mounts (verbatim from run_agent_sandbox)."""
    # docker -v REPLACES the mount point; the env snapshot ships its own /bench (env2:
    # 277 files — measured 2026-08-21) and /bench may not be removable as a directory,
    # so delete CONTENTS and verify empty before staging, or the integrity count below
    # sees the union and refuses the pod
    sh(pod, f"mkdir -p /bench /task /workspace/.agent /submissions /ovcache {RELAY_DIR}; "
            f"find /bench /task -mindepth 1 -delete; "
            f"chown -R {AGENT_USER}:{AGENT_USER} /workspace /submissions /ovcache", quiet=True)
    rc, out = sh(pod, "find /bench /task -mindepth 1 | wc -l", quiet=True)
    if int(out.strip() or 1) != 0:
        raise SystemExit(f"/bench,/task not empty after clearing ({out.strip()} entries) — "
                         f"refusing to stage over snapshot leftovers")
    n_bench = upload_dir(pod, stage / "bench", "/bench")
    n_task = upload_dir(pod, task_dir, "/task")
    sh(pod, "chown -R root:root /bench /task && chmod -R a+rX,go-w /bench /task && "
            "chmod 755 /bench /task", quiet=True)
    for remote, expected in (("/bench", n_bench), ("/task", n_task)):
        rc, out = sh(pod, f"find {remote} -type f | wc -l", quiet=True)
        if int(out.strip() or 0) != expected:
            raise SystemExit(f"{remote}: staged {out.strip()} of {expected} files")
        print(f"  mounted {remote}: {expected} files", flush=True)


def start_relay(pod: Pod, port: int, upstream_base: str, api_key: str,
                force_model: str | None = None,
                responses_via_chat: str | None = None,
                spool_dir: str | None = None) -> str:
    """The in-pod trajectory relay — run_agent_sandbox.start_relay with SSH transport.

    Root-owned 700 with its own interpreter, so the agent (a different user) cannot read the
    trajectory. The credential goes through the SSH env, never a command line.
    """
    src = Path(__file__).resolve().parents[2] / "sim_gen" / "super_relay"
    tmp = Path("/tmp/_relay_stage")
    subprocess.run(["rm", "-rf", str(tmp)], check=False)
    tmp.mkdir(parents=True)
    for p in sorted(src.glob("*.py")):
        (tmp / p.name).write_bytes(p.read_bytes())
    upload_dir(pod, tmp, RELAY_DIR)
    sh(pod, f"chown -R root:root {RELAY_DIR} && chmod -R go-rwx {RELAY_DIR} && "
            f"chmod 700 {RELAY_DIR} && mkdir -p {RELAY_DIR}/trajlog", quiet=True)
    relay_py = f"{RELAY_VENV}/bin/python"
    rc, out = sh(pod, f"{relay_py} -c 'import fastapi, uvicorn, httpx' 2>&1 || "
                      f"echo NEED_INSTALL", quiet=True)
    if "NEED_INSTALL" in out:
        raise SystemExit("the relay venv is missing from the snapshot; re-capture it")
    relay_env = {"SUPER_RELAY_API_KEY": api_key}
    if force_model:
        relay_env["SUPER_RELAY_FORCE_MODEL"] = force_model
    if responses_via_chat:
        # Via the SSH env, never the command line: the URL embeds the gateway access key,
        # and /proc/<pid>/cmdline is readable by the agent user (same rule as the API key).
        relay_env["SUPER_RELAY_RESPONSES_VIA_CHAT"] = responses_via_chat
    spool_flag = ""
    if spool_dir:
        # Spool mode: the relay makes NO network calls; a laptop-side daemon (outbound SSH
        # only, eval/scripts/laptop_aidp_bridge.py) fulfils each request through these files.
        sh(pod, f"mkdir -p {spool_dir}/req {spool_dir}/resp && chmod -R go-rwx {spool_dir}",
           quiet=True)
        spool_flag = f"--spool-dir {spool_dir} "
    sh(pod, f"cd {RELAY_DIR} && nohup {relay_py} {RELAY_DIR}/server.py "
            f"--host 127.0.0.1 --port {port} --log-dir {RELAY_DIR}/trajlog "
            f"{spool_flag}"
            f"--upstream-base {upstream_base} > {RELAY_DIR}/relay.log 2>&1 & "
            f"echo $! > {RELAY_DIR}/relay.pid; sleep 8",
       env=relay_env, quiet=True)
    rc, out = sh(pod,
                 f"kill -0 $(cat {RELAY_DIR}/relay.pid) 2>/dev/null && echo RELAY_ALIVE; "
                 f"grep -q 'address already in use' {RELAY_DIR}/relay.log && echo BIND_FAILED; "
                 f"curl -s --max-time 10 http://127.0.0.1:{port}/health", quiet=True)
    if "RELAY_ALIVE" not in out or "BIND_FAILED" in out:
        sh(pod, f"tail -20 {RELAY_DIR}/relay.log")
        raise SystemExit("the trajectory relay did not come up; a run we cannot replay is one "
                         "that did not happen")
    if f"{RELAY_DIR}/trajlog/raw_requests.jsonl" not in out:
        raise SystemExit(f"the listener on port {port} is not our relay; refusing to run untraced")
    rc, _ = sh(pod, f"ls {RELAY_DIR}", user=AGENT_USER, quiet=True)
    if rc == 0:
        raise SystemExit(f"{RELAY_DIR} is readable by the agent")
    return f"http://127.0.0.1:{port}"


def wait_for_tunnel(pod: Pod, timeout_s: float = 300.0) -> None:
    """Block until the pod-local reverse tunnel answers (the laptop forwarder's /_health).

    The laptop's tunnel daemon polls the pod list every ~45 s, so a fresh pod's tunnel
    appears shortly after it turns RUNNING; failing after `timeout_s` beats letting the
    agent start and burn its budget against a dead upstream."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rc, out = sh(pod, "curl -s --max-time 5 http://127.0.0.1:8899/_health || true",
                     quiet=True)
        if "ok" in out:
            print(f"  tunnel to laptop verified ({time.time() - t0:.0f}s)", flush=True)
            return
        time.sleep(15)
    raise SystemExit(
        "the pod-local tunnel (127.0.0.1:8899) never answered — are "
        "laptop_aidp_forwarder.py and laptop_tunnel_daemon.py running on the laptop?")


CODEX_MODELS_JSON = Path(__file__).resolve().parent / "codex_models.json"


def write_codex_config(pod: Pod, relay_url: str, model: str) -> None:
    """Point codex at the relay's Responses endpoint AND give it the model catalog.

    The catalog (`model_catalog_json`) is not optional for these models: gpt-5.6-sol/terra
    are seed-code aliases codex does NOT recognize natively, and without a catalog entry
    describing the model's capabilities codex falls back to a bare model with NO tool harness
    — it sent `tools: []`, the agent reported "no shell/exec available", wrote nothing, and
    the keep-going loop spun for the whole budget (measured 2026-09-06 across the entire
    fleet; every pod idle-GPU, zero solutions). The Sept-4 base_v1 codex runs that DID solve
    shipped exactly this catalog + `model_catalog_json`; this reproduces that config verbatim
    (models.json recovered from base_v1's agent_home)."""
    # wire_api "responses", NOT "chat": codex hard-removed chat-completions support in
    # Feb 2026 (a "chat" config makes every leg die at config load). The relay implements
    # /v1/responses and logs it.
    catalog = CODEX_MODELS_JSON.read_bytes()
    if b'"gpt-5.6' not in catalog:
        raise SystemExit(f"{CODEX_MODELS_JSON} does not look like the codex model catalog")
    # Stream the 190 KB catalog over ssh stdin (too big for a heredoc arg); config.toml names
    # it. Every knob below matches base_v1's working config.toml.
    toml = (f'model = "{model}"\n'
            f'model_provider = "relay"\n'
            f'model_catalog_json = "/home/{AGENT_USER}/.codex/models.json"\n'
            f'model_reasoning_effort = "high"\n'
            f'web_search = "disabled"\n\n'
            f'[features]\n'
            f'multi_agent = false\n\n'
            f'[model_providers.relay]\n'
            f'name = "relay"\n'
            f'base_url = "{relay_url}/v1"\n'
            f'env_key = "OPENAI_API_KEY"\n'
            f'wire_api = "responses"\n'
            f'stream_idle_timeout_ms = 1800000\n'
            f'request_max_retries = 8\n'
            f'stream_max_retries = 20\n\n'
            f'[projects."/"]\n'
            f'trust_level = "trusted"\n')
    sh(pod, f"mkdir -p /home/{AGENT_USER}/.codex && "
            f"cat > /home/{AGENT_USER}/.codex/config.toml << 'EOF'\n{toml}EOF\n"
            f"true", quiet=True)
    # push the catalog itself
    rc = subprocess.run(
        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port),
         "-o", "StrictHostKeyChecking=accept-new", f"root@{pod.ip}",
         f"cat > /home/{AGENT_USER}/.codex/models.json"],
        input=catalog, capture_output=True, timeout=120).returncode
    if rc != 0:
        raise SystemExit("failed to upload codex model catalog to the pod")
    rc, out = sh(pod, f"chown -R {AGENT_USER}:{AGENT_USER} /home/{AGENT_USER}/.codex && "
                      f"test -s /home/{AGENT_USER}/.codex/models.json && "
                      f"grep -q '{model}' /home/{AGENT_USER}/.codex/models.json && echo CATALOG_OK",
                 quiet=True)
    if "CATALOG_OK" not in out:
        raise SystemExit(f"codex model catalog missing {model!r} on the pod after upload")


AGENT_STATE = "agent_home.tgz"   # the CLI's own session state, kept beside the run's artifacts


def mirror_agent_state(pod: Pod, run_dir: Path) -> bool:
    """Copy the agent CLI's session state out so the run can be RESUMED (verbatim intent from
    run_agent_sandbox.mirror_agent_state; taken once, at teardown)."""
    data = download_tar(pod, f"test -d /home/{AGENT_USER} && tar czf - -C /home {AGENT_USER}",
                        timeout=900)
    if not data:
        print("  no agent session state to preserve (the CLI never started?)", flush=True)
        return False
    (run_dir / AGENT_STATE).write_bytes(data)
    print(f"  agent session state preserved: {len(data) / 1e6:.1f} MB "
          f"({run_dir / AGENT_STATE})", flush=True)
    return True


def restore_run_state(pod: Pod, run_dir: Path) -> bool:
    """Put a previous run's workspace, submissions and session state back into a fresh pod."""
    ws, subs = run_dir / "workspace", run_dir / "submissions"
    if not ws.is_dir():
        raise SystemExit(f"{ws} does not exist — nothing to resume from")
    tgz = run_dir / ".restore.tgz"
    parts = ["workspace"] + (["submissions"] if subs.is_dir() else [])
    subprocess.run(["tar", "czf", str(tgz), "-C", str(run_dir), *parts], check=True,
                   env={**os.environ, "COPYFILE_DISABLE": "1"})
    rc = subprocess.run(
        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port),
         "-o", "StrictHostKeyChecking=accept-new", f"root@{pod.ip}",
         # mkdir -p AFTER the tar: a fork source carries no submissions/ on purpose (fresh
         # submissions for the new task), and without this the pod would lose /submissions.
         "rm -rf /workspace /submissions && tar xzf - -C / && mkdir -p /workspace /submissions && "
         f"chown -R {AGENT_USER}:{AGENT_USER} /workspace /submissions 2>/dev/null; true"],
        stdin=tgz.open("rb"), capture_output=True, timeout=1800).returncode
    tgz.unlink(missing_ok=True)
    if rc != 0:
        raise SystemExit("restoring the previous run's workspace failed")
    state = run_dir / AGENT_STATE
    if not state.exists():
        print(f"  no {AGENT_STATE} beside this run: its files are restored but the conversation "
              f"is not — the agent will start fresh against its own previous work", flush=True)
        return False
    rc = subprocess.run(
        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port),
         "-o", "StrictHostKeyChecking=accept-new", f"root@{pod.ip}",
         f"rm -rf /home/{AGENT_USER} && tar xzf - -C /home && "
         f"chown -R {AGENT_USER}:{AGENT_USER} /home/{AGENT_USER} && echo STATE_RESTORED"],
        stdin=state.open("rb"), capture_output=True, timeout=1800)
    if b"STATE_RESTORED" not in rc.stdout:
        raise SystemExit("the agent session state did not restore; refusing to resume as if it "
                         "had (the run would silently start a new conversation)")
    print("  agent session state restored — the CLI resumes its own conversation", flush=True)
    return True


def start_entry(pod: Pod, env: dict) -> None:
    """SUBSTRATE for the container's ENTRYPOINT: the same agent-entry.sh, run as root."""
    entry = Path(__file__).resolve().parents[1] / "docker" / "entrypoints"
    tmp = Path("/tmp/_entry_stage")
    subprocess.run(["rm", "-rf", str(tmp)], check=False)
    tmp.mkdir(parents=True)
    (tmp / "agent-entry.sh").write_bytes((entry / "agent-entry.sh").read_bytes())
    (tmp / "submit").write_bytes((entry / "submit").read_bytes())
    (tmp / "verify_solution.py").write_bytes(
        (Path(__file__).resolve().parent / "verify_solution.py").read_bytes())
    upload_dir(pod, tmp, "/opt/entrypoints")
    rc, out = sh(pod,
                 "chown root:root /opt/entrypoints/* && "
                 "chmod 755 /opt/entrypoints/agent-entry.sh && "
                 "install -m 755 /opt/entrypoints/submit /usr/local/bin/submit && "
                 "mkdir -p /opt/harness/eval/scripts && "
                 "mv /opt/entrypoints/verify_solution.py "
                 "/opt/harness/eval/scripts/verify_solution.py && "
                 "chmod 700 /opt/harness/eval/scripts/verify_solution.py && "
                 "chmod -R go-rwx /opt/harness && "
                 "ln -sfn /opt/harness/eval/scripts/verify_solution.py "
                 "/opt/verify_solution.py && echo ENTRY_OK", quiet=True)
    if "ENTRY_OK" not in out:
        raise SystemExit("could not install the entry script / success check")
    rc, out = sh(pod, f"{VENV}/bin/python /opt/verify_solution.py --help > /dev/null 2>&1; "
                      f"echo rc=$?", quiet=True)
    if "rc=0" not in out:
        sh(pod, f"{VENV}/bin/python /opt/verify_solution.py --help 2>&1 | tail -5")
        raise SystemExit("the success check cannot even start; the agent could never be told "
                         "the task is solved")
    # `cd /` FIRST: docker runs this entry at `/` (no WORKDIR anywhere in the images), but
    # an SSH exec starts at /root — mode 700, unreadable to the agent user, so every process
    # the agent CLI spawned died at chdir and the agents wrote solutions BLIND without ever
    # running a sim (measured 2026-08-15: 39 claude runs, zero GPU use, "exec is fully
    # dead" in the transcripts). `/` is what a docker container's entry sees; mimic that.
    sh(pod, f"cd / && nohup /opt/entrypoints/agent-entry.sh > {RELAY_DIR}/container.log 2>&1 & "
            f"sleep 5; pgrep -f '[a]gent-entry' | wc -l", env=env, timeout=120)


def running(pod: Pod) -> bool:
    """SUBSTRATE for `docker inspect .State.Running`: the entry script IS the container."""
    rc, out = sh(pod, "pgrep -f '[a]gent-entry' | wc -l", quiet=True)
    return (out.strip() or "0") != "0"


def stop_entry(pod: Pod) -> None:
    """SUBSTRATE for `docker stop -t 30`."""
    sh(pod, "pkill -TERM -f '[a]gent-entry'; pkill -TERM -f '[c]laude'; "
            "pkill -TERM -f '[c]odex'; sleep 30; "
            "pkill -KILL -f '[a]gent-entry'; pkill -KILL -f '[c]laude'; "
            "pkill -KILL -f '[c]odex'; true", timeout=120, quiet=True)


def mirror_back(pod: Pod, run_dir: Path) -> None:
    """SUBSTRATE for the bind mounts being two-way (60 s poll, same as the sandbox path)."""
    data = download_tar(pod, "tar czf - -C / workspace submissions 2>/dev/null", timeout=900)
    if not data:
        print("  mirror: workspace tarball came back empty", flush=True)
        return
    tgz = run_dir / ".back.tgz"
    tgz.write_bytes(data)
    subprocess.run(["tar", "xzf", str(tgz), "-C", str(run_dir)], check=False)
    tgz.unlink(missing_ok=True)


def mirror_trajectory(pod: Pod, run_dir: Path) -> int:
    """Append only the new bytes of the relay's log (verbatim intent from the sandbox path)."""
    remote = f"{RELAY_DIR}/trajlog/raw_requests.jsonl"
    local = run_dir / "opt" / "relay" / "trajlog" / "raw_requests.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    rc, out = sh(pod, f"stat -c %s {remote} 2>/dev/null || echo 0", quiet=True)
    try:
        remote_size = int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        remote_size = 0
    have = local.stat().st_size if local.exists() else 0
    if remote_size <= have:
        return remote_size
    proc = subprocess.run(
        ["ssh", "-i", pod.ssh_key, "-p", str(pod.port), *BULK_SSH_OPTS, f"root@{pod.ip}",
         f"tail -c +{have + 1} {remote}"],
        capture_output=True, timeout=900)
    if proc.returncode != 0 or not proc.stdout:
        print(f"  mirror: trajectory tail came back empty; pod has {remote_size} bytes, "
              f"we have {have}", flush=True)
        return remote_size
    with local.open("ab") as fh:
        fh.write(proc.stdout)
    return remote_size


def relay_pulse(pod: Pod) -> tuple[bool, int, int, int]:
    """(relay alive, requests completed 200, bytes recorded, relay log bytes) — the sandbox
    path's evidence rules, unchanged."""
    rc, out = sh(pod,
                 f"echo ALIVE=$(if test -s {RELAY_DIR}/relay.pid; then "
                 f"kill -0 $(cat {RELAY_DIR}/relay.pid) 2>/dev/null && echo 1 || echo 0; "
                 f"else pgrep -c -f '[s]erver.py --host' 2>/dev/null || echo 0; fi); "
                 f"echo SERVED=$(grep -c '200 OK' {RELAY_DIR}/relay.log 2>/dev/null || echo 0); "
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


# ---------------------------------------------------------------------------------------
# main(): run_agent_sandbox.main() with the substrate calls swapped; the run logic —
# condition, task folder, budget loop, stamping, trajectory guard, run.json — is its code.
# ---------------------------------------------------------------------------------------

def single_stage(exp: Path) -> Path:
    stages = sorted((exp / "stages").iterdir())
    if not stages:
        sys.exit(f"no stages under {exp}")
    if len(stages) > 1:
        sys.exit(f"run_agent runs one scene + one task; this experiment has "
                 f"{len(stages)} stages {[s.name for s in stages]}")
    return stages[0]


def stage_record(exp: Path, stage: Path) -> dict:
    receipt = json.loads((exp / "resolved.json").read_text())
    hits = [r for r in receipt["stages"] if r["dir"] == stage.name]
    if not hits:
        sys.exit(f"stage {stage.name} not in {exp / 'resolved.json'} — rebuild the experiment")
    record = dict(hits[0])
    record["set_states"] = receipt.get("set_states", True)
    record.setdefault("control_mode_frozen", False)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="?", help="built experiment dir (from build_env.py)")
    ap.add_argument("--config", default="no_tools",
                    help="experimental CONDITION: a name in eval/configs/ or a path "
                         "(default: %(default)s)")
    ap.add_argument("--agent", default=None, choices=["claude", "codex"])
    ap.add_argument("--model", default=None, help="model id (MODEL env for the agent CLI)")
    ap.add_argument("--force-model", default=None,
                    help="pin EVERY upstream request to this id at the relay (subagent traffic "
                         "included); default: forward what the CLI asks for")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="cap on the response length the agent CLI asks for "
                         "(CLAUDE_CODE_MAX_OUTPUT_TOKENS); default: the CLI's own")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="wall-clock kill budget (minutes, default 240)")
    ap.add_argument("--auto-submit-min", type=float, default=None,
                    help="snapshot solution/ as a submission every N minutes")
    ap.add_argument("--relay-port", type=int, default=8118)
    ap.add_argument("--keep-going", action="store_true",
                    help="when the agent stops before the budget, continue the same "
                         "conversation instead of ending the run")
    ap.add_argument("--resume-from", default="",
                    help="continue a previous run from its mirrored run dir (<exp>/runs/<name>)")
    ap.add_argument("--pod", default="",
                    help="SUBSTRATE: run on this existing pod id (skips create+terminate); "
                         "the pod must be freshly provisioned — no reset path exists here "
                         "because pods are disposable")
    ap.add_argument("--reattach", action="store_true",
                    help="with --pod and --started: the run is ALREADY going on that pod (a "
                         "launcher restarted); skip every setup step, only resume the budget/"
                         "mirror loop from the original start time and own the pod's teardown. "
                         "Added 2026-09-06 so a rescheduled launcher never creates a second pod")
    ap.add_argument("--started", default="",
                    help="--reattach: the original run's start time (ISO 8601, UTC) — the budget "
                         "counts from here, not from now")
    ap.add_argument("--terminate-attached", action="store_true",
                    help="with --pod, treat that pre-reserved pod as launcher-owned and "
                         "terminate it after the run or any setup failure")
    ap.add_argument("--keep-pod", action="store_true",
                    help="SUBSTRATE: leave the pod running at teardown (debugging); the "
                         "default terminates it — a pod provisions in ~2 min, so unlike the "
                         "sandbox pool there is nothing worth keeping warm")
    ap.add_argument("--gpu-count", type=int, default=1,
                    help="GPUs on the pod (2 for tool-arm runs: agent's interactive Isaac on "
                         "device 0, parameter_search.launch's detached searches on device 1)")
    ap.add_argument("--resume-new-task", action="store_true",
                    help="with --resume-from: the restored conversation continues onto THIS "
                         "experiment's task — the first leg sends the new task's instructions "
                         "(carryover variant) into the resumed session instead of the nudge")
    ap.add_argument("--gpt-via-tunnel", action="store_true",
                    help="codex only: route GPT traffic to the AIDP gateway through the "
                         "pod-local reverse tunnel (127.0.0.1:8899 -> laptop -> AIDP). "
                         "Requires laptop_aidp_forwarder.py + laptop_tunnel_daemon.py "
                         "running on the laptop; the runner verifies the tunnel before "
                         "starting the agent. BANNED (inbound tunnel) — use --gpt-via-bridge.")
    ap.add_argument("--gpt-via-bridge", action="store_true",
                    help="codex only: the relay makes NO network calls — it spools each "
                         "request to /opt/relay/spool and a laptop-side daemon "
                         "(eval/scripts/laptop_aidp_bridge.py, outbound SSH only) fulfils it "
                         "against AIDP from inside the corp network. No inbound path, no "
                         "credentials on the pod. The bridge daemon must be running.")
    ap.add_argument("--via-seed-code", action="store_true",
                    help="reach the model through the seed-code gateway (SEED_CODE_BASE) with "
                         "the relay-owned seed-code key; the relay rewrites the model id to "
                         "the gateway alias (SEED_CODE_ALIASES, or --force-model). Claude and "
                         "codex alike — the gateway speaks both native APIs.")
    ap.add_argument("--hint-file", action="append", default=None,
                    help="generalization arms: copy this local file into the run's "
                         "/task/hints/ and announce it in instructions.md (repeatable, "
                         "e.g. another task's solve.py, or the same task's solution for a "
                         "different embodiment)")
    ap.add_argument("--hint-note", default=None,
                    help="the sentence in instructions.md that explains what the hint "
                         "files are (what task/robot they solve, and that they are a "
                         "REFERENCE, not this run's solution)")
    ap.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    ap.add_argument("--ssh-pub", default="",
                    help="public key text for the pod (default: <ssh-key>.pub's contents)")
    ap.add_argument("--run", default=None, help="run name (default: <agent>_<timestamp>)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    condition = load_condition(args.config)
    if not args.exp:
        ap.error("an experiment dir is required")
    agent = args.agent or "claude"
    budget_min = float(args.budget_min or 240)
    auto_submit_min = args.auto_submit_min
    model = args.model
    if agent == "codex" and not model:
        sys.exit("--agent codex needs --model (the codex config pins the provider+model)")

    exp = Path(args.exp).resolve()
    stage = single_stage(exp)
    record = stage_record(exp, stage)
    facts = {"set_states": record["set_states"],
             "control_mode_frozen": record["control_mode_frozen"]}
    built = (record.get("condition") or {}).get("config")
    if built and Path(built).name != condition.path.name:
        print(f"note: this world was built under '{Path(built).name}' and this run declares "
              f"'{condition.path.name}' — check the features still match")

    describe_file = stage / "describe.md"
    if not describe_file.exists():
        sys.exit(f"missing {describe_file} — rebuild the experiment")

    run_name = args.run or f"{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = exp / "runs" / run_name
    if args.reattach:
        if not (args.pod and args.started and args.run):
            ap.error("--reattach needs --pod, --started and --run")
        started_at = datetime.fromisoformat(args.started)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
    elif run_dir.exists() and not args.resume_from:
        sys.exit(f"refusing to overwrite existing {run_dir}")

    task_dir = prompts.render_task_dir(
        run_dir / "task",
        scene=record["preset"].split(".")[1],
        preset=record["preset"],
        describe_text=describe_file.read_text(),
        facts=facts,
        condition=condition,
        budget_min=budget_min,
        carryover=bool(args.resume_new_task),
    )
    if args.hint_file:
        # Generalization arms: reference material (e.g. another task's solve.py) shipped
        # under /task/hints/, announced at the END of instructions.md (the CLI's first
        # message). The hash map below is computed AFTER this block, so the hint files and
        # the modified instructions are part of the run's citable task_files record.
        import shutil as _shutil
        hints_dir = task_dir / "hints"
        hints_dir.mkdir(exist_ok=True)
        names = []
        for hf in args.hint_file:
            src_f = Path(hf).expanduser().resolve()
            if not src_f.is_file():
                sys.exit(f"--hint-file {hf}: no such file")
            _shutil.copyfile(src_f, hints_dir / src_f.name)
            names.append(f"/task/hints/{src_f.name}")
        note = args.hint_note or "Reference material is provided for this run."
        with (task_dir / "instructions.md").open("a") as fh:
            fh.write("\n\n== Provided reference ==\n" + note + "\n"
                     + "".join(f"  * {n}\n" for n in names))
    task_files = {
        str(p.relative_to(task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(task_dir.rglob("*")) if p.is_file()
    }
    workspace = run_dir / "workspace"
    submissions = run_dir / "submissions"

    ssh_pub = args.ssh_pub or Path(args.ssh_key + ".pub").read_text().strip()

    node_bin = "/opt/node-v22.11.0-linux-x64/bin"
    container_env = {
        "AGENT": agent,
        "PYTHONPATH": "/bench:/task/tools",
        "XDG_CACHE_HOME": "/ovcache",
        "PATH": f"{VENV}/bin:/opt/npm/bin:{node_bin}:/usr/local/sbin:/usr/local/bin:"
                f"/usr/sbin:/usr/bin:/sbin:/bin",
        "TMPDIR": "/workspace/tmp",
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        # 3900 = the verify watchdog's 3600 plus teardown headroom (raised with the watchdog,
        # 2026-08-15); the inner watchdog is what fires first.
        "SUCCESS_CHECK": f"timeout -k 30 3900 {VENV}/bin/python /opt/verify_solution.py "
                         f"--preset {record['preset']} --solution /workspace/solution",
    }
    if model:
        container_env["MODEL"] = model
        # reached-state grading source of record: every robobench env the agent boots
        # snapshots its states periodically + on score improvements (robobench/core/env.py
        # _statelog_tick); mirrored back with the workspace like everything else
        container_env["COSIGEN_STATELOG"] = "/workspace/.statelog"
    if args.max_output_tokens:
        container_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    if args.keep_going:
        container_env["KEEP_GOING"] = "1"

    if args.dry_run:
        print(f"task folder assembled: {task_dir}")
        print(f"pod: image={POD_IMAGE} gpu={GPU_TYPE} dc={DATACENTER} volume={NETWORK_VOLUME_ID}")
        print(f"env: {json.dumps(container_env, indent=2)}")
        return

    workspace.mkdir(parents=True, exist_ok=True)
    submissions.mkdir(parents=True, exist_ok=True)
    started = started_at if args.reattach else datetime.now(timezone.utc)

    pod_name = f"rb-{exp.name}-{run_name}"[:60]
    if args.pod:
        # --reattach may land while the pod is still allocating (the previous launcher died
        # right after creating it): wait for the endpoint as create_pod would, up to ~10 min.
        for attempt in range(120 if args.reattach else 1):
            info = rest("GET", f"/pods/{args.pod}")
            ssh_port = (info.get("portMappings") or {}).get("22")
            if info.get("publicIp") and ssh_port:
                break
            if not args.reattach:
                sys.exit(f"pod {args.pod} has no reachable SSH endpoint yet")
            time.sleep(5)
        else:
            sys.exit(f"pod {args.pod} never exposed an SSH endpoint (status "
                     f"{info.get('desiredStatus')}); reattach failed — it is NOT terminated here")
        pod = Pod(id=args.pod, ip=info["publicIp"], port=int(ssh_port), ssh_key=args.ssh_key)
        print(f"pod {pod.id} attached", flush=True)
    else:
        pod = create_pod(pod_name, ssh_pub, args.ssh_key, gpu_count=args.gpu_count)
    # pod.json the moment the pod exists (2026-09-06): a launcher that dies after this line
    # leaves a record of WHICH pod is its run's, so a restarted launcher can reattach to it
    # instead of creating a second one (the September orphan/duplicate churn).
    (run_dir / "pod.json").write_text(json.dumps({
        "pod_id": pod.id, "name": pod_name, "ip": pod.ip, "port": pod.port,
        "gpu_count": args.gpu_count, "started": started.isoformat(timespec="seconds"),
        "reattached": bool(args.reattach),
    }, indent=2) + "\n")
    # A setup failure must not leak a billing pod: everything between creation and the entry
    # starting is fatal-and-terminate (unless the pod is the caller's own or --keep-pod).
    owns_pod = not args.pod or args.terminate_attached or args.reattach
    force_model = args.force_model
    if args.via_seed_code and not force_model:
        if model not in SEED_CODE_ALIASES:
            raise SystemExit(f"--via-seed-code: no gateway alias known for model {model!r}; "
                             f"pass --force-model model_hub/<alias>")
        force_model = SEED_CODE_ALIASES[model]
    do_setup = not args.reattach
    if args.reattach:
        alive, served, recorded, _ = relay_pulse(pod)
        entry_up = running(pod)
        if not entry_up and not alive and recorded == 0:
            # The previous launcher died BEFORE the agent ever started (mid-provisioning:
            # clear_organic_objects, 2026-09-06 20:05). Nothing has been spent on this pod, so
            # finishing the setup here is a fresh, clean run — every setup step is idempotent
            # on a pod without an entry (provision_pod is, mount_like_docker clears /bench and
            # /task first, no relay/entry exists to duplicate). Budget counts from now.
            started = datetime.now(timezone.utc)
            print(f"reattached to pod {pod.id}: the agent never started here (no entry, no "
                  f"relay, nothing recorded) — completing setup on this pod; budget counts "
                  f"from {started.isoformat(timespec='seconds')}", flush=True)
            do_setup = True
        else:
            # Everything on the pod (env, /bench, /task, relay, entry) is in place and running;
            # touching any of it would restart the agent mid-run. Just look, then loop.
            print(f"reattached to pod {pod.id}: entry running={entry_up} relay alive={alive} "
                  f"served={served} recorded={recorded} B; budget counts from "
                  f"{started.isoformat(timespec='seconds')}", flush=True)
    if do_setup:
        try:
            provision_pod(pod)
            mount_like_docker(pod, stage, task_dir)
            if args.resume_from:
                resumed = restore_run_state(pod, Path(args.resume_from).resolve())
                if resumed:
                    container_env["RESUME"] = "1"
                    if args.resume_new_task:
                        container_env["RESUME_NEW_TASK"] = "1"
                elif args.resume_new_task:
                    raise SystemExit("--resume-new-task without a restored conversation would run "
                                     "the new task cold while claiming a fork; refusing")

            spool = None
            if args.via_seed_code:
                upstream_base, api_key = SEED_CODE_BASE, SEED_CODE_KEY
                via_chat = None       # native Anthropic + native Responses upstream, no bridge
            elif agent == "codex" and args.gpt_via_bridge:
                # AIDP via the outbound-only laptop bridge: the relay spools requests to files;
                # laptop_aidp_bridge.py pulls them over outbound SSH, calls AIDP from the corp
                # network, and writes responses back. No network path to AIDP exists on the pod,
                # so upstream_base is only a label the relay records; via_chat marks the request
                # kind for the log (Responses -> chat bridge shape, same as the tunnel path).
                upstream_base, api_key = OPENAI_BASE, "unused-bridge"
                via_chat = AIDP_CHAT_URL
                spool = "/opt/relay/spool"
            elif agent == "codex" and args.gpt_via_tunnel:
                # AIDP via the laptop bridge: /v1/responses traffic is chat-bridged to the
                # pod-local tunnel; the tunnel must answer before the agent starts burning budget.
                upstream_base, api_key = OPENAI_BASE, "unused-tunnel"
                via_chat = TUNNEL_AIDP_URL
                wait_for_tunnel(pod)
            elif agent == "codex":
                upstream_base, api_key = OPENAI_BASE, OPENAI_KEY
                via_chat = None       # OpenAI direct (AIDP needs the bridge: 10.x-internal host)
            else:
                upstream_base, api_key = ANTHROPIC_BASE, ANTHROPIC_KEY
                via_chat = None
            relay_url = start_relay(pod, args.relay_port, upstream_base, api_key,
                                    force_model=force_model,
                                    responses_via_chat=via_chat, spool_dir=spool)
            container_env["ANTHROPIC_BASE_URL"] = relay_url
            container_env["ANTHROPIC_API_KEY"] = "relay"    # substituted upstream by the relay
            if agent == "codex":
                container_env["OPENAI_API_KEY"] = "relay"   # ditto — codex env_key placeholder
                write_codex_config(pod, relay_url, model)

            start_entry(pod, container_env)
        except BaseException:
            if not args.keep_pod and owns_pod:
                print("setup failed — terminating the pod so it does not bill idle", flush=True)
                terminate_pod(pod)
            raise
    print(f"running: pod {pod.id}\n  watch:  {workspace}/.agent/\n  budget: {budget_min} min",
          flush=True)

    import shutil

    # --reattach: the budget clock is the ORIGINAL run's, not this launcher's.
    t0 = started.timestamp() if args.reattach else time.time()
    # ... and the original records when its clock actually started (after setup), so a
    # reattaching launcher can be given exactly that instant (--started) rather than the
    # pre-provisioning time, which would cut the run a few minutes short.
    pj = run_dir / "pod.json"
    try:
        pod_rec = json.loads(pj.read_text())
        pod_rec.setdefault("budget_started", datetime.fromtimestamp(
            t0, tz=timezone.utc).isoformat(timespec="seconds"))
        pj.write_text(json.dumps(pod_rec, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"could not record budget_started in {pj}: {exc!r}", flush=True)

    def transcript_lines() -> int:
        n = 0
        for name in ("transcript.jsonl", "transcript.txt"):
            transcript = workspace / ".agent" / name
            if transcript.exists():
                n += sum(1 for _ in transcript.open(errors="replace"))
        return n

    auto_state = {"last": t0, "hash": None}

    def auto_submit() -> None:
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
        """Stop a run whose relay is dead or has stopped recording what it serves — the
        sandbox path's four-times-refined evidence rules, verbatim."""
        alive, served, recorded, logb = relay_pulse(pod)
        if not alive:
            print("the trajectory relay is no longer running — stopping the run rather than "
                  "spending a budget on something nobody can replay", flush=True)
            sh(pod, f"tail -20 {RELAY_DIR}/relay.log; ls -la {RELAY_DIR}/trajlog")
            stop_entry(pod)
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
        print("the relay serves 200s but records nothing and its log is flat — stopping "
              "the run rather than spending a budget on something nobody can replay", flush=True)
        sh(pod, f"tail -20 {RELAY_DIR}/relay.log; ls -la {RELAY_DIR}/trajlog")
        stop_entry(pod)
        raise SystemExit("run stopped: the relay is not recording what it serves")

    status, exit_code = "completed", None
    cycle_failures = 0
    while True:
        try:
            mirror_back(pod, run_dir)
            mirror_trajectory(pod, run_dir)
            require_trajectory()
            scan_submissions()
            still_running = running(pod)
            cycle_failures = 0
        except SystemExit:
            # the trajectory guard stopped the run: the pod must not keep billing behind an
            # exited launcher (it did before 2026-09-06 — only the 255-min reaper caught it)
            if owns_pod and not args.keep_pod:
                terminate_pod(pod)
            raise
        except Exception as exc:  # noqa: BLE001
            cycle_failures += 1
            print(f"poll cycle failed ({cycle_failures}/10, retrying in 60s): {exc!r}",
                  flush=True)
            if cycle_failures >= 10:
                status = "unreachable"
                print("the pod has been unreachable for 10 consecutive cycles — ending the run",
                      flush=True)
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
            print(f"budget reached — stopping pod entry {pod.id}", flush=True)
            stop_entry(pod)
            break
        time.sleep(60)

    try:
        scan_submissions()
        if auto_submit_min:
            auto_submit()
    except Exception as exc:  # noqa: BLE001
        print(f"final submission scan failed: {exc!r}", flush=True)

    try:
        mirror_back(pod, run_dir)
        mirror_trajectory(pod, run_dir)
        mirror_agent_state(pod, run_dir)   # last thing off the box: resumability
        rc, logs = sh(pod, f"cat {RELAY_DIR}/container.log 2>/dev/null", quiet=True)
        (run_dir / "container.log").write_text(logs)
    except Exception as exc:  # noqa: BLE001
        print(f"final mirror failed (artifacts are as of the last good cycle): {exc!r}",
              flush=True)

    if args.keep_pod or not owns_pod:
        print(f"pod {pod.id} left running (--keep-pod/--pod)", flush=True)
    else:
        terminate_pod(pod)

    record_out = {
        "exp": str(exp), "stage": stage.name, "preset": record["preset"],
        "set_states": record["set_states"],
        "control_mode_frozen": record["control_mode_frozen"],
        "condition": condition.as_record(),
        "task_files": task_files,
        "config": args.config, "agent": agent, "model": model,
        "image": POD_IMAGE, "gpu": GPU_TYPE, "budget_min": budget_min,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status, "container_exit_code": exit_code, "argv": sys.argv,
        "harness": "run_agent_runpod (runpod pod over ssh, no docker)",
        "pod_id": pod.id, "datacenter": DATACENTER, "reattached": bool(args.reattach),
        "upstream": (f"seed-code direct: {SEED_CODE_BASE} {force_model}" if args.via_seed_code
                     else "aidp via laptop bridge (spool)" if (agent == "codex" and args.gpt_via_bridge)
                     else "aidp via laptop tunnel" if (agent == "codex" and args.gpt_via_tunnel)
                     else OPENAI_BASE if agent == "codex" else ANTHROPIC_BASE),
        "force_model": force_model,
        "max_output_tokens": args.max_output_tokens,
    }
    (run_dir / "run.json").write_text(json.dumps(record_out, indent=2) + "\n")
    print(f"{status}: raw artifacts in {run_dir}  (workspace/, task/, container.log, run.json)")


if __name__ == "__main__":
    main()
