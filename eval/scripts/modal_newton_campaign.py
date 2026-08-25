#!/usr/bin/env python3
"""Newton-suite campaign on Modal: agent + relay + Isaac (isaacsim 6 / IsaacLab develop /
Newton) all in ONE L4 container per task×model. This is the Modal-native analogue of
run_agent_runpod.py — same agent-entry.sh, same relay, same /bench|/task|/workspace layout and
the same budget/mirror loop — but with the RunPod-over-SSH substrate replaced by local
subprocess + a mounted volume, because Newton must run on Modal (user directive 2026-08-15).

Prereqs (once): `modal run eval/scripts/modal_newton_env.py::build` populates the
`cosigen-newton` volume with env_newton + IsaacLab, and `::build_worlds` (below) builds the
two experiment worlds into it.

    modal run eval/scripts/modal_newton_campaign.py::build_worlds     # build tshirt+latte worlds
    modal run eval/scripts/modal_newton_campaign.py                   # all 10 Newton pairs
    modal run eval/scripts/modal_newton_campaign.py --only tshirt:claude-opus-5 --budget-min 8
"""
from __future__ import annotations

import modal

REPO = "/repo"
NEWTON = "/newton"                        # env_newton + IsaacLab + experiments (built once)
RUNS = "/runs"                            # durable per-run artifacts
ENV_PY = f"{NEWTON}/env_newton/bin/python"
NODE_BIN = "/opt/node-v22.11.0-linux-x64/bin"
AGENT_USER = "agent"
BUDGET_MIN_DEFAULT = 240.0
# OpenAI DIRECT for codex (user directive 2026-08-15): the AIDP gateway resolves to a
# ByteDance-internal 10.x address and is unreachable from cloud VMs (measured from Modal:
# ConnectTimeout).
OPENAI_KEY = ("sk-proj-xV_ukdY524Uxtrz0RPANBXfZIX9Azw2J5Nj5XJlgwqu8_KgEjTz3AxF_sHvMnomUtwErbzDw6"
              "vT3BlbkFJVTC9MwXXYQXjpBx09iDDkkA8BvPCnDi-il9DQFr7bQvB9Q6yXydAotr40G99VWfOd8rWqMlmUA")
ANTHROPIC_KEY = ("sk-ant-api03-TiY0vWZvGI5IjicOJGaYRD-DkzLq6lH1ihCSBN4cJCShhEfGWXPvnGwSNOdoQPb"
                 "_useOr9y8jvn2QgKn1u34EQ-rXpLcQAA")

TASKS = {  # experiment name -> (stage spec, built dir on the newton volume)
    "tshirt": ("folding.tshirt:franka:joint", f"{NEWTON}/experiments/tshirt_notools"),
    "latte": ("pouring.latte:bimanual_franka:joint", f"{NEWTON}/experiments/latte_notools"),
}
MODELS = {
    "claude-opus-5": "claude", "claude-opus-4-8": "claude", "claude-fable-5": "claude",
    "gpt-5.6-sol": "codex", "gpt-5.6-terra": "codex",
}


def _short(model: str) -> str:
    return model.replace("claude-", "").replace(".", "_").replace("-", "_")


app = modal.App("cosigen-newton-campaign")
newton_vol = modal.Volume.from_name("cosigen-newton")
runs_vol = modal.Volume.from_name("cosigen-newton-runs", create_if_missing=True)

# Runner image: CUDA13 devel (matches the env's cu130), the GL/Vulkan userspace Isaac dlopens,
# node + the pinned agent CLIs, the relay's python deps, and the repo (harness + relay +
# robobench — the same /repo path env_newton installed robobench editable against).
image = (
    # 3.12 MUST match the env-build image: env_newton's venv symlinks its interpreter to
    # /usr/local/bin/python3.12, so a different base python leaves that symlink dangling.
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("curl", "libglvnd0", "libgl1", "libglx0", "libegl1", "libgles2",
                 "libvulkan1", "vulkan-tools", "libx11-6", "libxt6", "libxrandr2",
                 "libgomp1", "libglu1-mesa", "libsm6", "libice6", "git", "rsync",
                 # X11 client libs Kit dlopens at startup — MISSING these segfaults Kit on
                 # boot (root cause on Modal, measured 2026-08-15).
                 "libxext6", "libxi6", "libxrender1", "libxfixes3", "libxcursor1",
                 "libxinerama1")
    .pip_install("fastapi", "uvicorn", "httpx", "pyyaml")
    .run_commands(
        "curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.gz "
        "| tar xz -C /opt",
        f"ln -sf {NODE_BIN}/node /usr/local/bin/node && "
        f"ln -sf {NODE_BIN}/npm /usr/local/bin/npm && "
        f"ln -sf {NODE_BIN}/npx /usr/local/bin/npx",
        # node must be on PATH here: npm's shebang is `#!/usr/bin/env node`.
        f"PATH={NODE_BIN}:$PATH {NODE_BIN}/npm install -g "
        "@anthropic-ai/claude-code@2.1.216 @openai/codex@0.147.0",
    )
    .add_local_dir("robobench", remote_path=f"{REPO}/robobench")
    .add_local_dir("eval", remote_path=f"{REPO}/eval")
    .add_local_dir("sim_gen/super_relay", remote_path=f"{REPO}/sim_gen/super_relay",
                   ignore=["logs/**", "examples/**"])
    .add_local_file("pyproject.toml", remote_path=f"{REPO}/pyproject.toml")
)


def _write_vulkan_icd() -> None:
    import json
    import os
    os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
    with open("/usr/share/vulkan/icd.d/nvidia_icd.json", "w") as f:
        json.dump({"file_format_version": "1.0.0",
                   "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194"}}, f)
    os.makedirs("/usr/share/glvnd/egl_vendor.d", exist_ok=True)
    with open("/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "w") as f:
        json.dump({"file_format_version": "1.0.0",
                   "ICD": {"library_path": "libEGL_nvidia.so.0"}}, f)


@app.function(image=image, volumes={NEWTON: newton_vol}, gpu="L4", timeout=2 * 60 * 60)
def build_worlds(only: str = "") -> str:
    """Build the tshirt/latte no_tools worlds into the newton volume (boot-validated on L4)."""
    import os
    import subprocess

    _write_vulkan_icd()
    newton_vol.reload()   # see env_newton committed by the modal_newton_env app
    names = [only] if only else list(TASKS)
    out_root = f"{NEWTON}/experiments"
    os.makedirs(out_root, exist_ok=True)
    results = []
    for name in names:
        stage, built = TASKS[name]
        if os.path.isdir(built):
            results.append(f"{name}: already built")
            continue
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y",
                   PRIVACY_CONSENT="Y", NVIDIA_DRIVER_CAPABILITIES="all",
                   OMNI_KIT_CRASH_REPORTER="0", CARB_CRASHREPORTER_ENABLED="0",
                   OMNI_KIT_ALLOW_ROOT="1", PYTHONPATH=f"{REPO}/eval")
        # build_env.py re-execs under .venv if present; there is none here, so it runs under
        # the interpreter we give it — env_newton, which has isaacsim 6 + Newton.
        r = subprocess.run(
            [ENV_PY, f"{REPO}/eval/scripts/build_env.py", "--name", f"{name}_notools",
             "--stage", stage, "--config", "no_tools", "--out", out_root],
            env=env, capture_output=True, text=True)
        print(f"=== {name} build stdout tail:\n{r.stdout[-3000:]}", flush=True)
        if r.returncode != 0:
            print(f"=== {name} build stderr tail:\n{r.stderr[-3000:]}", flush=True)
            results.append(f"{name}: BUILD FAILED rc={r.returncode}")
        else:
            results.append(f"{name}: built")
    newton_vol.commit()
    return " | ".join(results)


@app.function(image=image, volumes={NEWTON: newton_vol, RUNS: runs_vol}, gpu="L4",
              timeout=6 * 60 * 60, retries=0, max_containers=12)
def run_one(task: str, model: str, budget_min: float = BUDGET_MIN_DEFAULT,
            attempt: str = "") -> str:
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    sys.path.insert(0, f"{REPO}/eval")
    from envbuild import prompts
    from envbuild.condition import load as load_condition

    _write_vulkan_icd()
    newton_vol.reload()   # see env_newton + built worlds committed by other containers
    agent = MODELS[model]
    stage_spec, built = TASKS[task]
    exp = Path(built)
    suffix = f"_r{attempt}" if attempt else ""
    label = f"{task}_{_short(model)}{suffix}"

    # --- Local /bench, /task, /workspace, /submissions, /opt/relay (was: SSH + tar) ---
    receipt = json.loads((exp / "resolved.json").read_text())
    srec = next(r for r in receipt["stages"])
    stage_dir = exp / "stages" / srec["dir"]
    preset = srec["preset"]
    condition = load_condition("no_tools")
    facts = {"set_states": receipt.get("set_states", True),
             "control_mode_frozen": srec.get("control_mode_frozen", False)}

    for d in ("/bench", "/task", "/workspace/solution", "/workspace/.agent",
              "/workspace/tmp", "/submissions", "/ovcache", "/opt/relay/trajlog"):
        os.makedirs(d, exist_ok=True)
    subprocess.run(f"cp -a {stage_dir}/bench/. /bench/", shell=True, check=True)

    task_dir = prompts.render_task_dir(
        Path("/task"), scene=preset.split(".")[1], preset=preset,
        describe_text=(stage_dir / "describe.md").read_text(),
        facts=facts, condition=condition, budget_min=budget_min)
    task_files = {str(p.relative_to(task_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(task_dir.rglob("*")) if p.is_file()}

    # non-root agent user (claude/codex refuse root); it owns only the writable dirs
    subprocess.run(f"id -u {AGENT_USER} >/dev/null 2>&1 || useradd -m -s /bin/bash {AGENT_USER}",
                   shell=True, check=True)
    subprocess.run(f"chown -R {AGENT_USER}:{AGENT_USER} /workspace /submissions /ovcache",
                   shell=True, check=True)
    subprocess.run("chown -R root:root /bench /task && chmod -R a+rX,go-w /bench /task",
                   shell=True, check=True)

    # --- relay (local subprocess; codex bridged to the AIDP chat gateway) ---
    relay_src = Path(f"{REPO}/sim_gen/super_relay")
    for p in relay_src.glob("*.py"):
        shutil.copy(p, Path("/opt/relay") / p.name)
    relay_env = dict(os.environ)
    port = 8118
    if agent == "codex":
        upstream_base = "https://api.openai.com/v1"
        relay_env["SUPER_RELAY_API_KEY"] = OPENAI_KEY
    else:
        upstream_base = "https://api.anthropic.com/v1"
        relay_env["SUPER_RELAY_API_KEY"] = ANTHROPIC_KEY
    relay_env["SUPER_RELAY_FORCE_MODEL"] = model
    relay_proc = subprocess.Popen(
        [sys.executable, "/opt/relay/server.py", "--host", "127.0.0.1", "--port", str(port),
         "--log-dir", "/opt/relay/trajlog", "--upstream-base", upstream_base],
        env=relay_env, stdout=open("/opt/relay/relay.log", "w"), stderr=subprocess.STDOUT)
    relay_url = f"http://127.0.0.1:{port}"
    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            if b'"ok"' in urllib.request.urlopen(f"{relay_url}/health", timeout=5).read():
                break
        except Exception:
            continue
    else:
        raise SystemExit("relay did not come up")

    # --- entry script + success check installed exactly like the pod path ---
    entry = Path(f"{REPO}/eval/docker/entrypoints")
    os.makedirs("/opt/entrypoints", exist_ok=True)
    shutil.copy(entry / "agent-entry.sh", "/opt/entrypoints/agent-entry.sh")
    shutil.copy(entry / "submit", "/usr/local/bin/submit")
    os.chmod("/opt/entrypoints/agent-entry.sh", 0o755)
    os.chmod("/usr/local/bin/submit", 0o755)
    os.makedirs("/opt/harness", exist_ok=True)
    shutil.copy(f"{REPO}/eval/scripts/verify_solution.py", "/opt/verify_solution.py")

    if agent == "codex":
        os.makedirs(f"/home/{AGENT_USER}/.codex", exist_ok=True)
        Path(f"/home/{AGENT_USER}/.codex/config.toml").write_text(
            f'model = "{model}"\nmodel_provider = "relay"\n\n'
            f'[model_providers.relay]\nname = "relay"\nbase_url = "{relay_url}/v1"\n'
            f'env_key = "OPENAI_API_KEY"\nwire_api = "responses"\n')
        subprocess.run(f"chown -R {AGENT_USER}:{AGENT_USER} /home/{AGENT_USER}/.codex",
                       shell=True, check=True)

    container_env = dict(
        os.environ,
        AGENT=agent, MODEL=model, KEEP_GOING="1",
        PYTHONPATH="/bench:/task/tools",
        XDG_CACHE_HOME="/ovcache", TMPDIR="/workspace/tmp",
        PATH=f"{NEWTON}/env_newton/bin:{NODE_BIN}:/opt/npm/bin:/usr/local/sbin:"
             f"/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y",
        NVIDIA_DRIVER_CAPABILITIES="all",
        # Isaac's breakpad handler segfaults in the container; disable it so the agent's sim
        # scripts and the success check can boot Kit.
        OMNI_KIT_CRASH_REPORTER="0", CARB_CRASHREPORTER_ENABLED="0", OMNI_KIT_ALLOW_ROOT="1",
        ANTHROPIC_BASE_URL=relay_url, ANTHROPIC_API_KEY="relay", OPENAI_API_KEY="relay",
        SUCCESS_CHECK=(f"timeout -k 30 3900 {ENV_PY} /opt/verify_solution.py "
                       f"--preset {preset} --solution /workspace/solution"),
    )

    started = datetime.now(timezone.utc)
    entry_proc = subprocess.Popen(
        ["/opt/entrypoints/agent-entry.sh"], cwd="/", env=container_env,
        stdout=open("/opt/relay/container.log", "w"), stderr=subprocess.STDOUT)

    run_dir = Path(RUNS) / label
    (run_dir / "opt" / "relay" / "trajlog").mkdir(parents=True, exist_ok=True)

    def mirror() -> None:
        for sub in ("workspace", "submissions"):
            src = Path("/") / sub
            if src.is_dir():
                subprocess.run(f"rsync -a --delete /{sub}/ {run_dir}/{sub}/", shell=True)
        traj = Path("/opt/relay/trajlog/raw_requests.jsonl")
        if traj.exists():
            shutil.copy(traj, run_dir / "opt" / "relay" / "trajlog" / "raw_requests.jsonl")
        if Path("/opt/relay/container.log").exists():
            shutil.copy("/opt/relay/container.log", run_dir / "container.log")
        if Path("/opt/relay/relay.log").exists():
            shutil.copy("/opt/relay/relay.log", run_dir / "relay.log")
        runs_vol.commit()

    status = "completed"
    t0 = time.time()
    while True:
        if entry_proc.poll() is not None:
            status = "completed"       # entry loop exited on its own (success check passed)
            break
        if time.time() - t0 > budget_min * 60:
            status = "timeout"
            subprocess.run("pkill -TERM -f '[a]gent-entry'; pkill -TERM -f '[c]laude'; "
                           "pkill -TERM -f '[c]odex'; sleep 20; pkill -KILL -f '[a]gent-entry'; "
                           "pkill -KILL -f '[c]laude'; pkill -KILL -f '[c]odex'; true",
                           shell=True)
            break
        if relay_proc.poll() is not None:
            status = "relay_died"
            break
        time.sleep(60)
        mirror()

    # agent session state (resumability) + final mirror
    subprocess.run(f"tar czf {run_dir}/agent_home.tgz -C /home {AGENT_USER} 2>/dev/null; true",
                   shell=True)
    mirror()
    try:
        relay_proc.terminate()
    except Exception as exc:
        print(f"relay terminate: {exc}", flush=True)

    (run_dir / "run.json").write_text(json.dumps({
        "exp": built, "stage": srec["dir"], "preset": preset,
        "set_states": facts["set_states"], "control_mode_frozen": facts["control_mode_frozen"],
        "condition": condition.as_record(), "task_files": task_files,
        "config": "no_tools", "agent": agent, "model": model,
        "gpu": "NVIDIA L4", "budget_min": budget_min,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "harness": "modal_newton_campaign (in-container, no docker/ssh)",
        "upstream": "https://api.openai.com/v1" if agent == "codex"
        else "https://api.anthropic.com/v1",
    }, indent=2) + "\n")
    runs_vol.commit()
    return f"{label}: status={status}"


@app.local_entrypoint()
def main(only: str = "", budget_min: float = BUDGET_MIN_DEFAULT, attempt: str = ""):
    grid: list[tuple[str, str]] = []
    if only:
        for pair in only.split(","):
            t, _, m = pair.partition(":")
            if t not in TASKS or m not in MODELS:
                raise SystemExit(f"unknown pair {pair!r}")
            grid.append((t, m))
    else:
        grid = [(t, m) for t in TASKS for m in MODELS]
    print(f"launching {len(grid)} Newton runs, budget {budget_min} min each", flush=True)
    calls = [run_one.spawn(t, m, budget_min, attempt) for t, m in grid]
    for c in calls:
        print(c.get(), flush=True)
