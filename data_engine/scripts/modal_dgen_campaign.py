"""Datagen campaigns on Modal: orchestrator + agent + Isaac in ONE container per task.

Everything runs on Modal (user directive 2026-09-01) — no pods, no laptop in any
loop: agent sessions call the seed-code gateway directly (publicly reachable,
Anthropic-native; robo_orange_o50 = Opus 5), Isaac runs on the container's L40S,
and the campaign mirrors to the `cosigen-dgen-out` volume every lap, so a lid
close or a dead container costs at most one mirror interval and a relaunch
RESUMES the same campaign (orchestrate is idempotent; its ledgers live in the
mirrored tree).

Two backends, one code path (`run_one`):
    physx    isaacsim 5.1 + isaaclab 2.3.2 baked into the image (the proven
             modal_grade_env/modal_physx_campaign stack)
    newton   isaacsim 6 + IsaacLab develop + Newton off the `cosigen-newton`
             volume (build once per workspace: modal_newton_env.py::build)

    modal run --detach data_engine/scripts/modal_dgen_campaign.py \\
        --tasks coffee,spatula [--newton-tasks tshirt] \\
        [--model model_hub/robo_orange_o50] [--budget flags ...]

Inputs come from the `cosigen-dgen-runs` volume: /inputs/<task>/{run.json,
workspace/solution[,workspace/candidates/<k>]} — eval-run-shaped, staged by the
launcher. Outputs mirror to cosigen-dgen-out:/<task>/data_gen/<gen name>/.
"""

from __future__ import annotations

import os

import modal

REPO = "/repo"
PHYSX_VENV = "/opt/venv"
NEWTON = "/newton"
NEWTON_PY = f"{NEWTON}/env_newton/bin/python"
IN_VOL, OUT_VOL = "/dgrun", "/dgout"
NODE_BIN = "/opt/node-v22.11.0-linux-x64/bin"

app = modal.App("cosigen-dgen-campaign")
# Campaign APPS run in the isolated `datagen` environment (workspace coordination:
# a broad `modal app stop` from another project's session must not be able to reach
# a half-finished campaign — see MODAL_WORKSPACE.md). The volumes stay in `main`,
# looked up cross-environment; app stops never touch volumes.
_VOL_ENV = os.environ.get("DGEN_VOLUME_ENV") or None
in_vol = modal.Volume.from_name("cosigen-dgen-runs", environment_name=_VOL_ENV)
out_vol = modal.Volume.from_name("cosigen-dgen-out", create_if_missing=True,
                                 environment_name=_VOL_ENV)
newton_vol = modal.Volume.from_name("cosigen-newton", create_if_missing=True,
                                    environment_name=_VOL_ENV)

_GL_APT = ("build-essential", "ca-certificates", "curl", "git", "jq", "rsync",
           "libglvnd0", "libgl1", "libglx0", "libegl1", "libgles2", "libvulkan1",
           "vulkan-tools", "libx11-6", "libxt6", "libxrandr2", "libgomp1",
           "libglu1-mesa", "libsm6", "libice6", "libxext6", "libxi6",
           "libxrender1", "libxfixes3", "libxcursor1", "libxinerama1")
_NODE_CMDS = (
    "curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.gz "
    "| tar xz -C /opt",
    f"PATH={NODE_BIN}:$PATH {NODE_BIN}/npm install -g "
    "@anthropic-ai/claude-code@2.1.216",
)


def _repo_dirs(image: modal.Image) -> modal.Image:
    return (image
            .add_local_dir("robobench", remote_path=f"{REPO}/robobench")
            .add_local_dir("data_engine", remote_path=f"{REPO}/data_engine",
                           ignore=["**/__pycache__/**", "tests/**"])
            .add_local_file("pyproject.toml", remote_path=f"{REPO}/pyproject.toml"))


physx_image = _repo_dirs(
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu24.04",
                              add_python="3.11")
    .apt_install(*_GL_APT)
    .pip_install("uv")
    .run_commands(
        f"uv venv --python 3.11 {PHYSX_VENV}",
        f"uv pip install --python {PHYSX_VENV}/bin/python torch==2.7.0 "
        f"--index-url https://download.pytorch.org/whl/cu128",
        f"uv pip install --python {PHYSX_VENV}/bin/python "
        f"'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com",
        f"uv pip install --python {PHYSX_VENV}/bin/python setuptools wheel",
        f"CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install --python "
        f"{PHYSX_VENV}/bin/python 'isaaclab[all]==2.3.2' "
        f"--extra-index-url https://pypi.nvidia.com "
        f"--no-build-isolation-package flatdict",
        f"uv pip install --python {PHYSX_VENV}/bin/python imageio imageio-ffmpeg pyyaml",
        *_NODE_CMDS,
    )
    .pip_install("pyyaml")
)

newton_image = _repo_dirs(
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install(*_GL_APT)
    .run_commands(*_NODE_CMDS)
    .pip_install("pyyaml")
)


def _icd_env() -> dict:
    """Vulkan/EGL loader files in a writable place + the env that points at them
    (driver-injected system dirs are read-only on some shapes; measured 2026-08-28)."""
    import json
    import os

    overrides = {}
    for var, path, body in (
        ("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json",
         {"file_format_version": "1.0.0",
          "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194"}}),
        ("__EGL_VENDOR_LIBRARY_FILENAMES",
         "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
         {"file_format_version": "1.0.0",
          "ICD": {"library_path": "libEGL_nvidia.so.0"}}),
    ):
        if os.path.exists(path):
            continue
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(body, f)
        except OSError:
            alt = f"/tmp/{os.path.basename(path)}"
            with open(alt, "w") as f:
                json.dump(body, f)
            overrides[var] = alt
    return overrides


def _run_one(task: str, backend: str, model: str, args: dict) -> str:
    """The shared campaign body — stage input, resume mirror, run orchestrate,
    mirror every lap, return the terminal status line."""
    import json
    import os
    import shutil
    import subprocess
    import threading
    from pathlib import Path

    in_vol.reload()
    out_vol.reload()
    if backend == "newton":
        newton_vol.reload()
        isaac_py = NEWTON_PY
        if not Path(isaac_py).is_file():
            raise RuntimeError("env_newton missing on the cosigen-newton volume — "
                               "run modal_newton_env.py::build in this workspace")
    else:
        isaac_py = f"{PHYSX_VENV}/bin/python"

    src = Path(IN_VOL) / "inputs" / task
    if not (src / "run.json").is_file():
        raise RuntimeError(f"no staged input for {task} on cosigen-dgen-runs")
    run_dir = Path("/work") / task
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{src}/", f"{run_dir}/"], check=True)

    # RESUME: a previous container's mirror is this campaign's durable state
    mirror_root = Path(OUT_VOL) / task
    prior = mirror_root / "data_gen"
    if prior.is_dir():
        print(f"[dgen {task}] resuming from mirrored campaign state", flush=True)
        subprocess.run(["rsync", "-a", f"{prior}/", f"{run_dir}/data_gen/"],
                       check=True)

    stop = threading.Event()

    def mirror() -> None:
        src_dg = run_dir / "data_gen"
        if src_dg.is_dir():
            mirror_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--delete", f"{src_dg}/",
                            f"{mirror_root}/data_gen/"], check=False)
            out_vol.commit()

    def mirror_loop() -> None:
        while not stop.wait(180):
            mirror()

    threading.Thread(target=mirror_loop, daemon=True).start()

    env = dict(
        os.environ,
        PATH=f"{NODE_BIN}:{os.environ.get('PATH', '')}",
        DGEN_AGENT_MODEL=model,
        DGEN_CODE_HASH=os.environ.get("DGEN_CODE_HASH", "modal-image"),
        OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y",
        NVIDIA_DRIVER_CAPABILITIES="all",
        OMNI_KIT_CRASH_REPORTER="0", CARB_CRASHREPORTER_ENABLED="0",
        OMNI_KIT_ALLOW_ROOT="1",
        **_icd_env(),
    )
    gen_name = args.get("--name", "gen_v1")
    cmd = [
        "python3", f"{REPO}/data_engine/scripts/orchestrate.py", str(run_dir),
        "--isaac-py", isaac_py,
        "--agent-cmd",
        f"python3 {REPO}/data_engine/scripts/claude_agent_loop.py --model {model}",
    ]
    for flag, value in args.items():
        cmd += [flag, str(value)]
    print(f"[dgen {task}] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(list(map(str, cmd)), env=env).returncode
    stop.set()
    mirror()
    status = {}
    status_p = run_dir / "data_gen" / gen_name / "status.json"
    if status_p.is_file():
        status = json.loads(status_p.read_text())
    shutil.rmtree(run_dir, ignore_errors=True)
    line = (f"{task}: rc={rc} stage={status.get('stage')} "
            f"ladder={status.get('ladder')} "
            f"physics={status.get('physics_set')}/{status.get('physics_target')} "
            f"samples={status.get('samples_done')}/{status.get('samples_target')}")
    print(f"[dgen {task}] {line}", flush=True)
    return line


COMMON = dict(timeout=23 * 60 * 60, retries=0, cpu=16, memory=65536,
              volumes={IN_VOL: in_vol, OUT_VOL: out_vol, NEWTON: newton_vol})
# Fallback order: Modal's L40S pool went unschedulable for 15+ min stretches on
# 2026-09-01; the render path peaks ~15.5 GB so every shape here fits.
GPUS = ["L40S", "A100-40GB", "L4"]


@app.function(image=physx_image, gpu=GPUS, max_containers=11, **COMMON)
def run_physx(task: str, model: str, args: dict) -> str:
    return _run_one(task, "physx", model, args)


@app.function(image=newton_image, gpu=GPUS, max_containers=2, **COMMON)
def run_newton(task: str, model: str, args: dict) -> str:
    return _run_one(task, "newton", model, args)


@app.local_entrypoint()
def main(tasks: str = "", newton_tasks: str = "",
         model: str = "model_hub/robo_orange_o50",
         physics_target: int = 200, visual_draws: int = 3,
         num_envs: int = 512, newton_num_envs: int = 4,
         stage_hours: float = 12.0,
         sessions: str = "scene,strategy,phase",
         job_concurrency: int = 1, newton_job_concurrency: int = 4,
         render_envs: int = 16, session_min: float = 120.0,
         gen_name: str = "gen_v1") -> None:
    physx = [t for t in tasks.split(",") if t]
    newton = [t for t in newton_tasks.split(",") if t]
    if not physx and not newton:
        raise SystemExit("pass --tasks and/or --newton-tasks")

    def args_for(nenvs: int, conc: int) -> dict:
        return {"--physics-target": physics_target,
                "--visual-draws": visual_draws, "--num-envs": max(1, int(nenvs)),
                "--stage-hours": stage_hours,
                "--sessions": sessions,
                "--job-concurrency": conc, "--render-envs": render_envs,
                "--intervention-session-min": session_min, "--name": gen_name}

    calls = [run_physx.spawn(t, model, args_for(num_envs, job_concurrency))
             for t in physx]
    calls += [run_newton.spawn(t, model,
                               args_for(newton_num_envs, newton_job_concurrency))
              for t in newton]
    print(f"launched {len(calls)} datagen campaigns (model {model})", flush=True)
    for c in calls:
        print(c.get(), flush=True)
