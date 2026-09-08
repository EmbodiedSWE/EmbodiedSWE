"""dgen_exec — run one data_engine Isaac invocation (generate/render) on Modal.

The Newton stack (isaacsim 6 + IsaacLab develop + Newton, built onto the
`cosigen-newton` volume by eval/scripts/modal_newton_env.py) does not exist on the
L20 pods that run the PhysX campaigns, so Newton tasks dispatch their GPU work
here: the orchestrator stays wherever the agent API is reachable and points
`--isaac-py` at scripts/modal_isaac_shim.py, which forwards every invocation to
this function through the `cosigen-dgen-io` volume.

    modal deploy data_engine/scripts/modal_dgen_app.py        # once per code change

Contract with the shim: the campaign lives on the io volume under its key; argv
arrives with all paths already rewritten to the remote layout; stdout/stderr of
the Isaac process stream back through the function log AND are returned with the
exit code — the shim propagates rc verbatim, so a failed remote batch fails the
orchestrator's invocation exactly like a local one (silently-swallowed remote
failures once let a repair agent "fix" a healthy solve on infra errors).

Concurrency: batches own disjoint batch dirs by construction, so concurrent
dgen_exec calls never write the same files. Derived metas (engine/meta.py) ARE
shared files — the remote run refreshes only its own container's view; the
conductor's local refresh after each wave is authoritative (POSIX locks do not
span Modal containers).
"""

from __future__ import annotations

import modal

REPO = "/repo"
NEWTON = "/newton"                       # env_newton + IsaacLab (built once)
IO = "/dgio"                             # campaign sync: <key>/ = one campaign
ENV_PY = f"{NEWTON}/env_newton/bin/python"

app = modal.App("cosigen-dgen")
newton_vol = modal.Volume.from_name("cosigen-newton")
io_vol = modal.Volume.from_name("cosigen-dgen-io", create_if_missing=True)

# the Newton runner userspace (mirrors eval/scripts/modal_newton_campaign.py, minus
# the agent CLIs — this function only runs Isaac work)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("curl", "git", "rsync", "libglvnd0", "libgl1", "libglx0", "libegl1",
                 "libgles2", "libvulkan1", "vulkan-tools", "libx11-6", "libxt6",
                 "libxrandr2", "libgomp1", "libglu1-mesa", "libsm6", "libice6",
                 "libxext6", "libxi6", "libxrender1", "libxfixes3", "libxcursor1",
                 "libxinerama1")
    .add_local_dir("robobench", remote_path=f"{REPO}/robobench")
    .add_local_dir("data_engine", remote_path=f"{REPO}/data_engine",
                   ignore=["**/__pycache__/**", "tests/**"])
    .add_local_file("pyproject.toml", remote_path=f"{REPO}/pyproject.toml")
)


def _write_vulkan_icd() -> None:
    import json
    import os
    for path, body in (
        ("/usr/share/vulkan/icd.d/nvidia_icd.json",
         {"file_format_version": "1.0.0",
          "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194"}}),
        ("/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
         {"file_format_version": "1.0.0",
          "ICD": {"library_path": "libEGL_nvidia.so.0"}}),
    ):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(body, f)
        except OSError:
            pass                          # driver-injected read-only mount: already provided


@app.function(image=image, gpu="L40S", timeout=4 * 60 * 60, retries=0,
              volumes={NEWTON: newton_vol, IO: io_vol}, max_containers=8)
def dgen_exec(script: str, argv: list[str]) -> dict:
    """Run `data_engine/scripts/<script> <argv...>` under env_newton; stream its
    output; commit the io volume; return {"returncode", "tail"}. argv paths must
    already be remote (/dgio/<key>/...) — the shim owns the rewrite."""
    import os
    import subprocess

    _write_vulkan_icd()
    newton_vol.reload()
    io_vol.reload()
    allowed = {"generate.py", "render.py"}
    if script not in allowed:
        raise ValueError(f"dgen_exec runs {sorted(allowed)}, not {script!r}")
    env = dict(
        os.environ,
        PYTHONPATH=f"{REPO}:{REPO}/data_engine",
        OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y",
        NVIDIA_DRIVER_CAPABILITIES="all",
        OMNI_KIT_CRASH_REPORTER="0", CARB_CRASHREPORTER_ENABLED="0",
        OMNI_KIT_ALLOW_ROOT="1",
    )
    cmd = [ENV_PY, f"{REPO}/data_engine/scripts/{script}", *argv]
    print(f"[dgen_exec] $ {' '.join(cmd)}", flush=True)
    tail: list[str] = []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)   # the full log, live in the function log
        tail.append(line)
        if len(tail) > 400:
            del tail[:200]
    rc = proc.wait()
    io_vol.commit()
    print(f"[dgen_exec] exit {rc}", flush=True)
    return {"returncode": rc, "tail": "".join(tail[-200:])}
