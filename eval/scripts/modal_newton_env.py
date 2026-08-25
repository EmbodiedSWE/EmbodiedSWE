#!/usr/bin/env python3
"""Build the `env_newton` stack (isaacsim 6 + IsaacLab develop + Newton, torch cu130) into a
Modal Volume, then boot-verify Isaac renders on an L4 GPU. This is the Modal-native analogue
of the RunPod `cosigen_env2.tar.zst` snapshot: built once, mounted read-write by every Newton
run container.

    modal run eval/scripts/modal_newton_env.py::build      # ~30-60 min, resumable
    modal run eval/scripts/modal_newton_env.py::verify      # boot isaacsim on L4

The recipe is the README's "Newton env" section, verbatim (pinned IsaacLab commit, the seven
editable packages), run inside a CUDA devel image so nvcc/build tools exist.
"""
from __future__ import annotations

import modal

REPO = "/repo"
NEWTON = "/newton"                         # the volume mount: holds env_newton + IsaacLab
ISAACLAB_COMMIT = "d7d004217c60b4790f721565bf5d40243addcb0e"  # README-pinned (develop 2026-06-17)

app = modal.App("cosigen-newton-env")
newton_vol = modal.Volume.from_name("cosigen-newton", create_if_missing=True)

# CUDA 13 devel base so the cu130 wheels + any source builds have a toolchain; add the GL/Vulkan
# userspace Isaac dlopens, git for the IsaacLab checkout, and uv for the install.
build_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "libglvnd0", "libgl1", "libglx0", "libegl1", "libgles2",
                 "libvulkan1", "vulkan-tools", "libx11-6", "libxt6", "libxrandr2",
                 "libgomp1", "libglu1-mesa",
                 # X11 client libs Kit dlopens at startup — MISSING these segfaults Kit on
                 # boot (root cause on Modal, measured 2026-08-15).
                 "libsm6", "libice6", "libxext6", "libxi6", "libxrender1", "libxfixes3",
                 "libxcursor1", "libxinerama1")
    .pip_install("uv")
    .add_local_dir("robobench", remote_path=f"{REPO}/robobench")
    .add_local_file("pyproject.toml", remote_path=f"{REPO}/pyproject.toml")
)


def _write_vulkan_icd() -> None:
    """Point the Vulkan loader at the NVIDIA driver lib Modal mounts (proven on L4: yields a
    DISCRETE_GPU / DRIVER_ID_NVIDIA_PROPRIETARY device instead of llvmpipe)."""
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


@app.function(image=build_image, volumes={NEWTON: newton_vol}, timeout=4 * 60 * 60,
              gpu="L4")
def build() -> str:
    import subprocess

    def run(cmd: str, **kw) -> None:
        print(f"\n$ {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True, **kw)

    src = f"{NEWTON}/IsaacLab/source"
    py = f"{NEWTON}/env_newton/bin/python"
    uv_env = "UV_INDEX_STRATEGY=unsafe-best-match "

    # 1. IsaacLab develop @ the pinned commit (consumed only as an editable source tree).
    run(f"test -d {NEWTON}/IsaacLab || git clone https://github.com/isaac-sim/IsaacLab.git "
        f"{NEWTON}/IsaacLab")
    run(f"git -C {NEWTON}/IsaacLab fetch --depth 1 origin {ISAACLAB_COMMIT} && "
        f"git -C {NEWTON}/IsaacLab checkout {ISAACLAB_COMMIT}")

    # 2. env_newton venv (python 3.12).
    run(f"test -x {py} || uv venv {NEWTON}/env_newton --python 3.12 --prompt env_newton")

    # 3. torch cu130, then isaacsim, then the seven editable isaaclab packages, then robobench.
    nv = ("--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match "
          "--prerelease=allow")
    run(f"{uv_env} uv pip install --python {py} torch==2.11.0 torchvision==0.26.0 "
        f"--index-url https://download.pytorch.org/whl/cu130")
    run(f"{uv_env} uv pip install --python {py} {nv} 'isaacsim[all,extscache]==6.0.0.1'")
    run(f"{uv_env} uv pip install --python {py} {nv} "
        f"-e '{src}/isaaclab_newton[all]' -e '{src}/isaaclab_physx[newton]' "
        f"-e '{src}/isaaclab_ovphysx' -e '{src}/isaaclab_visualizers[kit]' "
        f"-e '{src}/isaaclab_contrib' -e '{src}/isaaclab_assets' -e '{src}/isaaclab'")
    run(f"uv pip install --python {py} imageio imageio-ffmpeg")
    run(f"uv pip install --python {py} -e {REPO}")   # robobench (declares no other deps)

    newton_vol.commit()
    print("\nNEWTON_ENV_BUILT", flush=True)
    return "built"


@app.function(image=build_image, volumes={NEWTON: newton_vol}, timeout=30 * 60, gpu="L4")
def verify() -> str:
    import subprocess

    _write_vulkan_icd()
    py = f"{NEWTON}/env_newton/bin/python"
    code = (
        "from isaaclab.app import AppLauncher\n"
        "app = AppLauncher(headless=True).app\n"
        "print('APP_UP', flush=True)\n"
        "import torch, robobench\n"
        "print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)\n"
        "robobench.discover()\n"
        "from robobench.core.registries import ENVS\n"
        "names = [n for n in ENVS.list() if n.startswith(('folding','pouring'))]\n"
        "print('NEWTON_PRESETS', names, flush=True)\n"
        "print('BUILDING', flush=True)\n"
        "env = ENVS.get('folding.tshirt.franka.joint')().build(num_envs=1, seed=0)\n"
        "print('BUILT', flush=True)\n"
        "env.reset(seed=0)\n"
        "print('RESET_OK', flush=True)\n"
        "import os; os._exit(0)\n"
    )
    import os
    env = dict(os.environ,   # inherit LD_LIBRARY_PATH etc. (the driver mount) — required
               OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y",
               HOME="/root", NVIDIA_DRIVER_CAPABILITIES="all",
               PATH="/usr/bin:/bin:/usr/local/bin",
               # Isaac's breakpad crash-handler segfaults in Modal's restricted container
               # (cannot fork to write a dump); disabling it lets Kit boot.
               OMNI_KIT_CRASH_REPORTER="0", CARB_CRASHREPORTER_ENABLED="0",
               OMNI_KIT_ALLOW_ROOT="1")
    # A FILE, not `python -c`: isaacsim 6's AppLauncher parses sys.argv for Kit args, and a
    # bare "-c" argv[0] is misread as a Kit arg and segfaults Kit at startup (measured on L4,
    # 2026-08-15). Running a script file gives a clean argv[0].
    with open("/tmp/_verify_boot.py", "w") as f:
        f.write(code)
    r = subprocess.run([py, "/tmp/_verify_boot.py"], capture_output=True, text=True, env=env)
    print("STDOUT:\n", r.stdout, "\nSTDERR(tail):\n", r.stderr[-3000:], flush=True)
    return "verified" if "RESET_OK" in r.stdout else "FAILED"


@app.local_entrypoint()
def main(step: str = "build"):
    print({"build": build, "verify": verify}[step].remote())
