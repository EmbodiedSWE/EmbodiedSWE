#!/usr/bin/env bash

# Reproducible CoSiGen PhysX environment for Isaac Sim 5.1 / Isaac Lab 2.3.2.
#
# This deliberately uses uv for the large binary stack and pip for the two
# packages whose current wheels/sdists uv 0.12 rejects (flatdict and Warp).
# It is safe to re-run: valid pinned flatdict/Warp installs are retained.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${COSIGEN_VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
ISAACSIM_VERSION="5.1.0"
ISAACLAB_VERSION="2.3.2"
FLATDICT_VERSION="4.0.1"
WARP_VERSION="1.11.0"
CLICK_VERSION="8.1.7"
TYPING_EXTENSIONS_VERSION="4.12.2"

fail() {
    echo "[bootstrap] ERROR: $*" >&2
    exit 1
}

stage() {
    echo
    echo "[bootstrap] $*"
}

trap 'echo "[bootstrap] FAILED at line $LINENO" >&2' ERR

[[ "$(uname -s)" == "Linux" ]] || fail "Isaac Sim 5.1 requires Linux."
command -v uv >/dev/null 2>&1 || fail "uv is not installed or not on PATH."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available."
nvidia-smi >/dev/null || fail "the NVIDIA driver is not responding."

cd "$REPO_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
    stage "Creating Python 3.11 environment at $VENV_DIR"
    uv venv "$VENV_DIR" --python 3.11 --prompt cosigen
fi

PYTHON_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_MINOR" == "3.11" ]] || fail "$PYTHON_BIN is Python $PYTHON_MINOR; expected 3.11."

SITE_PACKAGES="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
RECOVERY_DIR="$VENV_DIR/.bootstrap-recovery/$(date +%Y%m%d-%H%M%S)-$$"

dist_ok() {
    local package="$1"
    local expected="$2"
    "$PYTHON_BIN" - "$package" "$expected" <<'PY' >/dev/null 2>&1
import importlib.metadata as md
import sys

name, expected = sys.argv[1:]
meta = md.metadata(name)
actual_name = (meta.get("Name") or "").lower().replace("_", "-")
expected_name = name.lower().replace("_", "-")
raise SystemExit(0 if actual_name == expected_name and md.version(name) == expected else 1)
PY
}

quarantine() {
    local label="$1"
    shift
    local moved=0
    local candidate
    mkdir -p "$RECOVERY_DIR"
    for candidate in "$@"; do
        [[ -e "$candidate" ]] || continue
        mv "$candidate" "$RECOVERY_DIR/"
        moved=1
    done
    if (( moved )); then
        echo "[bootstrap] quarantined stale $label files under $RECOVERY_DIR"
    fi
}

# uv reads every installed dist-info directory before resolving. Move only stale
# or malformed entries out of the environment before the first uv invocation.
if ! dist_ok flatdict "$FLATDICT_VERSION"; then
    quarantine flatdict \
        "$SITE_PACKAGES/flatdict.py" \
        "$SITE_PACKAGES"/flatdict-*.dist-info
fi
if ! dist_ok warp-lang "$WARP_VERSION"; then
    quarantine warp-lang \
        "$SITE_PACKAGES/warp" \
        "$SITE_PACKAGES"/warp_lang-*.dist-info
fi

stage "Installing PyTorch CUDA 12.8 wheels"
uv pip install --python "$PYTHON_BIN" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url https://download.pytorch.org/whl/cu128

stage "Installing Isaac Sim $ISAACSIM_VERSION"
uv pip install --python "$PYTHON_BIN" \
    "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com

stage "Pinning legacy-package build tools"
uv pip install --python "$PYTHON_BIN" --reinstall --no-cache \
    "pip==25.2" \
    "setuptools==81.0.0"

if ! dist_ok flatdict "$FLATDICT_VERSION"; then
    stage "Building flatdict $FLATDICT_VERSION with compatible setuptools"
    uv cache clean flatdict >/dev/null
    "$PYTHON_BIN" -m pip install \
        --no-cache-dir \
        --no-build-isolation \
        --force-reinstall \
        --no-deps \
        "flatdict==$FLATDICT_VERSION"
fi

if ! dist_ok warp-lang "$WARP_VERSION"; then
    stage "Installing release-matched Warp $WARP_VERSION with pip"
    uv cache clean warp-lang >/dev/null
    "$PYTHON_BIN" -m pip install \
        --no-cache-dir \
        --force-reinstall \
        --no-deps \
        "warp-lang==$WARP_VERSION"
fi

stage "Installing Isaac Lab $ISAACLAB_VERSION"
CMAKE_POLICY_VERSION_MINIMUM=3.5 "$PYTHON_BIN" -m pip install \
    --no-cache-dir \
    "flatdict==$FLATDICT_VERSION" \
    "warp-lang==$WARP_VERSION" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    "click==$CLICK_VERSION" \
    "typing-extensions==$TYPING_EXTENSIONS_VERSION" \
    "isaaclab[all]==$ISAACLAB_VERSION" \
    --extra-index-url https://pypi.nvidia.com

stage "Installing CoSiGen editable"
uv pip install --python "$PYTHON_BIN" -e "$REPO_ROOT"

stage "Checking the resolved environment"
"$PYTHON_BIN" -m pip check
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | sed 's/^/[bootstrap] gpu_driver=/'
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import torch

for package in (
    "torch", "torchvision", "torchaudio", "isaacsim", "isaaclab",
    "flatdict", "warp-lang", "click", "typing-extensions",
):
    print(f"[bootstrap] {package}={md.version(package)}")
print(f"[bootstrap] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[bootstrap] gpu={torch.cuda.get_device_name(0)}")
PY

# Isaac Sim's first import shows an interactive EULA prompt; accept it for this check (documented in the README).
OMNI_KIT_ACCEPT_EULA=YES "$PYTHON_BIN" -c 'import isaaclab, warp; print(f"[bootstrap] imports OK; warp={warp.__version__}")'

echo
echo "[bootstrap] COMPLETE"
echo "[bootstrap] activate with: source $VENV_DIR/bin/activate"
echo "[bootstrap] validate with: ./scripts/validate_egg_carton.sh"
