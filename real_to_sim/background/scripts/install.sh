#!/usr/bin/env bash
# One-time setup for the background stage.
#
# Creates this stage's OWN env at real_to_sim/background/.venv (uv-managed).
# Rationale: every real_to_sim stage gets a dedicated venv — the splat stack
# (torch 2.8+cu128, compiled CUDA kernels) conflicts with isaacsim's pins, and
# future stages (VLA training, object generation) will bring their own.
#
# 3dgrut is vendored as a pinned git submodule (like sim_gen/RoboVerse); its
# installer hardcodes `.venv` inside its own dir, so we satisfy it with a
# symlink 3dgrut/.venv -> ../.venv and the build lands in the stage venv.
#
#   Usage: bash setup.sh
#
# Requirements: uv, NVIDIA GPU + driver, gcc <= 14 available (CUDA 12.8),
# ffmpeg, wget, ~10 GB disk. Known-OS gotchas: docs/RUNBOOK.md.
set -euo pipefail
BG="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$BG/../.." && pwd)"
VENV="$BG/.venv"
GRUT="$BG/3dgrut"

echo "[1/5] init 3dgrut submodule (pinned)..."
git -C "$REPO" submodule update --init --recursive real_to_sim/background/3dgrut

echo "[2/5] create stage venv at $VENV ..."
if [ ! -f "$VENV/bin/activate" ]; then
    uv venv "$VENV" --python 3.11 --prompt r2s-background
fi
ln -sfn ../.venv "$GRUT/.venv"   # satisfy 3dgrut's hardcoded .venv path

echo "[3/5] CUDA toolkit into the venv (~4 GB download, cached in /tmp)..."
cd "$GRUT"
# new distros dropped libxml2.so.2, which the CUDA runfile installer needs
if ! ldconfig -p | grep -q libxml2.so.2; then
    XML16=$(ldconfig -p | grep -oE "/[^ ]*libxml2.so.16" | head -1 || true)
    if [ -n "$XML16" ]; then
        mkdir -p "$BG/.setup_shim" && ln -sf "$XML16" "$BG/.setup_shim/libxml2.so.2"
        export LD_LIBRARY_PATH="$BG/.setup_shim:${LD_LIBRARY_PATH:-}"
        echo "    (libxml2.so.2 shim active for the CUDA installer)"
    fi
fi
if [ ! -d "$VENV/cuda-12.8.1" ]; then
    FORCE_LOCAL_CUDA=1 CUDA_VERSION=12.8 bash scripts/create_venv_cuda.sh r2s-background
fi

# glibc >= 2.41 declares sinpi/cospi/rsqrt with noexcept, clashing with CUDA
# 12.8 headers (fixed upstream in CUDA >= 12.9). Idempotent local patch:
MF="$VENV/cuda-12.8.1/include/crt/math_functions.h"
if [ -f "$MF" ] && ! grep -q "noexcept (true);" "$MF"; then
    echo "    patching CUDA headers for new glibc (sinpi/cospi/rsqrt noexcept)..."
    sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(double|float) +(sinpi|cospi|sinpif|cospif|rsqrt|rsqrtf)\((double|float) x\));$/\1 noexcept (true);/' "$MF"
fi

echo "[4/5] python deps + CUDA kernel build (the long part)..."
bash -c "source '$VENV/bin/activate' && bash install_env_uv.sh r2s-background"

echo "[5/5] pip extras (pycolmap, pillow-heif)..."
bash -c "source '$VENV/bin/activate' && uv pip install -q pycolmap pillow-heif"

echo "SETUP OK — stage env: $VENV"
echo "Next: put a capture in data/captures/<scene>/ and follow README.md"
