#!/usr/bin/env bash
# One-time setup for the objects stage (real object photos -> 3D assets via TRELLIS.2).
#
# Mirrors ../background/install.sh conventions: stage-owned uv venv at
# real_to_sim/objects/.venv. The CUDA toolkit is REUSED from the background
# stage's vendored copy (background/.venv/cuda-12.8.1) — install the
# background stage first, or point CUDA_HOME at any CUDA >= 12.8 toolkit.
#
# Blackwell note (RTX 50xx / sm_120): upstream's flash-attn 2.7.3 predates
# these GPUs, so it is skipped; attention runs through xformers instead
# (stage scripts set ATTN_BACKEND=xformers).
#
#   Usage: bash scripts/install.sh
set -euo pipefail
OBJ="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$OBJ/.venv"
SRC="$OBJ/.setup_shim/src"
export CUDA_HOME="${CUDA_HOME:-$OBJ/../background/.venv/cuda-12.8.1}"

[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: nvcc not found at $CUDA_HOME — install the background stage first, or set CUDA_HOME"; exit 1; }

echo "[1/5] TRELLIS.2 source (pinned submodule; o-voxel needs its nested eigen)..."
REPO="$(cd "$OBJ/../.." && pwd)"
git -C "$REPO" submodule update --init --recursive real_to_sim/objects/trellis2

echo "[2/5] stage venv at $VENV ..."
[ -f "$VENV/bin/activate" ] || uv venv "$VENV" --python 3.11 --prompt r2s-objects
source "$VENV/bin/activate"

echo "[3/5] python deps (torch 2.8 cu128 + xformers)..."
uv pip install -q setuptools wheel ninja
uv pip install -q torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install -q imageio imageio-ffmpeg tqdm easydict "opencv-python-headless<5" \
    trimesh "transformers<5" pillow kornia timm einops zstandard \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

echo "[4/5] CUDA extensions (nvcc 12.8 + gcc-14, TORCH_CUDA_ARCH_LIST=12.0)..."
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST="12.0"
mkdir -p "$SRC"
[ -d "$SRC/nvdiffrast" ] || git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$SRC/nvdiffrast"
[ -d "$SRC/nvdiffrec" ]  || git clone -q -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$SRC/nvdiffrec"
[ -d "$SRC/CuMesh" ]     || git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git "$SRC/CuMesh"
[ -d "$SRC/FlexGEMM" ]   || git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$SRC/FlexGEMM"
for pkg in nvdiffrast nvdiffrec CuMesh FlexGEMM; do
    echo "    building $pkg ..."
    uv pip install -q "$SRC/$pkg" --no-build-isolation
done
echo "    building o-voxel ..."
uv pip install -q "$OBJ/trellis2/o-voxel" --no-build-isolation

echo "[5/5] verify imports..."
cd "$OBJ/trellis2"
ATTN_BACKEND=xformers python -c "
import torch, o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
print('cuda available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0))
"
echo "SETUP OK — stage env: $VENV"
echo "Next: python scripts/generate_mesh.py <image> <name>  (downloads TRELLIS.2-4B on first run)"
