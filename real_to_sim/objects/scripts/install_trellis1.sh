#!/usr/bin/env bash
# Add-on: TRELLIS v1 (multi-image conditioning) into the existing objects venv.
# Run scripts/install.sh first. v1 shares torch/xformers/nvdiffrast/utils3d with
# the TRELLIS.2 install; this adds its extras:
#   - pip deps (rembg/onnxruntime for segmentation, open3d/xatlas/pymeshfix/
#     pyvista/igraph/scipy for GLB postprocessing)
#   - spconv (sparse conv backend; prebuilt wheel, newest CUDA variant first)
#   - diff-gaussian-rasterization (mip-splatting fork; texture bake renders
#     the gaussians) — built from source for sm_120
#   - a kaolin SHIM: flexicubes imports only kaolin.utils.testing.check_tensor
#     (an assert helper), not worth the real kaolin (no torch-2.8 wheels)
# Skipped: diffoctreerast (radiance-field renderer; we never request that
# format), flash-attn (Blackwell), vox2seq/gradio (training/demo only).
set -euo pipefail
OBJ="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$OBJ/.venv"
SRC="$OBJ/.setup_shim/src"
export CUDA_HOME="${CUDA_HOME:-$OBJ/../background/.venv/cuda-12.8.1}"
source "$VENV/bin/activate"

echo "[1/4] TRELLIS v1 source (pinned submodule)..."
REPO="$(cd "$OBJ/../.." && pwd)"
git -C "$REPO" submodule update --init --recursive real_to_sim/objects/trellis1

echo "[2/4] python deps..."
uv pip install -q rembg onnxruntime open3d xatlas pyvista pymeshfix igraph scipy

echo "[3/4] compiled extensions..."
for v in cu126 cu124 cu120; do
    if uv pip install -q "spconv-$v"; then echo "    spconv-$v installed"; break; fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST="12.0"
mkdir -p "$SRC"
[ -d "$SRC/mip-splatting" ] || git clone -q --recursive https://github.com/autonomousvision/mip-splatting.git "$SRC/mip-splatting"
uv pip install -q "$SRC/mip-splatting/submodules/diff-gaussian-rasterization/" --no-build-isolation

echo "[4/4] kaolin shim + verify..."
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SITE/kaolin/utils"
touch "$SITE/kaolin/__init__.py" "$SITE/kaolin/utils/__init__.py"
cat > "$SITE/kaolin/utils/testing.py" <<'EOF'
# Shim for TRELLIS v1's flexicubes: real kaolin has no torch-2.8 wheels and
# flexicubes only uses this assert helper.
def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    return True
EOF
cd "$OBJ/trellis1"
ATTN_BACKEND=xformers SPCONV_ALGO=native python -c "
import torch, spconv, diff_gaussian_rasterization
from trellis.pipelines import TrellisImageTo3DPipeline
print('trellis1 OK; cuda:', torch.cuda.is_available())
"
echo "SETUP OK — trellis v1 available in the stage venv"
