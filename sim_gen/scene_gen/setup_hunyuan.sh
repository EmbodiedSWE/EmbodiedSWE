#!/bin/bash
# Single setup for the scene_gen Hunyuan backend (panorama-only, lean).
#   ./setup_hunyuan.sh [checkout-dir]     (default: ./HunyuanWorld-1.0)
# Installs into the CURRENT python env; install a CUDA torch first if missing.
set -euo pipefail

DIR="${1:-$PWD/HunyuanWorld-1.0}"
[ -d "$DIR" ] || git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git "$DIR"

python -c "import torch" 2>/dev/null || {
    echo "ERROR: install a CUDA build of torch first (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128)"; exit 1; }

pip install "diffusers==0.34.0" "transformers==4.51.0" accelerate peft \
    sentencepiece protobuf safetensors opencv-python-headless pillow "numpy<2" \
    "huggingface_hub[cli]" "imageio[ffmpeg]"

echo
echo "Done. Next:"
echo "  export HUNYUANWORLD_ROOT=$DIR"
echo "  accept the FLUX.1-dev license: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "  python -m sim_gen.scene_gen.generate_scene --spec sim_gen/scene_gen/prompts/robotics_kitchen.json --out-dir out/scene"
