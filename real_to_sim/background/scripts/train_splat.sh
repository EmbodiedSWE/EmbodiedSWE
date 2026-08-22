#!/usr/bin/env bash
# Stage 3: train the Gaussian splat on a COLMAP workspace.
#   Usage: bash 03_train_splat.sh <colmap_ws> <experiment_name> [downsample]
set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA="$(dirname "$SCRIPTS_DIR")/data"
VENV="$(dirname "$SCRIPTS_DIR")/.venv"
[ -f "$VENV/bin/activate" ] || VENV="$DATA/tools/3dgrut/.venv"   # legacy
GRUT="${GRUT_HOME:-$(dirname "$SCRIPTS_DIR")/3dgrut}"
[ -d "$GRUT/threedgrut" ] || GRUT="$DATA/tools/3dgrut"
WS="${1:?usage: 03_train_splat.sh <colmap_ws> <name> [downsample]}"
NAME="${2:?}"
DS="${3:-2}"
# pre-generate the downsampled cache (3dgrut's auto-gen is unreliable)
source "$VENV/bin/activate"
if [ "$DS" != "1" ] && [ ! -d "$WS/images_$DS" ]; then
    python - "$WS" "$DS" <<'PY'
import cv2, os, sys
ws, ds = sys.argv[1], int(sys.argv[2])
src, dst = f"{ws}/images", f"{ws}/images_{ds}"
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    img = cv2.imread(os.path.join(src, f)); h, w = img.shape[:2]
    cv2.imwrite(os.path.join(dst, f), cv2.resize(img, (w//ds, h//ds), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 96])
print("downscaled cache ready")
PY
fi
cd "$GRUT"
python train.py --config-name apps/colmap_3dgut.yaml path="$WS" \
    out_dir="$DATA/runs" experiment_name="$NAME" dataset.downsample_factor="$DS" \
    export_usd.enabled=false
echo "checkpoint: $(ls -d $DATA/runs/$NAME/*/ckpt_last.pt | tail -1)"
