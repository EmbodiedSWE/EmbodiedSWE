#!/usr/bin/env bash

# lerobot environment for the VLA side of CoSiGen: dataset bakes (vla/convert),
# policy training (lerobot-train) and the closed-loop eval client (lerobot-eval
# --env.type=cosigen). It is deliberately a SECOND venv next to ./.venv: Isaac Sim
# is pinned to Python 3.11 / torch 2.7 while lerobot needs Python >= 3.12 and its
# own torch. The two only talk over the socket in vla/eval/protocol.py.
#
# Safe to re-run. Knobs (env vars):
#   COSIGEN_LEROBOT_VENV_DIR   venv location          (default: <repo>/.venv-lerobot)
#   COSIGEN_LEROBOT_EXTRAS     lerobot extras         (default: training,pi,smolvla,diffusion)

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${COSIGEN_LEROBOT_VENV_DIR:-$REPO_ROOT/.venv-lerobot}"
PYTHON_BIN="$VENV_DIR/bin/python"
LEROBOT_DIR="$REPO_ROOT/vla/lerobot"
PLUGIN_DIR="$REPO_ROOT/vla/eval/lerobot_env_cosigen"
EXTRAS="${COSIGEN_LEROBOT_EXTRAS:-training,pi,smolvla,diffusion}"
PYTHON_VERSION="3.12"

fail() {
    echo "[bootstrap-lerobot] ERROR: $*" >&2
    exit 1
}

stage() {
    echo
    echo "[bootstrap-lerobot] $*"
}

trap 'echo "[bootstrap-lerobot] FAILED at line $LINENO" >&2' ERR

command -v uv >/dev/null 2>&1 || fail "uv is not installed or not on PATH."

cd "$REPO_ROOT"

if [[ ! -f "$LEROBOT_DIR/pyproject.toml" ]]; then
    stage "Fetching the vla/lerobot submodule"
    git submodule update --init vla/lerobot
fi
[[ -f "$LEROBOT_DIR/pyproject.toml" ]] || fail "vla/lerobot is empty; run: git submodule update --init vla/lerobot"

if [[ ! -x "$PYTHON_BIN" ]]; then
    stage "Creating Python $PYTHON_VERSION environment at $VENV_DIR"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --prompt lerobot
fi

PYTHON_MINOR="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_MINOR" == "$PYTHON_VERSION" ]] || fail "$PYTHON_BIN is Python $PYTHON_MINOR; expected $PYTHON_VERSION."

stage "Installing lerobot[$EXTRAS] (editable, from vla/lerobot)"
uv pip install --python "$PYTHON_BIN" -e "$LEROBOT_DIR[$EXTRAS]"

stage "Installing the CoSiGen eval plugin (lerobot-eval --env.type=cosigen)"
uv pip install --python "$PYTHON_BIN" -e "$PLUGIN_DIR"

stage "Verifying"
"$PYTHON_BIN" - <<'PY'
import contextlib
import importlib.metadata as md
import io

import torch

print(f"  lerobot {md.version('lerobot')}  torch {torch.__version__}  cuda={torch.version.cuda}")
# torchcodec dlopens FFmpeg 4-7 shared libs at import; a miss prints a long traceback to stderr.
with contextlib.redirect_stderr(io.StringIO()):
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        torchcodec_ok = True
    except Exception:  # noqa: BLE001
        torchcodec_ok = False
if torchcodec_ok:
    print("  torchcodec: OK (fast video decode)")
else:
    print("  torchcodec: NOT loadable -> lerobot falls back to pyav (several x slower dataloading).")
    print("     Fix: put an FFmpeg 5-7 lib/ on LD_LIBRARY_PATH before training, e.g.")
    print("       conda-forge:  export LD_LIBRARY_PATH=<conda env>/lib:$LD_LIBRARY_PATH")
    print("       lmod cluster: module load FFmpeg   (then re-run this script to confirm)")

import lerobot  # noqa: F401,E402

import lerobot_env_cosigen  # noqa: F401

print("  lerobot_env_cosigen: OK")
PY

for cli in lerobot-train lerobot-eval; do
    [[ -x "$VENV_DIR/bin/$cli" ]] || fail "$cli was not installed into $VENV_DIR/bin"
done

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "  WARNING: no ffmpeg on PATH (the pyav wheel encodes on its own; only torchcodec needs system FFmpeg)."
fi

stage "Done. Activate with: source $VENV_DIR/bin/activate"
echo "  bake:   $PYTHON_BIN vla/convert/convert.py <gen_root> --repo-id <id>"
echo "  train:  $VENV_DIR/bin/lerobot-train ..."
echo "  eval:   .venv/bin/python vla/eval/serve.py <bake.json> --headless   # Isaac venv, terminal 1"
echo "          $VENV_DIR/bin/lerobot-eval --policy.path=<ckpt> --env.type=cosigen ...   # terminal 2"
