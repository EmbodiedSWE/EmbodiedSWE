#!/usr/bin/env bash

# End-to-end Linux/NVIDIA evidence bundle for packing.egg_carton.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${COSIGEN_VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
SOLUTIONS_ROOT="${COSIGEN_SOLUTIONS_ROOT:-$REPO_ROOT/../CoSiGen_Solutions}"
SOLUTION_RUNS=1
SKIP_SOLUTION=0
ARTIFACTS_DIR=""
CURRENT_LOG=""

usage() {
    cat <<'EOF'
Usage: ./scripts/validate_egg_carton.sh [options]

Options:
  --artifacts-dir PATH   Write evidence under PATH instead of a timestamped directory.
  --solutions-root PATH  CoSiGen_Solutions checkout (default: sibling of CoSiGen).
  --solution-runs N      Number of fresh reference-solution runs (default: 1).
  --skip-solution        Validate only the public scene/oracle/G1 binding.
  -h, --help             Show this help.
EOF
}

while (( $# )); do
    case "$1" in
        --artifacts-dir)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --solutions-root)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
            SOLUTIONS_ROOT="$2"
            shift 2
            ;;
        --solution-runs)
            [[ $# -ge 2 ]] || { echo "missing value for $1" >&2; exit 2; }
            SOLUTION_RUNS="$2"
            shift 2
            ;;
        --skip-solution)
            SKIP_SOLUTION=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "$SOLUTION_RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "--solution-runs must be positive" >&2; exit 2; }
[[ "$(uname -s)" == "Linux" ]] || { echo "validation requires Linux" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || {
    echo "missing $PYTHON_BIN; run ./scripts/bootstrap_isaaclab_5_1.sh first" >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable" >&2; exit 1; }
nvidia-smi >/dev/null || { echo "NVIDIA driver is not responding" >&2; exit 1; }

if [[ -z "$ARTIFACTS_DIR" ]]; then
    ARTIFACTS_DIR="$REPO_ROOT/validation_artifacts/egg-carton-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$ARTIFACTS_DIR"
ARTIFACTS_DIR="$(cd "$ARTIFACTS_DIR" && pwd)"

trap 'echo "[validate] FAILED; inspect ${CURRENT_LOG:-$ARTIFACTS_DIR}" >&2' ERR

run_logged() {
    local name="$1"
    shift
    CURRENT_LOG="$ARTIFACTS_DIR/$name.log"
    echo
    echo "[validate] RUN $name"
    "$@" 2>&1 | tee "$CURRENT_LOG"
}

require_line() {
    local needle="$1"
    local file="$2"
    grep -Fq "$needle" "$file" || {
        echo "[validate] expected line not found in $file: $needle" >&2
        return 1
    }
}

cd "$REPO_ROOT"

CURRENT_LOG="$ARTIFACTS_DIR/environment.txt"
{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "cosigen_rev=$(git rev-parse HEAD)"
    echo "cosigen_branch=$(git branch --show-current)"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    uname -a
    cat /etc/os-release
    ldd --version
    nvidia-smi
    "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import torch

for package in ("torch", "torchvision", "torchaudio", "isaacsim", "isaaclab", "flatdict", "warp-lang", "cosigen"):
    print(f"{package}={md.version(package)}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
    "$PYTHON_BIN" -m pip check
} 2>&1 | tee "$CURRENT_LOG"
require_line "cuda_available=True" "$CURRENT_LOG"

run_logged registry "$PYTHON_BIN" -m robobench.scripts.smoke --list
for preset in \
    packing.egg_carton \
    packing.egg_carton.g1.joint \
    packing.egg_carton.g1.pink_ik; do
    require_line "$preset" "$ARTIFACTS_DIR/registry.log"
done

run_logged isaac-app \
    timeout --signal=TERM --kill-after=10s 5m \
    "$PYTHON_BIN" -c \
    'from isaaclab.app import AppLauncher; app = AppLauncher(headless=True).app; print("ISAAC_APP_SMOKE_OK", flush=True); import os; os._exit(0)'
require_line "ISAAC_APP_SMOKE_OK" "$ARTIFACTS_DIR/isaac-app.log"

run_logged oracle-full \
    timeout --signal=TERM --kill-after=30s 30m \
    "$PYTHON_BIN" -m robobench.suites.packing.smokes.egg_carton_smoke \
    --headless \
    --out "$ARTIFACTS_DIR/egg_carton_full.npz"
require_line "[smoke] RESULT: ALL PASS" "$ARTIFACTS_DIR/oracle-full.log"

run_logged oracle-demo \
    timeout --signal=TERM --kill-after=30s 20m \
    "$PYTHON_BIN" -m robobench.suites.packing.smokes.egg_carton_smoke \
    --headless \
    --demo \
    --out "$ARTIFACTS_DIR/egg_carton_demo.npz"
require_line "[smoke] RESULT: ALL PASS" "$ARTIFACTS_DIR/oracle-demo.log"

run_logged g1-binding \
    timeout --signal=TERM --kill-after=30s 20m \
    "$PYTHON_BIN" -m robobench.scripts.robot_binding_smoke \
    --env packing.egg_carton.g1.pink_ik \
    --reach_body basket \
    --hover 0,0,0.16 \
    --headless \
    --out "$ARTIFACTS_DIR/g1_binding_frames.npz"
require_line "[binding-smoke] RESULT: ALL PASS" "$ARTIFACTS_DIR/g1-binding.log"

if (( ! SKIP_SOLUTION )); then
    SOLVE_PY="$SOLUTIONS_ROOT/packing/egg_carton/g1/pink_ik/solve.py"
    [[ -f "$SOLVE_PY" ]] || {
        echo "missing reference solution: $SOLVE_PY" >&2
        echo "pass --solutions-root PATH or use --skip-solution" >&2
        exit 1
    }
    for (( run = 1; run <= SOLUTION_RUNS; run++ )); do
        run_logged "solution-$run" \
            timeout --signal=TERM --kill-after=30s 90m \
            "$PYTHON_BIN" "$SOLVE_PY"
        require_line \
            "[egg-carton] RESULT score=100 success=True" \
            "$ARTIFACTS_DIR/solution-$run.log"
    done
fi

CURRENT_LOG=""
echo
echo "[validate] ALL AUTOMATED CHECKS PASS"
echo "[validate] evidence: $ARTIFACTS_DIR"
echo "[validate] still required: watch the recorded frames/video end to end"
