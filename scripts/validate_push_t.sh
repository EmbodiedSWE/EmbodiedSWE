#!/usr/bin/env bash

# End-to-end Linux/NVIDIA evidence bundle for puzzle.push_t.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${COSIGEN_VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
SOLUTIONS_ROOT="${COSIGEN_SOLUTIONS_ROOT:-$REPO_ROOT/../CoSiGen_Solutions}"
SOLUTION_RUNS=3
SKIP_SOLUTION=0
ARTIFACTS_DIR=""
CURRENT_LOG=""

usage() {
    cat <<'EOF'
Usage: ./scripts/validate_push_t.sh [options]

Options:
  --artifacts-dir PATH   Write evidence under PATH instead of a timestamped directory.
  --solutions-root PATH  CoSiGen_Solutions checkout (default: sibling of CoSiGen).
  --solution-runs N      Number of fresh reference-solution runs (default: 3).
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

[[ "$SOLUTION_RUNS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--solution-runs must be positive" >&2
    exit 2
}
[[ "$(uname -s)" == "Linux" ]] || { echo "validation requires Linux" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || {
    echo "missing $PYTHON_BIN; run ./scripts/bootstrap_isaaclab_5_1.sh first" >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable" >&2; exit 1; }
nvidia-smi >/dev/null || { echo "NVIDIA driver is not responding" >&2; exit 1; }

if [[ -z "$ARTIFACTS_DIR" ]]; then
    ARTIFACTS_DIR="$REPO_ROOT/validation_artifacts/push-t-$(date +%Y%m%d-%H%M%S)"
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
    "$PYTHON_BIN" --version
    uname -a
    nvidia-smi
    "$PYTHON_BIN" - <<'PY'
import importlib.metadata as md
import torch

for package in ("torch", "isaacsim", "isaaclab", "cosigen"):
    print(f"{package}={md.version(package)}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
    "$PYTHON_BIN" -m pip check
} 2>&1 | tee "$CURRENT_LOG"
require_line "cuda_available=True" "$CURRENT_LOG"

run_logged registry "$PYTHON_BIN" -m robobench.scripts.smoke --list
for preset in puzzle.push_t puzzle.push_t.g1.joint puzzle.push_t.g1.pink_ik; do
    require_line "$preset" "$ARTIFACTS_DIR/registry.log"
done

run_logged oracle-full \
    timeout --signal=TERM --kill-after=30s 20m \
    "$PYTHON_BIN" -m robobench.suites.puzzle.smokes.push_t_smoke \
    --headless \
    --out "$ARTIFACTS_DIR/push_t_full.npz" \
    --video "$ARTIFACTS_DIR/push_t_full.mp4"
require_line "[smoke] RESULT: ALL PASS" "$ARTIFACTS_DIR/oracle-full.log"
[[ -s "$ARTIFACTS_DIR/push_t_full.mp4" ]]

run_logged oracle-demo \
    timeout --signal=TERM --kill-after=30s 15m \
    "$PYTHON_BIN" -m robobench.suites.puzzle.smokes.push_t_smoke \
    --headless \
    --demo \
    --out "$ARTIFACTS_DIR/push_t_demo.npz" \
    --video "$ARTIFACTS_DIR/push_t_demo.mp4"
require_line "[smoke] RESULT: ALL PASS" "$ARTIFACTS_DIR/oracle-demo.log"
[[ -s "$ARTIFACTS_DIR/push_t_demo.mp4" ]]

run_logged g1-binding \
    timeout --signal=TERM --kill-after=30s 15m \
    "$PYTHON_BIN" -m robobench.scripts.robot_binding_smoke \
    --env puzzle.push_t.g1.pink_ik \
    --reach_body block \
    --hover 0,0,0.16 \
    --headless \
    --out "$ARTIFACTS_DIR/g1_binding_frames.npz"
require_line "[binding-smoke] RESULT: ALL PASS" "$ARTIFACTS_DIR/g1-binding.log"

if (( ! SKIP_SOLUTION )); then
    SOLVE_PY="$SOLUTIONS_ROOT/puzzle/push_t/g1/pink_ik/solve.py"
    [[ -f "$SOLVE_PY" ]] || {
        echo "missing reference solution: $SOLVE_PY" >&2
        exit 1
    }
    for (( run = 1; run <= SOLUTION_RUNS; run++ )); do
        run_logged "solution-$run" \
            timeout --signal=TERM --kill-after=30s 30m \
            "$PYTHON_BIN" "$SOLVE_PY"
        require_line "[push-t] RESULT score=100 success=True" \
            "$ARTIFACTS_DIR/solution-$run.log"
    done
fi

CURRENT_LOG=""
echo
echo "[validate] ALL AUTOMATED CHECKS PASS"
echo "[validate] evidence: $ARTIFACTS_DIR"
echo "[validate] still required: watch both MP4s end to end"
