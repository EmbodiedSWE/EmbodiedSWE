#!/usr/bin/env bash

# Reproducible CoSiGen environment for Isaac Sim 6.2 / Isaac Lab 3.0 (branch `isaacsim-6`).
#
# ONE venv for every suite. Isaac Lab 3.0 ships the PhysX backend (isaaclab_physx) and the
# Newton backend (isaaclab_newton) side by side, so the deformable suite no longer needs the
# separate `env_newton` interpreter that main (Isaac Sim 5.1 / Isaac Lab 2.3.2) requires.
#
# How it installs. Isaac Lab 3.0 is a uv project: a lockfile (uv.lock) plus dependency
# overrides (numpy>=2, newton[sim]==1.6.1, torch==2.11.0 cu130, ...) that a hand-written pin
# list would have to replicate. Instead, this script clones Isaac Lab at a pinned
# release/3.0.0 commit and runs `uv sync --frozen` against THAT lockfile, pointed at
# CoSiGen's ./.venv through UV_PROJECT_ENVIRONMENT. CoSiGen is then installed editable on top.
#
# Re-runnable: a clone already at the pinned commit is reused, and `uv sync --inexact` keeps
# the CoSiGen editable install and its extras in place.
#
# Requirements: Linux x86_64 (aarch64 also resolves), git, uv >= 0.8, an NVIDIA driver >= r580
# (the torch cu130 wheels need CUDA 13; r590+ is also the driver line on which Isaac Sim 5.1
# crashes, GitHub issue #135), and roughly 25 GB of disk for Isaac Sim + its extension cache.
#
# Knobs (environment variables):
#   COSIGEN_VENV_DIR        venv path                      default: <repo>/.venv
#   COSIGEN_ISAACLAB_DIR    Isaac Lab clone location       default: <repo>/externals/IsaacLab
#   COSIGEN_ISAACLAB_EXTRAS comma-separated uv extras      default: isaacsim
#                           `isaacsim` = Isaac Sim 6.2 PhysX kernel + RTX rendering, what every
#                           CoSiGen suite uses. Add `ov` only to experiment with the Omniverse
#                           OVPhysX/OVRTX kernels (no CoSiGen suite is validated on them).
#   COSIGEN_SKIP_DRIVER_CHECK=1   skip the r580 driver floor (CPU-only or unusual hosts)

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${COSIGEN_VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

ISAACLAB_DIR="${COSIGEN_ISAACLAB_DIR:-$REPO_ROOT/externals/IsaacLab}"
ISAACLAB_REPO="${COSIGEN_ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_BRANCH="release/3.0.0"
# release/3.0.0 as of 2026-10-13. Its lockfile resolves: isaacsim 6.2.0.0, isaaclab 27.1.0,
# newton 1.6.1, warp-lang 1.17.0, mujoco-warp 3.12.0, torch 2.11.0+cu130, numpy 2.5.1,
# pin 3.9.0 / pin-pink 3.3.0 / daqp 0.8.5 (the pink_ik stack — do NOT layer main's README
# pins `pin==2.7.0 ... numpy==1.26.0` on top of this env; they belong to the 5.1 stack).
ISAACLAB_COMMIT="b88f56aeee07bdeeacfef5a286265715f9681171"
ISAACLAB_EXTRAS="${COSIGEN_ISAACLAB_EXTRAS:-isaacsim}"

PYTHON_VERSION="3.12"
MIN_UV_VERSION="0.8.0"
MIN_DRIVER_MAJOR=580
EXPECTED_ISAACSIM="6.2.0.0"
EXPECTED_NEWTON="1.6.1"
EXPECTED_TORCH="2.11.0"

fail() {
    echo "[bootstrap] ERROR: $*" >&2
    exit 1
}

stage() {
    echo
    echo "[bootstrap] $*"
}

version_ge() {
    # version_ge A B -> true when A >= B (dotted numeric versions)
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

trap 'echo "[bootstrap] FAILED at line $LINENO" >&2' ERR

# ----- prerequisites ---------------------------------------------------------------------------
[[ "$(uname -s)" == "Linux" ]] || fail "Isaac Sim 6.2 requires Linux."
command -v git >/dev/null 2>&1 || fail "git is not installed or not on PATH."
command -v uv >/dev/null 2>&1 || fail "uv is not installed or not on PATH (https://docs.astral.sh/uv/)."
UV_VERSION="$(uv --version | awk '{print $2}')"
version_ge "$UV_VERSION" "$MIN_UV_VERSION" \
    || fail "uv $UV_VERSION is too old for Isaac Lab's lockfile; need >= $MIN_UV_VERSION (uv self update)."

if [[ "${COSIGEN_SKIP_DRIVER_CHECK:-0}" != "1" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available (set COSIGEN_SKIP_DRIVER_CHECK=1 to bypass)."
    nvidia-smi >/dev/null || fail "the NVIDIA driver is not responding."
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
    DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
    [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] || fail "could not parse the driver version from nvidia-smi ('$DRIVER_VERSION')."
    (( DRIVER_MAJOR >= MIN_DRIVER_MAJOR )) \
        || fail "NVIDIA driver $DRIVER_VERSION is below r$MIN_DRIVER_MAJOR; the torch cu130 wheels need CUDA 13."
    echo "[bootstrap] driver $DRIVER_VERSION ok"
fi

# A venv of the wrong Python is almost certainly main's Isaac Sim 5.1 env (Python 3.11). Refuse
# to touch it: `uv sync` would recreate it in place.
if [[ -x "$PYTHON_BIN" ]]; then
    EXISTING_MINOR="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "$EXISTING_MINOR" == "$PYTHON_VERSION" ]] \
        || fail "$VENV_DIR holds Python $EXISTING_MINOR (the Isaac Sim 5.1 env?). This branch needs $PYTHON_VERSION: run from a separate checkout/worktree, or set COSIGEN_VENV_DIR to a new path."
fi

cd "$REPO_ROOT"

# ----- Isaac Lab source checkout, pinned ------------------------------------------------------
if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
    stage "Cloning Isaac Lab ($ISAACLAB_BRANCH) into $ISAACLAB_DIR"
    git clone --branch "$ISAACLAB_BRANCH" --single-branch "$ISAACLAB_REPO" "$ISAACLAB_DIR"
fi

CURRENT_COMMIT="$(git -C "$ISAACLAB_DIR" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$ISAACLAB_COMMIT" ]]; then
    if [[ -n "$(git -C "$ISAACLAB_DIR" status --porcelain)" ]]; then
        fail "$ISAACLAB_DIR is at ${CURRENT_COMMIT:0:10} with local modifications; expected ${ISAACLAB_COMMIT:0:10}. Commit or stash them, or point COSIGEN_ISAACLAB_DIR at a clean checkout."
    fi
    stage "Pinning Isaac Lab to $ISAACLAB_COMMIT ($ISAACLAB_BRANCH)"
    git -C "$ISAACLAB_DIR" fetch --quiet origin "$ISAACLAB_BRANCH"
    git -C "$ISAACLAB_DIR" cat-file -e "${ISAACLAB_COMMIT}^{commit}" 2>/dev/null \
        || git -C "$ISAACLAB_DIR" fetch --quiet origin "$ISAACLAB_COMMIT"
    git -C "$ISAACLAB_DIR" checkout --quiet --detach "$ISAACLAB_COMMIT"
fi
[[ -f "$ISAACLAB_DIR/uv.lock" ]] || fail "$ISAACLAB_DIR has no uv.lock; is it an Isaac Lab 3.0 checkout?"
echo "[bootstrap] Isaac Lab at $(git -C "$ISAACLAB_DIR" rev-parse --short HEAD) ($(git -C "$ISAACLAB_DIR" log -1 --format=%cd --date=short))"

# ----- the locked Isaac Lab environment, materialized as CoSiGen's venv -----------------------
# `--frozen` installs exactly what uv.lock says (no re-resolution); `--inexact` leaves packages
# the lock does not know about (CoSiGen, imageio-ffmpeg) in place on re-runs; `--no-dev` skips
# Isaac Lab's docs/test tooling. uv downloads its own managed CPython 3.12 (Isaac Lab sets
# python-preference = only-managed), so no system Python 3.12 is required.
EXTRA_ARGS=()
IFS=',' read -r -a EXTRAS <<< "$ISAACLAB_EXTRAS"
for extra in "${EXTRAS[@]}"; do
    [[ -n "$extra" ]] && EXTRA_ARGS+=(--extra "$extra")
done

stage "Syncing Isaac Lab's locked environment into $VENV_DIR (extras: $ISAACLAB_EXTRAS)"
(
    cd "$ISAACLAB_DIR"
    UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --frozen --no-dev --inexact \
        --python "$PYTHON_VERSION" "${EXTRA_ARGS[@]}"
)
[[ -x "$PYTHON_BIN" ]] || fail "uv sync did not produce $PYTHON_BIN."

# ----- CoSiGen on top ---------------------------------------------------------------------------
stage "Installing CoSiGen editable (+ imageio-ffmpeg for scripts/record_video.py)"
uv pip install --python "$PYTHON_BIN" -e "$REPO_ROOT" imageio-ffmpeg

# ----- checks ---------------------------------------------------------------------------------
stage "Checking the resolved environment"
# Isaac Lab's overrides (numpy>=2 over packages that still declare numpy<2, etc.) are expected
# to trip metadata checks; report, do not fail.
uv pip check --python "$PYTHON_BIN" \
    || echo "[bootstrap] note: metadata conflicts above are expected under Isaac Lab's uv overrides"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        | sed 's/^/[bootstrap] gpu_driver=/'
fi
"$PYTHON_BIN" - "$EXPECTED_ISAACSIM" "$EXPECTED_NEWTON" "$EXPECTED_TORCH" <<'PY'
import importlib.metadata as md
import sys

expected_isaacsim, expected_newton, expected_torch = sys.argv[1:]
for package in (
    "torch", "torchvision", "isaacsim", "isaaclab", "isaaclab-physx", "isaaclab-newton",
    "isaaclab-contrib", "isaaclab-visualizers", "newton", "warp-lang", "mujoco-warp",
    "numpy", "pin", "pin-pink", "daqp", "imageio-ffmpeg", "cosigen",
):
    try:
        print(f"[bootstrap] {package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"[bootstrap] {package}=MISSING")
problems = []
if md.version("isaacsim") != expected_isaacsim:
    problems.append(f"isaacsim {md.version('isaacsim')} != {expected_isaacsim}")
if md.version("newton") != expected_newton:
    problems.append(f"newton {md.version('newton')} != {expected_newton}")
if not md.version("torch").startswith(expected_torch):
    problems.append(f"torch {md.version('torch')} is not {expected_torch}")
if problems:
    raise SystemExit("[bootstrap] ERROR: unexpected versions: " + "; ".join(problems))
import torch
print(f"[bootstrap] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[bootstrap] gpu={torch.cuda.get_device_name(0)}")
PY

# Isaac Sim's first import shows an interactive EULA prompt; accept it for this check (as the
# README documents for every headless run). pinocchio goes first on purpose: its eigenpy
# converters must register before Isaac Lab's app launcher pulls in Kit (PR #124).
OMNI_KIT_ACCEPT_EULA=YES "$PYTHON_BIN" - <<'PY'
import pinocchio  # noqa: F401  (must precede isaaclab)
import isaaclab, isaaclab_physx, isaaclab_newton, newton, warp  # noqa: F401
print(f"[bootstrap] imports OK; isaaclab={isaaclab.__version__} newton={newton.__version__} warp={warp.__version__}")
PY

echo
echo "[bootstrap] COMPLETE"
echo "[bootstrap] activate with: source $VENV_DIR/bin/activate"
echo "[bootstrap] then:          export OMNI_KIT_ACCEPT_EULA=YES"
echo "[bootstrap] list tasks:    python -m robobench.scripts.smoke --list"
echo "[bootstrap] smoke a task:  python -m robobench.scripts.smoke --env assembly.bulb.franka.osc --headless"
echo "[bootstrap] deformable:    python -m robobench.suites.deformable.smokes.tshirt_fold_smoke --headless   # same venv"
