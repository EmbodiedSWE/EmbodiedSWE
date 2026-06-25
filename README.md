# CoSiGen

Sim data generation with coding agents — built on **robobench**, a relocatable robot-assembly
benchmark suite running on **Isaac Lab 5.1 / Isaac Sim** (PhysX 5).

## Prerequisites

- Linux + an NVIDIA GPU (CUDA 12.x driver).
- [`uv`](https://docs.astral.sh/uv/) for env + package management.
- Isaac Sim 5.1 + Isaac Lab — installed in [Setup](#setup) below.

robobench imports `isaaclab*` / `isaacsim` / `torch` / `pxr` from the venv; it consumes the Isaac
stack purely as installed packages, so **no Isaac Lab source clone is needed**.

## Setup

Everything installs through **uv** into a project-local venv named **`cosigen`** (Python 3.11).

### 1. Create + activate the venv

From the CoSiGen repo root:

```bash
uv venv --python 3.11 --prompt cosigen      # creates ./.venv (gitignored); prompt shows (cosigen)
source .venv/bin/activate
```

### 2. Install PyTorch + Isaac Sim

```bash
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 3. Install Isaac Lab (from pip — no clone)
Install the Isaac Lab pip packages directly into the active `cosigen` venv (**no `git clone`, no
`isaaclab.sh`**), or follow the official [Isaac Lab Pip Packages guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html).
Pin a version compatible with Isaac Sim 5.1 (`isaaclab` 2.3.2 = the v2.3 release that supports
Isaac Sim 4.5/5.0/5.1):

```bash
uv pip install setuptools wheel
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install "isaaclab[all]==2.3.2" \
  --extra-index-url https://pypi.nvidia.com \
  --no-build-isolation-package flatdict
```

### 4. Install robobench (this repo)

Back in the CoSiGen repo root, with the venv still active:

```bash
uv pip install -e .
```


## Run

Always `source .venv/bin/activate` first

```bash
# List every registered env (suite.scene[.robot[.control_mode]])
python -m robobench.scripts.smoke --list

# Smoke-test one env with random actions
python -m robobench.scripts.smoke --env assembly.ikea_table.g1.joint

```