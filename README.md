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

## Newton env (folding suite)

The `folding` suite (T-shirt folding, `robobench/suites/folding/`) runs cloth — which needs
IsaacLab **develop**'s Newton physics backend (MJWarp rigid + VBD cloth). That branch is not on
PyPI, so the folding suite gets its own project-local venv, **`env_newton`** (Python 3.12,
isaacsim 6.0, torch cu130), with the isaaclab packages installed *editable* from an IsaacLab
**develop** checkout. The assembly suite keeps using `.venv` (isaaclab 2.3.2 / PhysX); the two
venvs coexist — only the interpreter you launch with differs.

Extra prerequisite: the torch cu130 wheels need an NVIDIA driver ≥ r580 (CUDA 13).

### 1. Clone IsaacLab (develop)

Clone anywhere you like — it is only consumed as an editable source tree (do **not** run
IsaacLab's own installer / `isaaclab.sh`):

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
git -C ~/IsaacLab checkout d7d004217c60b4790f721565bf5d40243addcb0e   # tested commit (develop, 2026-06-17)
```

Newer `develop` may work, but this commit is what the suite is tested against — develop moves
fast and breaks conventions vs 2.x (e.g. quaternions are **xyzw** there, not wxyz).

### 2. Build env_newton (one-time)

From the CoSiGen repo root, with `SRC` pointing at *your* checkout's `source/` dir:

```bash
SRC=~/IsaacLab/source                 # <-- adjust to your IsaacLab checkout
PY=env_newton/bin/python
# isaacsim deps span pypi.org + pypi.nvidia.com at different versions, and isaacsim pins some
# pre-release deps, so its installs take these extra flags:
NV="--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow"

uv venv env_newton --python 3.12 --prompt env_newton
uv pip install --python "$PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PY" $NV "isaacsim[all,extscache]==6.0.0.1"   # large first download
uv pip install --python "$PY" $NV \
  -e "$SRC/isaaclab_newton[all]" -e "$SRC/isaaclab_physx[newton]" \
  -e "$SRC/isaaclab_ovphysx" -e "$SRC/isaaclab_visualizers[kit]" \
  -e "$SRC/isaaclab_contrib" -e "$SRC/isaaclab_assets" -e "$SRC/isaaclab"
uv pip install --python "$PY" imageio imageio-ffmpeg   # for record_video
uv pip install --python "$PY" -e .                     # robobench itself (declares no other deps)
```

Notes:

- The Newton engine itself needs no separate install — `isaaclab_newton[all]` pins and pulls the
  exact `newton` git commit it is built against.
- All seven `-e` packages are required: `isaaclab_ovphysx`/`isaaclab_physx` are hard imports of
  isaaclab's app launcher, and `isaaclab_visualizers[kit]` drives rendering (the folding smokes
  default to the kit visualizer).
- Optional — only to run IsaacLab's in-tree reference tasks (e.g. `Isaac-Lift-Cloth-Franka-v0`),
  not needed by the folding suite:
  `uv pip install --python "$PY" $NV -e "$SRC/isaaclab_tasks" -e "$SRC/isaaclab_rl" -e "$SRC/isaaclab_ov"`

### 3. Run the folding suite

From the CoSiGen repo root (`OMNI_KIT_ACCEPT_EULA=YES` skips isaacsim 6's first-run EULA prompt
in headless runs):

```bash
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.folding.scripts.tshirt_fold_smoke --headless

# watch it live in the Isaac Sim GUI: isaaclab develop runs headless unless a kit visualizer is
# requested — pass --viz kit (and do NOT pass --headless, which force-disables visualizers)
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.folding.scripts.tshirt_fold_smoke --viz kit

# record a video (the folding smoke auto-enables the kit visualizer when cameras are on —
# isaaclab develop pumps rendering through visualizers, else the capture stays empty)
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python scripts/record_video.py \
  robobench.suites.folding.scripts.tshirt_fold_smoke \
  --video robobench/suites/folding/videos/tshirt_fold.mp4 \
  --eye 0.9 -1.6 0.9 --target-at 0.0 -0.5 0.2
```
