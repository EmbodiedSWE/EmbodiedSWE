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

Everything installs into a project-local venv named **`cosigen`** (Python 3.11), using **uv** for
the binary stack plus targeted pip workarounds for legacy upstream packages.

The recommended fail-fast installer pins the release-era transitive dependencies that upstream
Isaac Lab 2.3.2 leaves open (`warp-lang`) and repairs the legacy `flatdict` build on current package
indexes. It is safe to re-run and reuses an existing valid `.venv`:

```bash
./scripts/bootstrap_isaaclab_5_1.sh
source .venv/bin/activate
```

The manual equivalent is documented below for debugging.

### 1. Create + activate the venv

From the CoSiGen repo root:

```bash
uv venv --python 3.11 --prompt cosigen      # creates ./.venv (gitignored); prompt shows (cosigen)
source .venv/bin/activate
```

### 2. Install PyTorch + Isaac Sim

```bash
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 3. Install Isaac Lab (from pip — no clone)
Install the Isaac Lab pip packages directly into the active `cosigen` venv (**no `git clone`, no
`isaaclab.sh`**), or follow the official [Isaac Lab Pip Packages guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html).
Pin a version compatible with Isaac Sim 5.1 (`isaaclab` 2.3.2 = the v2.3 release that supports
Isaac Sim 4.5/5.0/5.1):

```bash
uv pip install pip==25.2 setuptools==81.0.0
python -m pip install --no-cache-dir --no-build-isolation flatdict==4.0.1
python -m pip install --no-cache-dir --no-deps warp-lang==1.11.0
CMAKE_POLICY_VERSION_MINIMUM=3.5 python -m pip install --no-cache-dir \
  flatdict==4.0.1 warp-lang==1.11.0 "isaaclab[all]==2.3.2" \
  --extra-index-url https://pypi.nvidia.com
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

For the `packing.egg_carton` contribution, run the evidence-producing validation pipeline from the
CoSiGen root. It stops at the first failure and writes logs plus recorded frame archives under
`validation_artifacts/`. With `CoSiGen_Solutions` checked out beside this repo, it also runs the held-out
G1 reference solution:

```bash
./scripts/validate_egg_carton.sh

# Final stability gate after the first run is calibrated:
./scripts/validate_egg_carton.sh --solution-runs 3
```

An automated pass proves registry wiring, oracle happy/negative paths, recorded rendering, G1 reach,
and reference actuation. The recorded output must still be watched end to end before the task is
accepted; agent difficulty is evaluated separately.

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
# pre-release deps, so its installs take these extra flags. Keep NV an ARRAY expanded as
# "${NV[@]}" (works in bash and zsh) — a scalar NV="..." breaks in zsh, which does not
# word-split unquoted $NV and passes the whole string as one argument.
NV=(--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow)

uv venv env_newton --python 3.12 --prompt env_newton
uv pip install --python "$PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PY" "${NV[@]}" "isaacsim[all,extscache]==6.0.0.1"   # large first download
uv pip install --python "$PY" "${NV[@]}" \
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
  `uv pip install --python "$PY" "${NV[@]}" -e "$SRC/isaaclab_tasks" -e "$SRC/isaaclab_rl" -e "$SRC/isaaclab_ov"`

### 3. The folding suite

The `folding` suite (`robobench/suites/folding/`) folds a T-shirt (VBD cloth) on the coupled
MJWarp+VBD substrate. The in-tree smoke is a simulation CAPABILITY CHECK, not a solution: on
the benchmark env the Franka pinches the shirt with its real fingers and lifts it clear of the
table (cloth-rise verdict). The Franka folding solution is kept out of the benchmark tree
(gitignored `experiments/2026-07-16_tshirt_franka_joint/`).

```bash
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.folding.smokes.tshirt_fold_smoke --headless
```

`OMNI_KIT_ACCEPT_EULA=YES` skips isaacsim 6's first-run EULA prompt in headless runs; on
isaaclab develop a run is headless unless a kit visualizer is requested (pass `--viz kit`; do
NOT combine with `--headless`, which force-disables visualizers).

### 4. The pouring suite (same venv)

The `pouring` suite (`robobench/suites/pouring/`) runs particle liquids (implicit **MPM**)
coupled with MJWarp rigid dynamics: two dynamic Frankas grasp both vessels and pour milk into
coffee. ONE registered env on ONE registered scene: `pouring.latte.bimanual_franka.joint` (the
benchmark: dynamic arms + dynamic vessels + auto-weld grasp contract + 1.5-way liquid
feedback). The in-tree smoke is a simulation CAPABILITY CHECK, not a solution: both Frankas
grasp the vessels through the scene's auto-weld contract and lift them (rise/upright/spill
verdicts). The bimanual-Franka solution is kept out of the benchmark tree (gitignored
`experiments/2026-07-20_latte_bimanual_franka_joint/`).

```bash
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
  -m robobench.suites.pouring.smokes.latte_pour_smoke --headless
```

Note: do **not** record COUPLED-substrate pouring runs with `scripts/record_video.py` live —
live rendering corrupts the coupled MPM physics on this stack. Record via `--dump_states`
(poses + particles to an `.npz`) plus offline replay (a replay renderer last exists at
`f8c101d`: `scripts/replay_render.py`).
