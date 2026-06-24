# CoSiGen

Sim data generation with coding agents — built on **robobench**, a relocatable robot-assembly
benchmark suite running on **Isaac Lab 5.1 / Isaac Sim** (PhysX 5).

## Prerequisites

- Linux + an NVIDIA GPU (CUDA 12.x driver).
- [`uv`](https://docs.astral.sh/uv/) for env + package management.
- Isaac Sim 5.1 + Isaac Lab — installed in [Setup](#setup) below.

robobench imports `isaaclab*` / `isaacsim` / `torch` / `pxr` from the venv; it never reaches into the
Isaac Lab source tree, so that clone can live anywhere.

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
uv pip install "isaacsim[all,extscache]==5.1.0"
```

### 3. Install Isaac Lab

Clone Isaac Lab **anywhere** and install it into the active `cosigen` venv (official quickstart:
<https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html>). Use a version compatible
with Isaac Sim 5.1 (current `main` works):

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
CMAKE_POLICY_VERSION_MINIMUM=3.5 ./isaaclab.sh --install
```

- `CMAKE_POLICY_VERSION_MINIMUM=3.5` is **required** — without it the build of `egl-probe` (a
  transitive dep) fails because its CMakeLists predates the minimum modern CMake accepts.
- First run prompts for the NVIDIA Omniverse EULA — answer `Yes` (or set `OMNI_KIT_ACCEPT_EULA=Yes`).
- This installs the `isaaclab*` packages editable into the active venv; the clone location doesn't matter
  afterward.

### 4. Install robobench (this repo)

Back in the CoSiGen repo root, with the venv still active:

```bash
uv pip install -e .
```

robobench declares no heavy deps of its own — it consumes the Isaac stack from the venv built above.

## Run

Always `source .venv/bin/activate` first — Isaac scripts won't find Python otherwise.

```bash
# List every registered env (suite.scene[.robot[.control_mode]])
python -m robobench.scripts.smoke --list

# Smoke-test one env with random actions, headless
python -m robobench.scripts.smoke --env assembly.ikea_table.g1.joint --headless

# Assembly firmness smoke: legs -> studs -> screw -> auto-weld -> lift + flip (watch live)
python -m robobench.suites.assembly.scripts.ikea_table_assembly_smoke --livestream 2

# Reachability test: can the fixed-base humanoid reach the legs under Pink IK? (visual)
python -m robobench.suites.assembly.scripts.ikea_table_assembly_reachable_test --robot g1 --livestream 2
python -m robobench.suites.assembly.scripts.ikea_table_assembly_reachable_test --robot gr1t2 --arm left --livestream 2

# Regenerate a robot's kinematics URDF from its vendored USD (needs the app)
python -m robobench.scripts.make_urdf --robot gr1t2
```

Notes:
- `--headless` runs without a display; `--livestream 2` streams the viewport (WebRTC) for remote viewing.
- Isaac Sim swallows stdout once it starts — scripts that report a verdict write to `--out_file`, or you
  watch the livestream.
- Pink IK scripts must `import pinocchio` **before** launching the app (the provided scripts already do).

## Troubleshooting (rendering)

Headless training/sim needs nothing extra. The GUI / livestream / camera paths need system libs and a
compatible driver:

```bash
sudo apt install -y libglu1-mesa libegl1 libgl1 libsm6 libxrandr2 libxi6 libxcursor1 \
                    libxinerama1 libfontconfig1 libxkbcommon0
```

- On RTX 50-series (Blackwell) GPUs, the **595 driver branch crashes** Isaac Sim 5.1's RTX renderer
  (`rtx.scenedb.plugin`). Use the **580** branch instead (`nvidia-driver-580-open`).
- Shaders compile on first launch (slow once, then cached under `~/.cache/ov`).
