# CoSiGen Grading Kit — full-trajectory replay grading

Grades every reconstructed code version of every agent run by actually replaying it in
Isaac (PhysX), under the official verify rules (env reset by the harness; `reset` and
`set_states` blocked for the delivered code; `scene.success()` is the verdict; the
scene's own score is tracked at every sim step so each version is graded by the best
state it ever reaches). After each replay the full env state (`env.get_states()`:
object poses, joint states, grasp state) is saved as a `.pt` for later analysis.

## Contents
    manifest.json          label -> {preset, versions, tgz_bytes}   (327 runs, 7480 versions)
    versions/<label>.tgz   per-run reconstructed workspaces, one dir per code version
                           (versions/vNNNN/... + versions/versions.json with timestamps
                           + versions/overlay/ = run data files, e.g. calibration .pt)
    grader/grade_replay_batch.py     the batch replay grader (one Isaac boot, many jobs)
    grader/run_label_standalone.py   per-label driver: untar -> grade -> grades_v2.json
    robobench/             the benchmark envs + assets (put its PARENT on PYTHONPATH)
    pyproject.toml

## Environment (exact recipe; needs an RTX-capable GPU, e.g. L20, and driver >= 535)
    python3.11 -m venv /opt/venv && . /opt/venv/bin/activate
    pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
    pip install setuptools wheel
    CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install 'isaaclab[all]==2.3.2' \
        --extra-index-url https://pypi.nvidia.com --no-build-isolation-package flatdict
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
    export PYTHONPATH=/path/to/grading_kit     # so `import robobench` works

If Kit segfaults on boot in a container, install the X11 client libs
(libsm6 libice6 libxext6 libxi6 libxrender1 libxfixes3 libxcursor1 libxinerama1
libglvnd0 libgl1 libegl1 libvulkan1) and write the NVIDIA Vulkan ICD json
(/usr/share/vulkan/icd.d/nvidia_icd.json -> libGLX_nvidia.so.0). On bare-metal nodes
with a normal driver install this is usually already fine.

## Run (one label per GPU; parallelize freely across nodes)
    python grader/run_label_standalone.py \
        --label coffee_opus_5_r2 --kit /path/to/grading_kit --out /path/to/grades_out

  * `--stage-dir /workspace` (default) stages each version at /workspace so agents'
    hardcoded absolute paths resolve; the node needs a writable /workspace (mkdir it).
  * Timeouts: intermediate versions 1800 s, the final version 3600 s (the official
    verify budget). A hung solve is killed by a watchdog AFTER writing its partial
    result with the peak score reached, then the batch resumes automatically.
  * Output per label: <out>/<label>/grades_v2.json (+ states_v2/*.pt).
  * Already-graded labels are skipped, so the sweep is resumable and shardable by
    simply invoking every label on whatever node is free.

## Semantics guarantees
  * Every version's workspace was reconstructed from the run's FULL agent history
    (every Write/Edit, bash heredoc, codex apply_patch, and the agents' own inline
    python edit scripts, gated on tool calls that actually executed), anchored to the
    run's submission snapshots at their recorded timestamps, and verified byte-exact
    against the run's mirrored final workspace.
  * Identical programs (same import-closure hash) are graded once per run and copied.
  * Grading is deterministic per (preset, seed): env built and reset with seed 0.
