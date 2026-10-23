# RoboBench on Harbor

This adapter exports the published **28-task** RoboBench roster as Harbor tasks.
It uses Harbor **0.24.0**, Docker Compose, evaluator-provided NVIDIA GPUs, and the
existing RoboBench world builder and hidden graders. It does not rent hardware,
invoke a model during export, or require a particular coding agent.

The task list is explicit in [catalog.py](catalog.py). It excludes the extra
`assembly.pc_motherboard_gpu_ram` scene, variant foods, and alternative robot
bindings. The selected bindings use Franka OSC, bimanual Franka OSC, G1 Pink IK,
and the registered Newton bindings. In this checkout `deformable.knot` uses
ALOHA. These are versioned adapter choices, not a claim to reconstruct every
historical paper run. `list --validate-registry` verifies all 28 against the
actual environment and grader registries without starting Isaac.

## Evaluation protocol

Agents write `solution/solve.py` exposing `solve(env)` and call
`submit "note"` to freeze checkpoints. Harbor transfers `/submissions` and the
final `/workspace/solution` into a **separate verifier container**. Hidden grader
files are only in that container's build context; agent filesystem changes,
background processes, and installed packages are not inherited.

Each distinct checkpoint, plus the final solution, runs once per configured
seed in a **new simulator process**. Each seed uses `num_envs` trajectories.
Scores and success rates are averaged over seeds before selecting the highest
score. Ties prefer higher success, then the earliest checkpoint, with final
last. The reported success rate comes from that same selected checkpoint.
Byte-identical solution directories are graded once, preventing repeated
submissions from buying extra stochastic attempts.
Directory layout and executable flags are included in checkpoint identities.
The verifier freezes all inputs before running submitted code and gives each
seed its own writable working copy, so generated files cannot affect later seeds.

Harbor receives `reward.json` with separate `score` and `success` metrics in
`[0, 1]`; do not average these two metrics together. The native rubric and
`GradedEnv` restrictions remain authoritative, including partial progress
before a solution exception or graceful timeout. A solution that hangs without
a verdict after entering its own code scores zero for that rollout. Missing
solutions and changed snapshots also score zero. Infrastructure errors,
incomplete verdicts, nonfinite scores, and camera/VLM failures produce **no
reward** and an `error.json`, so Harbor reports an evaluation error rather than
an ordinary model score. **Harbor's default mean still substitutes zero for
missing rewards. Do not publish that aggregate when any trials failed.** Use
the strict summary below, which refuses incomplete or failed evaluations.

Defaults: four hours for the agent, 30 minutes per rollout, seed 0, one
trajectory, and up to 64 frozen checkpoints plus final. Limits are explicit in
both the prompt and `protocol.json`. The verifier's outer timeout covers the
worst-case number of checkpoints and seeds, including termination grace. Change
these at export time; report them alongside results. These defaults can make
grading substantially longer than the agent phase.

The separate container protects the evaluation from agent-phase modifications.
The existing benchmark deliberately runs arbitrary `solve(env)` Python in the
grader process and enforces behavioral restrictions with `GradedEnv` and the
task rules. This is **not a security boundary against adversarial submitted
Python**: a submission can introspect its process. This adapter does not claim
to solve that existing benchmark limitation. Use isolated hosts and retain
artifacts for auditing frontier-model runs.

## Requirements

- Linux NVIDIA host with the driver required by the chosen backend. Newton's
  documented CUDA 13 stack requires driver r580 or newer.
- Docker with NVIDIA Container Toolkit/CDI configured, and Compose >= 2.24.4.
  Follow [the existing host setup](../docker/INSTALL.md).
- Two evaluator-provided runtime images, pinned by `@sha256:` digest for a
  release: PhysX and Newton. Each must expose `/opt/venv/bin/python`, the matching
  Isaac stack, `torch`, `numpy`, `huggingface_hub`, and `filelock`; include
  Pinocchio/Pink/DAQP for G1 bindings. The image must provide bash, coreutils,
  `useradd`, and the runtime libraries needed by Isaac/Vulkan. Harbor installs
  its selected agent itself. Runtime dependencies must be baked into these
  images; agent-installed dependencies are not copied to the verifier.
- Python 3.12+ on the Harbor orchestration host. This is separate from the
  PhysX Python 3.11 and Newton Python 3.12 runtimes.

Installation, export, and container smoke checks need no grading-service
credentials. Actual deformable scoring retains the native grader's existing
`VLM_JUDGE_API_KEY` and network requirements. Harbor passes that credential only
to the verifier phase; it is never included in task images or export receipts.

The generated Compose file uses **CDI**, clears Harbor's legacy GPU reservation,
and preserves `gpus = 1` in the Harbor resource declaration. It is intended for
the local Docker backend. Other Harbor providers are not validated by this
adapter. There is no implicit GPU scheduler: run **one concurrent trial per
GPU**. Agent and separate-verifier containers may overlap in lifetime; their
simulation phases use the same selected device sequentially.

## Build and export

From the repository root, install host dependencies into a separate environment:

```bash
uv venv .venv-harbor --python 3.12
uv pip install --python .venv-harbor/bin/python -r eval/harbor_support/requirements.txt
.venv-harbor/bin/python eval/scripts/harbor_benchmark.py list
```

Prepare the simulator environments using the [PhysX installer](../../scripts/bootstrap_isaaclab_5_1.sh)
and [Newton instructions](../../robobench/suites/deformable/README.md). Build
worlds on the GPU host, with the interpreter specified for each backend:

```bash
.venv-harbor/bin/python eval/scripts/harbor_benchmark.py build \
  --worlds experiments/harbor-worlds \
  --physx-python .venv/bin/python --newton-python env_newton/bin/python
```

This uses the existing extractor, feature patches, asset repair, and compulsory
boot validation. Descriptions come from the live scene and robot. There is no
`--skip-validation` export path. Worlds must be built with the adapter's
[condition.yaml](condition.yaml) or a compatible explicit `--condition`.
The default grants staged-planning guidance, no additional tools, and freezes
the prescribed controller. To grant tools, supply a condition file; preserve
the same world feature settings and avoid rules that conflict with Harbor's
checkpoint protocol.
The extractor retains required shared suite source (locomotion inherits a
packing scene) while excluding graders and smoke solutions from every suite.

Export with your immutable runtime image references:

```bash
.venv-harbor/bin/python eval/scripts/harbor_benchmark.py export \
  --worlds experiments/harbor-worlds --output experiments/harbor-tasks \
  --physx-image 'YOUR_REGISTRY/physx@sha256:YOUR_DIGEST' \
  --newton-image 'YOUR_REGISTRY/newton@sha256:YOUR_DIGEST' \
  --seeds 0 1 2 --num-envs 1
```

Both commands accept repeatable `--task assembly.bulb` selectors for development
or a declared subset. With no selectors, all 28 are required: missing worlds do
not silently shrink the benchmark. Build refuses existing world directories;
select only the remaining tasks when resuming an interrupted build. Export
refuses to overwrite an existing dataset and publishes the complete dataset
atomically. It verifies each world's boot receipt, content hash, selected
preset, and feature settings, and rejects grader/smoke leakage and symlinks.

The export contains independent `environment/` and `tests/` build contexts with
vendored worlds. It can be moved off the source machine. Assets must already be
present from boot validation; `HF_HUB_OFFLINE=1` disables asset downloads in
evaluation containers. Each task's `export.json` records content hashes,
world provenance, and image references; `dataset.json` records the exact
roster. Keep these with published results. Mutable image tags are accepted only
with the explicit development flag `--allow-mutable-images`.

## Run

For one evaluator-provided GPU, start with a no-op infrastructure check:

```bash
.venv-harbor/bin/python eval/scripts/harbor_benchmark.py validate experiments/harbor-tasks
ROBOBENCH_GPU=0 .venv-harbor/bin/harbor run \
  -p experiments/harbor-tasks/assembly.bulb -a nop -e docker -n 1
```

`nop` leaves no solution and should receive zero. It checks container lifecycle
and artifact transfer, but does not exercise a simulator rollout. Follow it with
an agent-created `solve(env)` and verify the native rollout artifacts below.
Empty-delivery checks do not require credentials for native grader services.

To exercise GPU simulation as well, run the container smoke fixture across the
exported roster:

```bash
ROBOBENCH_GPU=0 .venv-harbor/bin/python eval/scripts/harbor_container_smoke.py \
  --dataset experiments/harbor-tasks --output experiments/harbor-container-smoke
```

This builds and resets each world inside the actual Harbor agent container,
then checks empty-delivery verification in a separate container. It checks the
declared batch size at the first exported seed, runs one task at a time, and
fails on a missing completion marker or any trial exception. It does not call a
coding model or native grader. Native scoring parity is a separate acceptance
check below.

For infrastructure runs partitioned into separate Harbor jobs, validate their
combined coverage without starting Docker again:

```bash
.venv-harbor/bin/python eval/scripts/harbor_container_smoke.py \
  --dataset experiments/harbor-tasks --output experiments/harbor-container-report \
  --job jobs/PARTITION_1 --job jobs/PARTITION_2
```

Supply only completed boot-fixture jobs. Their union must cover every exported
task exactly once, with matching task checksums and agent configuration. Missing
tasks, duplicated trials, interrupted jobs, or missing GPU completion evidence
fail validation; a passing subset cannot produce a full-roster certificate.

Then select any Harbor-supported agent and model using its usual credentials:

```bash
ROBOBENCH_GPU=0 .venv-harbor/bin/harbor run \
  -p experiments/harbor-tasks -a codex -m YOUR_MODEL -e docker -n 1
```

Do not increase `-n` while pinning every container to the same GPU. For multiple
GPUs, partition task selections across independent jobs, set `ROBOBENCH_GPU`
per job, and keep each at `-n 1`. Pin the agent version and model in the lab's
Harbor job configuration for reproducibility. No oracle solution is shipped:
the suite's smoke programs are capability probes and are not valid solutions.

Harbor retains its usual agent trajectory and logs. Verifier artifacts include:

- `results.json`: every checkpoint, deduplication, per-seed scores, and selection.
- `inputs/<submission>/`: the frozen program and data used for replay.
- `rollouts/<submission>/seed_<seed>/`: native verdicts, per-trajectory progress,
  simulator output, exit status, and deformable final-frame audits.
- `progress.json`: completed checkpoints even if a later rollout fails.
- `reward.json` on successful verification, or `error.json` on evaluation error.

Before publishing scores, validate the completed Harbor job against its export:

```bash
.venv-harbor/bin/python eval/scripts/harbor_benchmark.py summarize \
  --dataset experiments/harbor-tasks --job jobs/YOUR_COMPLETED_JOB > summary.json
```

The summary requires exactly one completed trial per declared task by default;
set `--attempts N` for repeated trials, and repeat `--job` when jobs partition
the dataset. It checks Harbor's task checksums, agent/model configurations,
separate verification, protocol equality, and agreement between Harbor and
native verifier rewards. It rejects failures, missing tasks, unexpected tasks,
duplicate trials, and mixed configurations. Agent-budget exhaustion remains
eligible when verification completed successfully. It reports separate macro
means for score and success, plus every task and trial; it never fills gaps
with zero. Resolve failed trials before producing a publication summary.

## Validation

See the [validation record and deployment scope](VALIDATION.md).

The automated tests use the installed Harbor release, not a copied schema:

```bash
uv pip install --python .venv-harbor/bin/python pytest torch numpy
.venv-harbor/bin/python -m pytest tests/test_harbor_support.py -q
```

They load all 28 exported tasks through Harbor, verify separate-verifier
resolution and artifact paths, check deterministic receipts and transactional
failure, stress concurrent checkpoint submission, exercise real child-process
failures/timeouts, and run Docker Compose's actual merge against Harbor's GPU
resource overlay. These checks do not require a Docker daemon or GPU.

Before publishing a benchmark release, run on the evaluator's real GPU Docker
host: build/boot all 28 worlds, build both task and verifier images, run `nop`,
then replay trusted solutions through checkpoint submission and final delivery.
Compare retained native verdicts with the existing grader on the same
solutions/seeds, and confirm verifier failures produce no reward.
CPU contract tests alone do not validate simulation or container isolation.

[`harbor_acceptance.py`](../scripts/harbor_acceptance.py) automates that replay
through the real Harbor Docker backend. It uses a transport fixture agent to
deliver supplied solutions and call the public `submit` command. It checks
non-root execution, read-only task files, hidden-grader separation, transferred
and frozen checksums, checkpoint/final ordering, strict job aggregation, and
native verdict parity **per trajectory**, not just the aggregate score:

```bash
.venv-harbor/bin/python eval/scripts/harbor_acceptance.py \
  --dataset experiments/harbor-tasks \
  --references experiments/references/manifest.json \
  --output experiments/harbor-acceptance
```

The reference manifest has `schema_version: 1` and a `tasks` object keyed by the
export's exact task IDs. Each task has a nonempty `checkpoints` list and a
`final` record. Each record contains the solution-directory `path`, its
`sha256` from `content_hash(inventory(path))`, and `native`, keyed by every
exported seed. Each native record contains a verdict-file `path` and its file
SHA-256. Paths may be relative to the manifest. Keep references outside the
export. Supply independently retained native verdicts for the same runtime,
world, seeds, and trajectory count; failed or timed-out baselines are rejected.
Score comparison is exact by default. An explicit `--score-tolerance` up to
0.01 may accommodate declared numerical variation; success must still match.
A passing run certifies only the supplied reference matrix. It does not establish
the provenance or correctness of those solutions, or coverage of other tasks.

For repeatable **direct GPU replay** diagnostics, there is also
[`harbor_gpu_smoke.py`](../scripts/harbor_gpu_smoke.py). Run it as root on a
disposable evaluation host with a non-root `agent` account, readable exported
worlds, and both simulator interpreters. It checks two checkpoints plus a
duplicate final solution for each selected task using the export's actual seeds
and time limits. All four Newton tasks require the real judge credential.

```bash
.venv-harbor/bin/python eval/scripts/harbor_gpu_smoke.py \
  --dataset experiments/harbor-tasks --output experiments/harbor-smoke \
  --physx-python .venv/bin/python --newton-python env_newton/bin/python
```

Use repeatable `--task` selectors for a smaller diagnostic run. This deliberately
executes simple control programs; its rewards are not meaningful benchmark
performance. It complements, and does not replace, the Docker acceptance run.

Harbor references: [task format](https://docs.harborframework.com/tasks/overview),
[separate verification](https://docs.harborframework.com/tasks/separate-verifier),
[Compose](https://docs.harborframework.com/tasks/multi-container).
