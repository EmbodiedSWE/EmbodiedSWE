# Harbor integration validation — 2026-10-22 UTC

This records integration checks, **not benchmark task-solving performance**.
The native rubric was retained. No trusted successful solution set was supplied.
The Harbor infrastructure acceptance checks passed for the tested profiles below.
The deployment boundary and native grading trust model remain explicit; this is
not a certification of task-solving performance or adversarial security.

## Initial GPU environment

The GPU checks used a temporary Ubuntu 24.04 host with one NVIDIA RTX 4090
(24 GB), driver 595.91.07. Hardware rental was confined to validation; the
integration requires evaluator-provided GPUs and contains no provisioning code.

| Component | PhysX | Newton |
| --- | --- | --- |
| Python | 3.11 | 3.12 |
| PyTorch | 2.7.0+cu128 | 2.11.0+cu130 |
| Isaac Sim | 5.1.0.0 | 6.0.0.1 |
| Isaac Lab | 2.3.2 | 7.0.3 |
| Warp | 1.11.0 | 1.16.0 |

PhysX also used Pinocchio 2.7.0, Pink 3.1.0, DAQP 0.8.5, and NumPy 1.26.0.
Newton used Isaac Lab commit `d7d004217c60b4790f721565bf5d40243addcb0e`
with this repository's `scripts/isaaclab_newton16_compat.patch`, Newton commit
`f420998186ec70bc39323ccc374bcb6c2be1d14f` (1.6.0.dev0),
newton-usd-schemas 0.6.0, and MuJoCo/MuJoCo Warp 3.11.0.

## Checks

- **117 tests and 2 subtests passed** with `python -m pytest tests -q` on the
  development host. Tests use installed Harbor 0.24.0, including its task and
  result models, separate-verifier resolution, and Docker Compose's actual
  resource merge. Compose 2.38.2 was used; a daemon is not needed for this check.
- All **28 extracted worlds built and reset** on the GPU, including the four
  Newton worlds. The complete real export passed Harbor parsing and content
  validation on Linux. This initial Runpod export used placeholder image tags; actual image builds
  and container checks are covered separately in the Modal validation below.
- All **24 PhysX checkpoint/final-delivery smoke checks passed**: 48 native
  grading rollouts at seed 0, plus four additional bulb rollouts at seeds 0 and 1.
  None of these 52 rollouts timed out or reported a solution error; the action
  programs advanced the native simulator. Outcomes are recorded in
  [`validation-results.json`](validation-results.json). These programs do nothing
  or take one robot action; they test grading, selection, and final deduplication.
- All **four Newton final-frame captures passed**. The images were inspected.
  A deliberately injected judge transport failure after real rendering was
  rejected by the verifier. These diagnostic runs produced no accepted reward;
  no live judge call or substitute judge decision was used.

The CPU stress tests include 32 concurrent submissions, atomic exports,
checkpoint corruption and symlinks, fresh per-seed working copies, exact decimal
ties, subprocess exits and hangs, stale rewards, nonfinite or inconsistent
verdicts, missing judge credentials, storage failures, and incomplete/mixed
Harbor job summaries. The summary tests ensure infrastructure failures cannot
silently become benchmark zeros through Harbor's default mean aggregation.

GPU checks exposed and fixed missing cross-suite source dependencies in
locomotion, merging a Git-tracked manifest into a partially downloaded asset
directory for dicing, private export-directory permissions, and owner-only
downloaded asset permissions. Regression tests cover those failures. Direct
GPU replay reproduces the Dockerfile's writable asset cache in a disposable
world copy, leaving the exported content receipts intact.

Session logs, native verdicts, submission inputs, and rendered frames are kept
locally under `experiments/harbor-validation/gpu-20261022/`, which is excluded
from Git. The reproducible direct replay command is
[`harbor_gpu_smoke.py`](../scripts/harbor_gpu_smoke.py); see the
[integration guide](README.md) for its prerequisites and invocation.

## Modal container validation

A second validation host used an NVIDIA L4, driver 610.57.04, Docker 28.3.3,
Compose 2.39.2, and NVIDIA Container Toolkit 1.20.1. Both simulator runtime
images were built locally. All 28 worlds built/reset again, and the final
28-task export passed receipt validation and parsing with Harbor 0.24.0.

The final adapter passed real Harbor reference-replay jobs for `assembly.bulb`
(two trajectories per seed) and `assembly.allen_bolt` (one trajectory per seed),
each at seeds 0 and 1. Both delivered two checkpoints and a duplicate final
solution through the public submission CLI. All eight actual Harbor rollouts
matched eight independently executed native rollouts exactly, including each
trajectory's score and success. The bulb case exercised the earliest-checkpoint
tie rule; allen_bolt selected checkpoint `000002` with score 0.0002. These are
transport/physics probes, not successful task solutions. Hidden files, empty
directories, executable flags, frozen inputs, duplicate detection, agent user,
grader separation, and strict result aggregation were checked in the real job.
A separate `nop` job also completed with zero reward and no exception.

This run exposed missing write permissions for Kit's installation-local
`cache`, `data`, and `logs` directories. The exporter now prepares them for
non-root execution. The reference agent checks those permissions, and both
PhysX jobs above passed after the fix. Empty deliveries now receive their
specified zero without requiring native grader credentials; valid submissions
still require the native grader's dependencies before their rollouts start.

Nested Docker in Modal's gVisor requires **host networking**. The test export
explicitly changed both agent and verifier Compose files and regenerated their
content receipts, preserving the original and tested hashes. The normal export
keeps Docker's default networking. This validates GPU/CDI and Harbor's separate
container lifecycle under that declared networking configuration; it does not
certify production bridge/egress policies. Runtime image IDs and results are
retained with the local validation artifacts; no image was published.

An earlier Newton container attempt was **incomplete**. That captured A10 run
used Python 3.12.1, matching the native environment, a 64 GiB sandbox, and
declared 32 GiB task limits. `deformable.dumpling` completed GPU build/reset and
empty-delivery verification in a separate Harbor container without an exception.
`deformable.knot` also completed GPU build/reset, but its verifier had not
completed in the last saved snapshot. Latte and T-shirt did not run in that
attempt. These empty deliveries exercise infrastructure, not native scoring.

Two A10 sandboxes ended with exit 137; Modal's API explicitly reported
**GPU preemption** for both. An earlier interpretation involving memory pressure
was not confirmed and is not a finding. The earlier A10 attempt reached Warp
kernel hashing in Latte before preemption. An initial Newton image on L4 also
produced a Kit crash; its cause was not established. None of these incomplete
attempts is counted as a passing four-task matrix.

The final Newton trial artifacts were saved incrementally and checked against
the exported protocol, reward, completion marker, and 16 checkout source hashes.
They are retained under `experiments/harbor-validation/modal-20261022/`, along
with provider termination reasons and cleanup receipts. Both `main` and
`datagen` were checked after cleanup: zero active apps, containers, or sandboxes.
The temporary Modal credential file was removed. The earlier Runpod host was
also deleted.

## Continuation: standard Docker checks

A Modal CPU VM running Linux 7.2.9, Docker 28.3.3, and cgroup v2 passed a
separate host-compatibility probe through Harbor 0.24.0. It used the adapter's
unchanged Compose template with an explicit CPU-only overlay removing the GPU
device, and a Python image pinned by registry digest. The checks confirmed:

- Harbor's CPU and memory settings reached both Docker's configuration and the
  container's actual cgroup files.
- Three busy processes under a one-CPU limit caused 31 throttled periods.
- Allocating 256 MiB under a 128 MiB limit terminated the worker with exit 137;
  the kernel recorded an OOM kill.
- Docker bridge networking, peer name resolution, peer HTTP access, and public
  HTTPS egress worked without host networking.

This verifies real bridge and cgroup behavior on a CPU VM. It does not combine
these checks with GPU execution on the same host: Modal GPU Sandboxes use
gVisor, while its VM runtime is CPU-only. The raw result, runnable probe, package
lists, immutable simulator-image references, and subsequent GPU records are
retained under `experiments/harbor-validation/continuation-20261022/`.

The container-smoke checker now accepts partitioned Harbor jobs using repeated
`--job` arguments. Acceptance still requires the complete exported roster,
exact task checksums, the declared boot fixture, successful GPU completion,
and empty separate verification for every task. Tests reject missing or
duplicated partitions, missing GPU evidence, and other agents' results.

## Completed 28-task Harbor acceptance

A subsequent NVIDIA A10 run completed **all 28 actual Harbor trials** with no
trial exception: all 24 PhysX tasks and all four Newton tasks. Each trial built
and reset its exported world as the non-root agent, produced the exact GPU
completion marker, and finished empty-delivery verification in a separate
container. The full-roster checker accepted their union with exact task
checksums and the declared fixture identity; no subset or interrupted trial was
counted as a pass. The boot matrix made **zero native-grader or visual-judge calls**.

Both runtime images were pinned by registry digest. They used Python 3.11.17
(PhysX) and Python 3.12.1 (Newton), with the simulator versions recorded above.
The GPU host used driver 610.57.04, Docker 28.3.3, Compose 2.39.2, a 48 GiB memory
request and 96 GiB limit. Tasks declared 32 GiB. The immutable image references,
individual trial IDs/checksums, and 30 verified checkout source hashes are in
[`validation-results.json`](validation-results.json).

Fresh reference jobs on those pinned images also passed for `assembly.bulb`
(two trajectories) and `assembly.allen_bolt` (one trajectory), at seeds 0 and 1.
**Eight Harbor rollouts matched eight independent native rollouts exactly**, with
zero per-trajectory score delta and no tolerance. Each job submitted two
checkpoints and a duplicate final delivery, exercising hidden files, empty
directories, executable flags, immutable inputs, deduplication, and selection.
These remain no-op/one-action transport and physics fixtures.

The A10 run used the declared Modal host-network override. Separately, the CPU VM
checks above verified the unchanged production Compose template's bridge
networking and real CPU/memory enforcement. These are distinct host profiles;
GPU execution and production bridge/cgroup policies were not jointly tested on
one machine. Run the supplied container smoke and reference acceptance commands
on the evaluator's configured GPU host before publishing its results.

Test-host recovery included an OOM during initial image extraction, before any
trial started. Pre-unpacked image snapshots avoided that extraction. New Docker
layers used its native tmpfs storage, with base layers linked from the saved
cache, avoiding slow fallback difference scans on snapshot mounts. The runtime
image identities and task contents were preserved. A native reference attempt
with a root-owned shared log directory produced no verdict and was rejected;
after correcting the test-host directory permissions, the references passed.
These temporary test-host arrangements are confined to validation; the shipped
adapter contains no Modal or Runpod provisioning.

The final evidence was downloaded and independently audited locally against the
catalog, receipts, completion markers, native/Harbor rewards, and checkout
source hashes. Logs, original/tested network receipts, recovery records, package
lists, and runnable validation helpers are retained under
`experiments/harbor-validation/continuation-20261022/` (Git-ignored).
All validation compute, the temporary volume, and five owned snapshot images
were removed; the temporary credential file was deleted. Final inventory counts
and the evidence archive SHA-256 are in `validation-results.json`.

## Scope and evaluator deployment checks

A reviewed successful/unsuccessful solution matrix for all tasks is a separate
benchmark-quality exercise, not a prerequisite for this Harbor infrastructure
acceptance. The probes establish transport and score parity, not all-task
solvability. A historical allen-bolt solution exceeded the declared 600-second
diagnostic budget on the L4 and was excluded from parity evidence; a historical
successful score was not treated as a fresh pass.

Live visual-judge evaluation is outside this Harbor integration validation scope.
The adapter preserves the existing native grader; no replacement judge was added.

The existing in-process Python grading trust model remains. Separate verifier
containers prevent agent-phase filesystem changes from becoming the grading
environment; they do not make submitted Python adversarially tamper-proof.
