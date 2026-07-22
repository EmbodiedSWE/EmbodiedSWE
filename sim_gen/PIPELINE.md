# sim_gen — scalable construction of MuJoCo simulation tasks

Pipeline for constructing new simulation problems from seed problems, for building a
post-training mix for coding agents. Seeds come from the vendored
[RoboVerse](RoboVerse/) corpus; new problems are standalone **MuJoCo** tasks implemented
in the CoSiGen house style (scene-is-task, config dataclass, teleport-oracle solution (no robot), smoke
test with named checks, rendered video).

```
seed (RoboVerse task file)
   │
   ▼
[1] seed selection ──────────── pipeline/seeds.py (deduplicated pool, ~194 tasks)
   │
   ▼
[2] construction agent ──────── pipeline/spawn_agent.py (Claude Code, trajectory logged
   │                            through super_relay; the agent decides the mutation, but
   │                            must be strategically different from the seed)
   ▼
[3] new task package ────────── tasks/<task_name>/{scene.py, smoke.py, TASK.md}
   │                            + teleport-oracle solution (no robot) inside smoke.py
   ▼
[4] validation ──────────────── pipeline/validate.py (generic physical gates)
   │                            + the task's own smoke battery (ALL PASS required)
   ▼
[5] novelty judgment ────────── pipeline/novelty.py (LLM judge vs. the seed)
   │
   ▼
admitted task + artifacts (video, checks report, agent trajectory)
```

## Directory layout

| Path | What |
|---|---|
| `RoboVerse/` | Vendored seed corpus (declarative task files under `roboverse_pack/tasks/`). Read-only; never imported, only read as source text. |
| `super_relay/` | Vendored logging relay: CC agents point `ANTHROPIC_BASE_URL` at it; it forwards to the real API and logs every request/response for trajectory export. |
| `core/` | The small MuJoCo task framework generated tasks are written against: `scene.py` (BaseScene + SceneCfg + registry), `recorder.py` (H.264 video), `checks.py` (named-check harness). |
| `tasks/` | One package per task: `scene.py` (the environment), `smoke.py` (oracle solution + test cases), `TASK.md` (task card: seed, what changed, why strategically different). `tasks/push_cube_ref/` is the hand-written reference port of the ManiSkill PushCube seed — the format exemplar. |
| `pipeline/` | `seeds.py` (deduplicated seed pool + sampling) · `prompt.py` (construction-agent prompt) · `spawn_agent.py` (spawn CC agent with relay logging) · `validate.py` (generic gates + smoke runner) · `novelty.py` (LLM judge). |
| `artifacts/` | Videos, check reports, relay logs, exported agent trajectories. Gitignored. |

## Stage 1 — seed selection

`pipeline/seeds.py` enumerates the **deduplicated seed pool**: one seed per distinct
task file (~194 manipulation tasks). The raw corpus is over-represented by a few
bulk-variant files (one file registers ~1,589 asset variants of a single task), so
sampling is at the task-file level, never over raw registrations. Passthrough wrappers,
base classes, and locomotion/whole-body families are excluded. `sample_seeds(n, seed)`
draws uniformly from this pool (`python -m sim_gen.pipeline.seeds --list / --sample N`).

A seed is handed to the construction agent as **source text** — the seed defines the
*semantics* to mutate away from; the implementation is rewritten from scratch against
`core/`, so no metasim/RoboVerse dependency leaks into generated tasks.

## Stage 2 — construction agent

`pipeline/spawn_agent.py` spawns one Claude Code agent per (seed, new-task-name):

- The agent reads the seed file, the `core/` framework, and the reference task, then
  designs and implements the new problem. **The agent chooses the mutation freely**
  (objective, objects, mechanism, constraints — any combination), under one hard
  instruction: the new problem must be **strategically different** from the seed — a
  solution to the seed must not be a solution to the new task, and vice versa a solver
  should need a different plan, not different parameters. A difficulty tier
  (easy/middle/hard, the allocation guide below) is passed as a rough target.
- Billing goes through the Claude subscription (OAuth token), not an API key.
- All traffic is routed through the vendored **super_relay** proxy
  (`sim_gen/super_relay/server.py`), which logs every request/response.
  `--export-trajs` exports agent trajectories to
  `artifacts/trajectories/training_trajs.jsonl` (one line per leaf request, reference
  schema). The trajectory is itself a training-data artifact.
- Every run writes `artifacts/reports/<task>.run.json`: seed id, tier, model, the
  agent's session id, exit code, timestamps — the machine-readable link between a
  task, its seed, and its trajectory. Batch construction:
  `spawn_agent.py --batch N --rng-seed S --tier T` (samples N pool seeds; task names
  auto-derived as `<seed>_dK`). Paths/ports are env-overridable for other machines:
  `SIM_GEN_PYTHON`, `SIM_GEN_RELAY_DIR`, `SIM_GEN_RELAY_PORT`, `SIM_GEN_OAUTH_ENV`.

## Stage 3 — the task contract

A generated task is a package `tasks/<name>/` with three files:

- **`scene.py`** — subclass of `core.scene.BaseScene`:
  - `xml()` → MJCF string (the whole model: bodies, joints, geoms, actuators);
  - `reset_instance(rng)` → sample + apply one episode instance (poses, masses, goal);
    randomization must be real (different seeds → different instances);
  - `success()` → bool, `score()` → float in [0,1] — the task's RUBRIC: partial
    progress rewarded in whatever structure the constructor judges reasonable (event
    credits, stage prefixes, shaping, count fractions...). Invariants: 1.0 iff
    `success()`, ~0 for the null policy, and credit doesn't evaporate under correct
    behavior (latch transient achievements). TASK.md declares a rough stage count,
    used only as a difficulty label;
  - `describe()` → the natural-language task statement a solving agent would receive;
  - a `SceneCfg` dataclass holding every tunable init parameter.
- **`smoke.py`** — the teleport-oracle solution (no robot) and the test-case battery (see stage 4).
  The oracle solution manipulates objects directly (teleport + settle via the BaseScene
  helpers) — no robot embodiment at this stage, mirroring how robobench tasks are
  validated scene-first.
- **`TASK.md`** — task card: seed provenance, what was changed, why it is strategically
  different, expected strategy, check list.

## Stage 4 — validation ("physically correct" + "success criteria makes sense")

Two layers, both must pass:

**Generic gates** (`pipeline/validate.py`, identical for every task):
1. build — XML compiles, all names referenced by scene code exist;
2. settle — zero-action rollout: no NaN, kinetic energy decays, objects rest;
3. determinism — same seed → identical trajectory hash; state snapshot/restore round-trips;
4. randomization-is-real — different seeds → different instances (and applied values
   verified by readback, not trust);
5. null policy (do-nothing): `success()` False, `score()` ≈ 0 on every instance;
6. horizon — the oracle solution finishes within the episode budget with margin.

**Task-specific battery** (the task's own `smoke.py`, written by the construction agent):
- the oracle solution reaches `success()` end-to-end on multiple sampled instances
  ("achievable"); it is exported as a callable `oracle_solution(scene)` so the validator
  can re-run it;
- negative controls — deliberately wrong executions that MUST fail ("refusable").
  Two are REQUIRED, not just examples:
  - **the seed's strategy, executed in the new scene, must fail** (this is the
    executable half of "strategically different"; if the seed strategy is not even
    expressible in the new scene, document the N/A in TASK.md);
  - tasks may require a specific execution order or not — **both are legitimate
    designs**; TASK.md states which. For order-requiring tasks only: **wrong-order
    execution must fail** (the certificate that the order requirement is real);
  plus task-specific ones (wrong object, near-miss outside tolerance, ...);
- rubric monotonicity — score is non-decreasing as the oracle solution progresses through
  stages, spread over [0,1], max exactly at success;
- a calibration probe where tolerances matter (offset sweep → the pass/fail knee is
  where the task intends it).
- Every check is a named `check(name, cond)`; the smoke's verdict is the conjunction
  (`ALL PASS n/n`), and **every run records a video** (H.264, artifacts dir).

**Post-smoke gate** (`pipeline/validate.py` again, re-running the exported
`oracle_solution` under the validator's own instrumentation):
- **success persistence** — after success first triggers, keep simulating: it must not
  flicker off (no single-frame "success"; this also rejects success triggered by
  transient fly-through states).

## Stage 5 — novelty ("strategically different")

`pipeline/novelty.py` runs an LLM judge (through the same relay) over: the seed's source
+ the new task's `TASK.md`, `describe()`, and oracle-solution strategy. Verdict JSON:
`{strategically_different: bool, shared_strategy: str, reasoning: str}`. Judged
**within the seed family only** (checking against the global pool is out of scope;
cross-family duplication is handled as mix bookkeeping, not a per-task gate). Where two
variants of the same seed both exist in `tasks/`, the executable cross-check also
applies: variant A's oracle solution must not pass variant B.

## Difficulty distribution (guide only)

For mix construction we **roughly take difficulty = stage count / horizon**. This is
only one axis of difficulty — precision tolerances, hidden state, irreversibility etc.
also matter — but stage count is the axis we can label mechanically, so it is the one
the mix is balanced on.

**Not a naive stack of stages.** Stage count offers a rough idea of difficulty; it is
NOT a mandate to build tasks as mechanical concatenations of sub-goals, and implementers
should not stick to it inflexibly. In particular:
- a task with few stages can be legitimately middle/hard on other axes (precision,
  hidden state, irreversibility) — say so in TASK.md and tier it accordingly;
- unordered "breadth" tasks (place all N items, any order) are legitimate and can be
  long-horizon without an ordered chain; the wrong-order control simply doesn't apply
  to them (they don't declare an ordered chain);
- stages in real tasks often entangle (later stages constrain how earlier ones must be
  executed), which is part of what makes a task genuinely hard — don't flatten that
  away to make the stage count legible.

**The label is the constructor's declared stage count** (in TASK.md, alongside the
rubric that encodes those stages — the smoke's rubric-monotonicity checks already
verify the score actually steps through them). Quota-driven stage inflation is handled
by review sampling, and ultimately by the deferred solver-based difficulty probe — not
by extra per-task machinery.

**Guide allocation, not a hard budget** (per 1000 tasks):

| Tier | Stages (rough) | Guide count |
|---|---|---|
| easy | 1–2 | ~300 |
| middle | 3–4 | ~400 |
| hard | 5+ | ~300 |

These numbers are a guide; exact ratios are not load-bearing, and the trained-mix
distribution can be reweighted at rollout-sampling time without rebuilding tasks.

## Stage 6 — robot embodiment bindings (required; the trainable surface)

The teleport-oracle solution (no robot) validates the *scene*; the training data is a solving agent
writing code against the CoSiGen-loop-style API (`move_to`, `set_gripper`,
`get_state`, ...) that drives a **robot arm** in the scene. So every admitted task gets
embodiment bindings before it enters the mix:

- **Arm embodiments, as in CoSiGen** — Franka first, then the other arm types
  (bimanual Franka, Piper, WXAI...). Humanoids are skipped for now.
- Each binding = robot model placed in the task scene (base pose, table height) +
  the solving-agent API implemented on MuJoCo (IK-backed `move_to`, gripper control),
  behind the same API surface the CoSiGen harness exposes on Isaac.
- Each binding is validated by its own smoke: boot, reach the task's manipulands, and
  a **scripted robot solution of the task**, with video.
- **Mix membership requires the scripted robot solution.** A task whose scene passes
  every gate but has no robot solve is NOT admitted to the mix or counted toward tier
  quotas — scene-solvable does not imply arm-solvable (gripper aperture, occluded
  grasps, reachability). An arm-unsolvable task wastes rollout compute (zero reward
  variance = zero learning signal) and is an embarrassing artifact in a public dataset.
  Such tasks stay in `tasks/` as staging until a robot solve lands. The scripted solve
  doubles as a solvability certificate shipped with the task and as the reference
  oracle for the future difficulty probe.
  Known trade-off, accepted deliberately: this requirement biases the pool toward
  scriptable tasks (scripts may use privileged scene state, which softens but does not
  remove the bias). Revisit if the mix looks starved of feedback-heavy tasks.

`core/robots/` hosts the robot layer; bindings live next to the task
(`tasks/<name>/binding_smoke.py`).

## Deferred (deliberate, not missing)

- **Difficulty measurement / reference-solver panel** — solver-calibrated difficulty
  (pass-rate panels) is deferred per plan; the structural difficulty label (stage count,
  above) is already implemented via the rubric. Every admitted task exposes
  `describe()` + `score()`, which is all the future difficulty probe needs.
- **Scaling layers above the single-task unit** — this doc specifies the pipeline for
  ONE task. Producing the full mix additionally needs (from the agreed design, not yet
  implemented): skill-taxonomy tagging + per-family quotas when composing the released
  mix, and batch-level human acceptance sampling (review a stratified sample per
  generated batch; a bad sample quarantines the batch and fixes the generating recipe).
  The acceptance sample is also where the **adversarial checker probe** runs: a second
  agent prompted to make `success()` return True *without* doing the described task —
  per-batch, not per-task, which is where it's cheap.

## Running: one agent, one problem, worked until accepted

The operating model is **per-problem, not per-stage**: each construction agent owns
exactly ONE problem and drives it through the whole lifecycle below, iterating until
the problem is ACCEPTED. Do NOT run one stage across all problems before the next
stage. Scaling up = running many single-problem lifecycles concurrently (agents share
one relay; concurrent sessions are separated automatically).

**Setup, once per machine** (not per problem):

```bash
cd <CoSiGen checkout>   # requires: mujoco, numpy, imageio-ffmpeg, claude CLI + OAuth
                        # env overrides: SIM_GEN_PYTHON / SIM_GEN_RELAY_DIR /
                        #                SIM_GEN_RELAY_PORT / SIM_GEN_OAUTH_ENV
python sim_gen/pipeline/spawn_agent.py --start-relay-only   # idempotent
```

**Lifecycle of ONE problem** (`<task>` is auto-derived as `<seed>_dK`):

```bash
# 1. spawn the construction agent on a seed (samples: python -m sim_gen.pipeline.seeds --sample 1)
python sim_gen/pipeline/spawn_agent.py --seed <family/stem> --tier <easy|middle|hard>
#    the agent designs the task, implements scene.py / smoke.py / TASK.md, and
#    iterates until its own smoke prints "SIM_GEN_SMOKE: ALL PASS"

# 2. validate — generic gates + the task's smoke re-run + persistence gate
python sim_gen/pipeline/validate.py --task <task>          # must end "PASS"

# 3. novelty judgment vs. the seed (seed read from the task's run.json)
python sim_gen/pipeline/novelty.py --task <task>           # must end "PASS"

# 4. Franka binding with a scripted robot solution (the mix-admission bar)
MUJOCO_GL=osmesa python -m sim_gen.tasks.<task>.binding_smoke \
    --video sim_gen/artifacts/<task>_franka.mp4            # must end "ALL PASS"
```

Any failing step = fix the root cause (scene, smoke, bindings — never weaken a check)
and re-run that step. **Accepted** = steps 2, 3, 4 all pass. Only accepted problems
count toward the mix and the tier quotas; everything else stays in `tasks/` as staging.

**Anytime** (covers all sessions logged so far):

```bash
python sim_gen/pipeline/spawn_agent.py --export-trajs   # -> artifacts/trajectories/
```

`spawn_agent.py --batch N --rng-seed S --tier T` is a convenience that runs N of these
constructions sequentially; for real scale, launch N concurrent single-problem runs.
