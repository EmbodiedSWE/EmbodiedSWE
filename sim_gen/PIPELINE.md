# sim_gen — construction of simulation tasks

Pipeline for constructing new simulation problems from seed problems, for building a
post-training mix for coding agents. Seeds come from the vendored
[RoboVerse](RoboVerse/) corpus; new problems are tasks implemented in the CoSiGen house
style (scene-is-task, config dataclass, **teleport solution**, rubric battery,
rendered video). One construction session owns one problem end to end — scene,
teleport solution, rubric, checks — under a **1-hour wall-clock budget**.

## Simulation backend: Isaac Lab (decision 2026-07-22)

Generated tasks target **Isaac Lab / PhysX, authored as robobench suite tasks** — the
same framework and conventions as the existing CoSiGen tasks (`../robobench/`:
`scenes/` + `configs/` + `smokes/`). The construction agent writes a robobench scene;
everything else in this pipeline (seed sampling, strategic-difference requirement,
agent spawning + trajectory logging, validation gates, novelty judgment) is
backend-independent.

Two backend notes:
- **Newton / MuJoCo-Warp**: Newton 1.0 integrates with Isaac Lab 3.0 as a swappable
  physics backend, but that integration is experimental (develop branch, limited
  feature/robot coverage). Our stack needs custom robots, OSC controllers, cameras,
  and state snapshot/restore — not guaranteed there yet. Decision: generate on PhysX
  now; authoring stays at the robobench layer, so a later Newton switch is a backend
  swap, not a rewrite.
- **`legacy/` is the retired MuJoCo variant**, kept as reference only; its task
  contract (MJCF `xml()` etc.) does NOT apply to generated tasks.

```
seed (RoboVerse task file)
   │
   ▼
[1] seed selection ──────────── pipeline/seeds.py (deduplicated pool, 194 tasks)
   │
   ▼
[2] construction agent ──────── pipeline/generate_batch.py spawns one agent per
   │                            (seed, attempt); the agent designs the scene, writes a
   │                            TELEPORT solution that reaches the goal on its GPU
   │                            forge, then writes the rubric from the demonstrated
   │                            solution (1 h budget)
   ▼
[3] task package ────────────── tasks/<name>/{scene.py, solve.py, smoke.py, TASK.md}
   │
   ▼
[4] validation ──────────────── orchestrator re-runs smoke (ALL PASS) AND solve
   │                            (verified success) on the forge; agent's word is not
   │                            trusted
   ▼
[5] LLM judges ──────────────── novelty vs. the seed (pipeline/novelty.py) +
   │                            solution legitimacy + description clarity
   │                            (pipeline/judges.py)
   ▼
admitted task + artifacts (video, checks report, solve trajectory, agent trajectory)
```

## Setup (a fresh clone needs both)

```bash
# 1. the seed corpus is a SUBMODULE (public repo, ~600 MB) — without it stage 1
#    samples an empty pool:
git submodule update --init sim_gen/RoboVerse

# 2. the execution layer is cluster-specific: scripts/launch_cosigen_render_pool.py
#    submits Arnold/mlx jobs whose forges register their URLs to HDFS, and
#    isaac/forge_server.py runs the Isaac build at /home/tiger/isaaclab_build.
#    On another cluster, replace those two with an equivalent that (a) starts
#    forge_server.py next to an Isaac Lab install and (b) publishes each forge URL
#    where pipeline/generate_batch.py's forge_urls() can read it. Everything above
#    the forge (seeds, prompt, acceptance, judges, ledger) is substrate-independent.
```

## Directory layout

| Path | What |
|---|---|
| `RoboVerse/` | Seed corpus, a git submodule (`.gitmodules` -> RoboVerseOrg/RoboVerse): declarative task files under `roboverse_pack/tasks/`. Read-only; never imported, only read as source text. |
| `super_relay/` | Vendored logging relay: CC agents point `ANTHROPIC_BASE_URL` at it; it forwards upstream and logs every request/response for trajectory export + cost accounting. Runs in auth-passthrough mode (client's own credentials) or relay-owned-key mode (`SUPER_RELAY_API_KEY` + `--force-model`). |
| `legacy/` | The retired MuJoCo variant (was `legacy_mujoco/`) — reference only, see its README. |
| `isaac/` | `forge_server.py` — the per-agent GPU forge: an HTTP service on a warm L20 pod that runs a task package's modules as fresh Isaac subprocesses (submit / run / fetch). Deployed via `scripts/launch_cosigen_render_pool.py --forge N`. |
| `tasks/` | One package per task: `scene.py`, `solve.py` (teleport solution), `smoke.py` (rubric battery), `TASK.md` (task card). |
| `pipeline/` | `seeds.py` (deduplicated seed pool + sampling) · `prompt.py` (construction prompt) · `forge_client.py` (agent<->forge CLI) · `generate_batch.py` (campaign orchestrator: parallel agents, count quota, cost ledger) · `novelty.py` (novelty judge) · `judges.py` (solution-legitimacy + description-clarity judges). |
| `artifacts/` | Videos, check reports, relay logs, exported agent trajectories. Gitignored. |

## Stage 1 — seed selection

`pipeline/seeds.py` enumerates the **deduplicated seed pool**: one seed per distinct
task file (~194 manipulation tasks). The raw corpus is over-represented by a few
bulk-variant files, so sampling is at the task-file level, never over raw
registrations. Passthrough wrappers, base classes, and locomotion/whole-body families
are excluded.

A seed is handed to the construction agent as **source text** — the seed defines the
*semantics* to mutate away from; the implementation is rewritten from scratch, so no
metasim/RoboVerse dependency leaks into generated tasks.

## Stage 2 — construction agent

`pipeline/generate_batch.py` runs the campaign: one worker per GPU forge, each worker
spawning one Claude Code agent per (seed, attempt):

- The agent reads the seed file and the reference tasks, then designs, implements,
  and **solves** the new problem in one session. **The agent chooses the mutation
  freely** (objective, objects, mechanism, constraints — any combination), under one
  hard instruction: the new problem must be **strategically different** from the seed
  — a solver should need a different plan and a different code structure, not
  different parameters or minor changes.
- **The solution is a teleport solution** — written as if any object could be
  teleported anywhere, which is far cheaper to write than a real-robot solution while
  still certifying the task: teleportation handles only **transport**, and every
  **load-bearing interaction** still goes through the simulator's contact dynamics
  (to thread a nut onto a bolt, the solution teleports the nut to just above the
  bolt, then presses and twists it down the thread with applied forces until the
  scene reports success). A working teleport solution certifies that the interactions
  the task requires are physically achievable in the scene.
- **Order of work is fixed: solution before rubric.** The agent first reaches the goal
  state with the teleport solution, then writes `success()`/`score()` anchored in the
  demonstrated solution (see stage 3). A rubric written before any solution exists is
  guesswork about feasibility — that is what this ordering eliminates.
- **1-hour cutoff per attempt**, enforced by the orchestrator (`--agent-timeout`).
  The agent is deliberately not told the budget — it is an empirical operating
  number, not design guidance.
- All traffic is routed through **super_relay**, which logs every request/response;
  the trajectory is itself a training-data artifact. Every attempt is recorded in
  `artifacts/campaign/ledger.jsonl` (seed, accepted/reason, minutes, token usage,
  USD estimate); `artifacts/campaign/state.json` keeps running totals.

## The embodiment (design for it from the start)

Every task must be solvable by a **single Franka arm with a parallel-jaw gripper**
driven through the robobench OSC controller. The teleport solution certifies the
scene's required interactions are physically achievable, but it does not exercise the
arm — so beyond the solution working, the construction agent must make sure the
Franka can actually manipulate the objects in the way the task requires, and argue it
in TASK.md (the embodiment argument: per manipulated object, the intended contact
strategy, plus one plausible base pose):

- every object the robot must move needs an intended contact strategy up front: a
  graspable feature that fits the jaw with room for the hand to approach, or a
  pushable face;
- required precision must stay within what closed-loop arm control can reliably hit;
  tolerances near the control noise turn a sound design into a lottery;
- clearances are where solutions die: contacts very near the ground, under low
  overhangs, or through apertures barely larger than the object;
- the action must sit within comfortable reach of the stated base pose.

This is not a mandate for trivial tasks — mechanisms (interlocks, counterweights,
ordered fixtures) are welcome — the requirement is that every contact the task
*requires* is one the arm can actually make.

## Stage 3 — the task contract

A generated task is a package `tasks/<name>/` with four files:

- **`scene.py`** — robobench-format scene module:
  - a `SceneCfg` dataclass holding every tunable init parameter, with real
    randomization (different seeds → different instances, verified by readback);
  - a `BaseScene` subclass implementing `assets()` / `reset(env_ids)` / `get_state` /
    `set_state` / `describe()`, plus `success()` and a graded `score()`;
  - `describe()` is the task statement a solving agent receives: it must state the
    goal, how targets are identified visually, and any ordering constraints —
    complete enough that following only `describe()` can solve the task;
  - `instruction()` is the short form for VLA training: one or two imperative
    sentences (< 200 tokens) stating the goal and the constraints whose violation
    fails the task;
  - registered scene-level (`robot="null"`); `solve.py` and `smoke.py` build the
    same scene-level env.
- **`solve.py`** — the **teleport solution** and the task's legitimacy certificate:
  - builds the scene-level env (`robot="null"`; the arm strategy lives in TASK.md
    as the embodiment argument);
  - teleportation handles **transport only** (setting poses to move objects across
    free space); every **load-bearing interaction** the task requires (insertion,
    threading, pressing, latching...) is executed through contact dynamics, with
    applied forces/torques as the tool — never teleporting an object into a state
    that bypasses the interaction;
  - must reach the goal state with everything settled, print scene readouts,
    `SIM_GEN_SCORE <value>` at each phase boundary (the rubric's latched credit must
    never decrease along the solution trajectory — checked at acceptance); after
    `success()` first turns True it keeps simulating ≥ 3 more simulated seconds with
    no further intervention, and only if success still holds prints exactly
    `SIM_GEN_SOLVE: SUCCESS` (the acceptance marker; the persistence window rejects
    fly-through successes); must pass on at least 2 seeds in the agent's own testing;
  - hard exit after the verdict (Kit teardown hangs).
- **`smoke.py`** — **rejection tests for the rubric** (teleported probe states; not a
  solution of any kind). A successful solve only proves the rubric *accepts* correct
  outcomes; smoke proves it *rejects* wrong ones. For every outcome the rubric claims
  to reject, construct that outcome as a settled state and assert rejection:
  - the end state of the seed's strategy → `success()` False (the executable half of
    "strategically different"; documented N/A in TASK.md if inexpressible);
  - a settled near-miss just outside each load-bearing tolerance → `success()` False;
  - if a required execution order is declared: an out-of-order end state → fails;
  - any other wrong outcome the task claims to reject (wrong object, wrong place...).
  Plus generic sanity (until these move to a shared harness): null policy scores ≈ 0,
  randomization differs across seeds (verified by readback), settle/no-NaN. Records
  video `frames.npz`; prints `SIM_GEN_SMOKE: ALL PASS n/n`.
- **`TASK.md`** — task card: seed provenance, what changed, why strategically
  different, the teleport-solution outline (phases), the embodiment argument (per
  manipulated object the intended Franka contact strategy, plus one plausible base
  pose), whether execution order is required, and the check list.

**Rubric after solution.** `success()` starts as a minimal goal predicate so the
agent can iterate `solve.py`; the final rubric is written only after the solution
works, anchored in it: latch the stages the demonstrated solution actually passes
through, `~0` for the null policy, `1.0` iff `success()`, credit that doesn't
evaporate under correct behavior. The `SIM_GEN_SCORE` prints along the final solve
run are the monotonicity evidence; the acceptance gate checks them.

**Post-acceptance packaging.** `solve.py` is the answer key: after acceptance it moves out
of the task package into `sim_gen/solutions/<task>/solve.py`, so a shipped/evaluated task
directory never contains its own solution. (TASK.md keeps a solution *outline*; strip it
too when packaging tasks for solving agents.)

## Stage 4 — validation

The orchestrator trusts nothing the agent reports: everything is re-run on the forge
from the files on disk. The checks, and where each runs:

1. **rejection tests** — re-run `smoke.py`: every bad state correctly rejected
   (`SIM_GEN_SMOKE: ALL PASS` required);
2. **null policy** — scores ~0 on the crafted rubric (a check inside the smoke
   battery, re-run by gate 1);
3. **randomization + stability** — scene randomization actually varies across seeds
   (verified by readback) and the scene settles without numerical blow-ups (also
   inside the smoke battery);
4. **score monotonicity** — re-run `solve.py` fresh: `SIM_GEN_SOLVE: SUCCESS`
   required, and the `SIM_GEN_SCORE` prints along the run must be non-decreasing
   (latched credit never evaporates along the teleport-solution trajectory);
5. **success persistence** — the environment keeps simulating after success first
   triggers and success must hold through the window (≥ 3 simulated seconds, no
   intervention) before `solve.py` may print the success marker — rejects
   single-frame and fly-through successes.

## Stage 5 — LLM judges

Three independent judge calls, all through the same relay; all must pass:

- **novelty** (`pipeline/novelty.py`) — is the task strategically different from the
  seed? Judged over the seed's source + `TASK.md` + `scene.py` semantics + `solve.py`.
  Within the seed family; where two variants of the same seed both exist in `tasks/`,
  the executable cross-check also applies: variant A's solution must not pass B.
- **solution legitimacy** (`pipeline/judges.py`) — does the teleport solution
  genuinely solve the task rather than exploiting a loophole: teleports carry
  transport only, every load-bearing interaction runs through contact dynamics, no
  teleporting past a required interaction, no pinning objects against physics, no
  rubric loopholes.
- **description clarity** (`pipeline/judges.py`) — could a competent solver perform
  the task from `describe()` alone: goal state, how targets are identified, and any
  ordering constraints all stated.

## Running: the campaign

The operating model is per-problem: each construction agent owns ONE problem on ONE
GPU forge for one 1-hour attempt; the orchestrator respawns finished workers with
newly sampled seeds until the count quota is met.

```bash
cd <CoSiGen checkout>

# 1. GPU forges (one per parallel agent; L20; register to hdfs .../simgen_forge/)
python ../scripts/launch_cosigen_render_pool.py --forge 10

# 2. campaign relay. Two auth modes:
#    passthrough (agents' own credentials; relay only logs):
python sim_gen/super_relay/server.py --port 8119 \
    --log-dir sim_gen/artifacts/relay_logs_campaign &
#    relay-owned key (platform billing; model forced at the relay):
SUPER_RELAY_API_KEY=<key> python sim_gen/super_relay/server.py --port 8119 \
    --log-dir sim_gen/artifacts/relay_logs_campaign \
    --upstream-base <gateway>/v1 --force-model <model> &

# 3. the campaign: count quota + parallel agents + respawn + acceptance + cost ledger
#    (SIM_GEN_CAMP_DIR selects the campaign state dir; its ledger is replayed on resume)
SIM_GEN_CAMP_DIR=sim_gen/artifacts/campaign_v2 \
python sim_gen/pipeline/generate_batch.py --count 50 --workers 10

# live status / spend:
cat sim_gen/artifacts/campaign/state.json
```
