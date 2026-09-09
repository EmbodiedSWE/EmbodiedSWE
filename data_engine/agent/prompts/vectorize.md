# Vectorize session — make the solve drive parallel envs

You are inside a **data_gen campaign** that multiplies one verified robobench
solve into a large demonstration dataset. Dataset throughput comes from running
multiple environments in one simulation: `generate --num_envs {num_envs}` steps
{num_envs} worlds together, each with its own sampled parameters, and grades
every episode. The current wide probe failed the gate: the batch must complete
env-batched with at least one success WHOSE RECORDED ACTIONS REPLAY at that
width (last probe: {successes} successes, {replayed} of them replay). A batch
that crashes at build, or whose log ends without `DONE`, may also have been
KILLED by the OOM killer (the recorder keeps every env's state in RAM: 512 envs
x 20k steps exceeds 60 GB) — the log's last lines say which. Diagnose that failure
from the solve and the probe log, then make the solve correct for
independently varying environments. Common causes include single-row state
reads, targets derived from one environment and broadcast to all rows, a
shared phase clock for environments with different progress, and batch-unsafe
helper state. Do not assume which cause applies before reading the evidence.

**This is a hard gate. The campaign does not move on until it passes.** If
your session ends with the wide probe still failing, the next vectorize
session starts from your edits, until the pre-check budget ({hours_left} h
left) is exhausted — then the whole campaign is abandoned as PRECHECK_FAILED.
Do not stop at a diagnosis or a partial lift: change the solve, run the probe,
repeat. If a `VECTORIZE_NOTES.md` already sits next to the solve, read it first
and build on it rather than repeating what it records as tried.

## Facts

- campaign root (your cwd, writable): `{gen}`
- the solve to vectorize (edit IN PLACE — this is the base every future cell
  copies): `{solve}`
- a GPU is available; `python` has Isaac Sim + Isaac Lab.

## Failing wide-run evidence

The COMPLETE failing-probe log ({fail_log_size}) is at:

    {fail_log_path}

Read it with your shell (grep/tail/sed — it is a full {num_envs}-env Isaac run,
so search it rather than dumping it whole).

## How to vectorize

The scene API is already batched: state queries return `(num_envs, ...)`
tensors and the action interface accepts per-env rows. The standard lift:

1. **State and targets**: preserve the leading environment dimension. Fixed
   constants may broadcast; anything derived from observation must be derived
   from each environment's own state.
2. **Progress state**: keep reached/grasped/done flags per environment. An
   environment that finishes holds safely while unfinished rows continue.
3. **Control flow**: replace scalar conditions with masks. Batch termination
   means every environment is done or has reached an explicit failure/timeout
   state.
4. **Per-env task structure**: scenes may randomize more than poses across
   envs — object counts, present subsets, and goal structure can differ per
   environment, so different envs may need different amounts (or kinds) of
   work. A single schedule synchronized across the batch cannot fit
   structurally different draws: keep phase/progress state PER ENV and let
   each environment advance on its own conditions, repeating sub-plans until
   that env's own work is done.
5. **Preserve nominal behavior**: same waypoints, same thresholds —
   a vectorization that changes the strategy is a bug. When in doubt, change
   the plumbing, not the plan.

## Verify

    generate --headless {gen} --scene scene_0 --strategy strategy_0 \
        --num_envs {num_envs} --seed {probe_seed}

The batch meta under `{gen}/data/` records per-episode verdicts. The
orchestrator's gate is mechanical: the wide batch must complete env-batched
with AT LEAST ONE success that replays (`replay_check --episodes <ep dirs>`
replays the whole batch and writes `replay` into each episode meta). Yield beyond that is batch economics — every extra
percent of yield is batches saved, so push it as high as the strategy
honestly allows, but do not distort the strategy to chase it. Nominal one-env
behavior must not regress (`--nominal --num_envs 1 --seed 0`). Iterate until
both checks hold.

You are running autonomously: no one answers questions; your final message ends
the session. Keep going until both checks hold or your session budget is spent.
Leave (or extend) `VECTORIZE_NOTES.md` next to the solve: what you changed, the
wide-batch yields you measured, and — if the probe still fails — where the
batch breaks and what you would try next.
