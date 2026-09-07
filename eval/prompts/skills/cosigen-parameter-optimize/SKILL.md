---
name: cosigen-parameter-optimize
description: Tunes numeric constants with parameter_search after a stable viable maneuver exists. Use for offsets, depths, angles, gains, timings, or thresholds—not for choosing a strategy.
---

# Parameter optimization

Eligibility gate: first establish and replay a stable viable maneuver. If you are still choosing
control flow, grasp side, phase ordering, or recovery policy, compare strategies with `sweep`
when it is granted, or with an explicit controlled comparison otherwise. Once only numeric
uncertainty remains, `parameter_search.tune` evaluates a whole population of candidate
settings in parallel —
one candidate per env of a WIDE env — scores every end state with your objective, and adapts
the search with CMA-ES over several generations. One call searches hundreds of settings;
re-running by hand searches one.

The signal to use it: a maneuver already works and you are about to re-run it after nudging a
number or sweep a value in a loop. Stop and search instead.

1. Let the tool build the search env: pass the preset name (512 parallel envs by default)
   and anchor it with `start_state=` (a checkpoint node's .pt). Pass your own env instead
   when you need a different width or config — the population is evaluated in passes of
   `num_envs`, and the search replicates env 0's state into every env each generation.
2. Lift each value you want tuned into the search space, and write your rollout to take them
   from `p` — each entry is a `(num_envs,)` array, one value per env — driving all envs at
   once with batched actions.
3. Call `tune(..., goal=, quality=, invalid=)`. `goal` is a public boolean scene method name
   or callable; `invalid` marks dropped/escaped/unrecoverable candidates that can never win;
   bounded lower-is-better `quality` breaks ties only after validity and goal status. Do not
   hand-write penalty weights that let depth trade against lateral alignment or safety.
4. Always pass `seed_values=` from the viable maneuver: they run verbatim as one candidate
   of the first generation, so the reported in-search best cannot score worse than the seed
   under that evaluation. This is not a replay guarantee; noisy physics, integration, or a
   holdout can regress, and only replay establishes otherwise.
5. Noisy contact outcomes: `repeats=k` averages each candidate over k envs. Constants that
   must survive start variation: `randomize` (per-object pose sigmas, meters/radians, sized
   from the task's real start variation) — and with jitter use `repeats` > 1, or each
   candidate is scored on its own single random start and you are ranking noise.
6. Read the returned `verdict` first. Integrate only `ADOPT`; `REJECT`, `INCONCLUSIVE`, and
   `WIDEN` name the next action. Check `improvement_vs_seed`, paired replay, invalid fractions,
   `sensitivity`, and bounds. Population spread inside the noise with near-zero sensitivities
   means the search had nothing to rank; the
   winning values are then arbitrary within the box. Diagnose which root cause applies —
   the objective no longer discriminates where the candidates land (score a continuous
   quality, not a threshold they all clear), the searched constants do not drive this
   outcome (search different ones), or the outcome is noise-dominated (raise repeats).
   More generations fix none of these.
7. Never block on a search: put it in its own script file and `launch()` it — the tool runs
   it detached on the freest GPU — then `status()` reports generations done / best so far /
   the final result while you keep working. A generation can take time at full width; the
   log heartbeats while one is evaluating, so a wait between generation lines is normal —
   let it run instead of killing it.

After an `ADOPT` verdict, write the winning values into the real program and replay the
integrated path. Record failed candidates and regressions. Checkpoint the result only if the
state is settled, portable, useful, and costly to re-derive. The final complete solution must
still pass the queued harness verifier from a fresh reset.
