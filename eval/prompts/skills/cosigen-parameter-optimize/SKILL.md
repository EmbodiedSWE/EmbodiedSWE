---
name: cosigen-parameter-optimize
description: Tunes the numeric constants of a maneuver with parameter_search's parallel CMA-ES. Use whenever an outcome depends on hand-picked numbers (offsets, depths, angles, timings, thresholds) — and always instead of re-running a program with nudged values.
---

# Parameter optimization

`parameter_search.search` evaluates a whole population of candidate settings in parallel —
one candidate per env of a WIDE env — scores every end state with your objective, and adapts
the search with CMA-ES over several generations. One call searches hundreds of settings;
re-running by hand searches one.

The signal to use it: you are about to re-run a program after nudging a number, or to sweep
a value in a loop. Stop and search instead.

1. Let the tool build the search env: pass the preset name (512 parallel envs by default)
   and anchor it with `start_state=` (a checkpoint node's .pt). Pass your own env instead
   when you need a different width or config — the population is evaluated in passes of
   `num_envs`, and the search replicates env 0's state into every env each generation.
2. Lift each value you want tuned into the search space, and write your rollout to take them
   from `p` — each entry is a `(num_envs,)` array, one value per env — driving all envs at
   once with batched actions.
3. Write the objective over the WHOLE env: per-env scores, shape `(num_envs,)`, lower is
   better. Score the same measurement you already print to judge success — and make it
   reflect the whole goal of the maneuver, because the search finds exactly what you score
   and nothing more: if a lift must also keep the object upright and stay gripped, score all
   three terms. Cap credit at the physically achievable value and penalize ruined outcomes
   (dropped, tilted past recovery), or the search will find the ruined outcome that happens
   to score well. Put `inf`/`nan` in an env's slot when that candidate must never win.
4. Always pass `seed_values=` (the numbers you use now): they run verbatim as one candidate
   of the first generation, so the search can never return worse than the incumbent.
5. Noisy contact outcomes: `repeats=k` averages each candidate over k envs. Constants that
   must survive start variation: `randomize` (per-object pose sigmas, meters/radians, sized
   from the task's real start variation) — and with jitter use `repeats` > 1, or each
   candidate is scored on its own single random start and you are ranking noise.
6. Check the returned `sensitivity` and any warning before trusting the result. A gain
   inside the noise with near-zero sensitivities means the search had nothing to rank; the
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

Afterwards, write the winning values back into your program as constants, re-verify once,
and checkpoint the improved stage.
