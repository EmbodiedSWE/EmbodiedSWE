---
name: cosigen-parameter-optimize
description: Tunes the numeric constants of a control program with the optimize tool's parallel search. Use whenever a program's outcome depends on hand-picked numbers (offsets, depths, angles, timings, thresholds) — and always instead of re-running a program with nudged values.
---

# CoSiGen parameter optimization

The optimize tool runs many copies of your program in parallel on a second simulator,
each with different values substituted for the constants you name, scores every end
state with your objective file, and adapts the search over several rounds. One call
searches hundreds of settings; re-running by hand searches one.

It returns as soon as the search starts, so you keep working while it runs. Each
round's best setting is appended to your following tool results and kept in
/workspace/optimize_status.json, tagged with the program version it tuned.

The signal to use it: you are about to re-run a program after nudging a number, or to
sweep a value in a loop. Stop and optimize instead.

1. Lift each value you want tuned to a top-level constant of the program file
   (`PRESS_DZ = 0.006` at module level, used inside your functions). The program must
   act through the toolkit functions — raw env/articulation handles are unavailable
   inside a search.
2. Write a small objective file: `def objective(v) -> float`, lower is better, scored
   on each copy's end state via its oracle view (v.object_pose, v.eef_pose, v.seated,
   v.hand_frac). Score the same measurement you already print to judge success — and
   make it reflect the whole goal of the maneuver, because the search finds exactly
   what you score and nothing more: if a lift must also keep the object upright and
   stay gripped, score all three terms. Cap credit at the physically achievable value
   and penalize ruined outcomes (dropped, tilted past recovery), or the search will
   find those instead of the task.
3. Call optimize(program='/workspace/<stage>.py', space={'PRESS_DZ': (0.001, 0.02)},
   objective='/workspace/<objective>.py'). Ranges can be ('choices', (...)) for
   discrete sets, or carry 'int'/'log' as a third element. Your file's current values
   always run as one candidate, so the result is never worse than what you have.
4. Start your next piece of work rather than waiting; results arrive on their own.
   When one lands, read best/best_score, sensitivity (which constants matter), and the
   notes — a best value at a search-range edge means widen the range and search again,
   and a budget-truncated search can be continued with a higher budget_s.
5. Put the best values into the program file, verify with execute, and checkpoint.
   The best found within a budget is not necessarily the optimum: the next optimize
   call seeds from the file's current values, so refining further is one more call.

For numbers that must hold up under start variation (not fit one exact pose), add
repeats=2 and randomize={'<object>': {'pos': 0.01, 'yaw': 0.2}}.
