# This level: strategy

A **strategy is one way to do the task**, delivered as its own `solve.py`.
The world stays the same (scene `{base}`); what multiplies here is
**behavior**: a different order, a different grasp, a different division of
labor produce demonstrations that genuinely look different — that is the
variety this level adds to the dataset.

What makes a strategy worth shipping:

1. **It works** — shown by a test batch with successful episodes, judged by
   the same grader as everything else. Propose freely and iterate: a
   strategy that fails its first batches can usually be fixed, and bold
   attempts that need a few rounds are worth more than safe ones that
   don't.
2. **It brings a different way of doing the task.** Aim for episodes that
   would look different side by side — a different order, a different
   grasp, a different role for each arm. Re-tuned numbers on the same
   motion add little; a genuinely different approach is the value of this
   level.

## What counts as a new strategy

- **Small modifications** of the base strategy: do the subtasks in a
  different valid order; grasp a different site on the part; approach from
  the other side; hand a part over instead of reaching across.
- **Big changes**: a genuinely new plan — e.g. where one strategy has a
  single arm assemble while the other only reorients the workpiece, a new
  strategy hands the part over and both arms work.
- **Not a new strategy**: an existing solve with different constants
  (speeds, forces, clearances). Those belong inside a solve as keyword
  arguments (below), not in a new cell.

## Deliverables

1. `strategies/strategy_N/` — one cell PER distinct way of doing the task
   (via `create_cell`; it starts empty):
   - `solve.py` — your strategy, written by you.
   - `SUMMARY.md` — what you thought, wrote, measured, learned.

## Writing solve.py

**Study the base strategy's `solve.py` first** — it is the working
reference for everything: the env API, the scene's handles, the control-loop
rhythm, how task-done is detected. Your solve follows the same contract:

- `def solve(env)` — drive ALL envs of the batch together (everything is
  batched; one call solves `num_envs` worlds in parallel).
- Act only through `env.step(action)`, read the world through the env's
  states and the scene's handles. **Never teleport parts or the robot**:
  everything you write into the sim becomes part of the recorded
  demonstration.
- The runner resets the world before calling you — your solve simply starts
  from whatever state the scene's reset produced, runs to task-done, and
  returns.
- Keep your phases visible in the code (hover → grasp → place → …): a later
  phase-level session divides your solve at exactly those seams.
- **Name your constants**: magic numbers — speeds, forces, clearances,
  hover heights — live as plain constants at the top of `solve.py`, not
  buried inline (see the next section for why).

## Solve parameters: constants the sampler can vary

`solve(env)` never takes extra arguments. Instead, its tuning numbers sit at
the top of the file as UPPERCASE constants, with the values you actually
solved with — those values ARE the nominal:

    HOVER_DZ  = 0.02    # hover height above the part (m)
    WIND_RATE = 1.0     # screwing-speed multiplier
    GRIP_N    = 40.0    # grip force (N)

    SOLVE_PARAMS = {
        "HOVER_DZ":  {"dist": "uniform", "lo": 0.015, "hi": 0.03},
        "WIND_RATE": {"dist": "uniform", "lo": 0.8, "hi": 1.3,
                      "reason": "speed diversity at no risk"},
        "GRIP_N":    None,   # declared, not sampled
    }

`SOLVE_PARAMS` tells `generate` which constants it may vary: it draws ONE
set per batch (`--solve_draw` picks which draw) and writes the values onto
the module before calling `solve(env)`; `--nominal` leaves the file's values
untouched. Conditions that must hold:

- every name in `SOLVE_PARAMS` must be a module-level constant that exists
  in the file — checked loudly before anything runs; a typo kills the batch,
  never silently goes nominal.
- read a constant where you use it, inside the solve. Do not copy it into
  another module-level value at import time (`DESCEND_Z = HOVER_DZ - 0.01`
  at the top of the file would keep the OLD number after sampling — compute
  such things inside the solve instead).
- available distributions:
  - `{"dist": "uniform", "lo": 0.015, "hi": 0.03}` — flat between bounds;
    the default choice for most physical-ish values.
  - `{"dist": "loguniform", "lo": 1e2, "hi": 1e4}` — for values spanning
    decades (gains, stiffness-like numbers); `lo` must be > 0.
  - `{"dist": "gaussian", "mean": 0.02, "std": 0.004, "lo": 0.01}` — bell
    around a mean; optional `lo`/`hi` clip the tails.
  - `{"dist": "choice", "options": ["left", "right"]}` — discrete
    alternatives, cycled evenly across draws.
  - `"reason": "..."` — optional free text on any entry, recorded in the
    batch meta.
  - `None` — the constant is declared (visible, documented) but never
    sampled.
- only band values you know are safe to vary — the base solve's history
  says which were tuned to a knife's edge; when unsure, declare with `None`
  and note it in `SUMMARY.md`. Grading thresholds and controller gains
  never go here. Do not invent any other sampling mechanism.

## Verification

    generate --headless /workspace --scene {base} --strategy strategy_N --num_envs 8 --seed 0

This generates one batch of data using your strategy, under
`/workspace/data/<batch>/`: one `ep_NNNN/` folder per episode, success/fail
in each episode's `meta.json`, and the batch summary (yield) in
`data/<batch>/meta.json`. Physical parameters are sampled automatically
(env 0 always keeps the plain, unsampled world); add `--nominal` to turn
sampling off while you debug. Judge by success against the base strategy's
yield on the same scene (the pool's batch metas have it). A cell whose tests
never produce a successful episode must not ship: fix it or delete it.

Either way, write `SUMMARY.md` at the cell root: what you tried, what you
wrote, the yields you measured, and what you learned — failed ideas
included; they are what the next session learns from.
