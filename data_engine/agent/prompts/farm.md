# Farm session — deliver the base set

You are inside a **data_gen campaign** that multiplies one verified robobench
solve into a large demonstration dataset. The cells below — the base cell plus
the scene/strategy/phase variants earlier sessions authored — are your fields.
Your session's one deliverable: **a base set of {k} verified successful
trajectories, as diverse as you can make it.** How to split the work across
cells is entirely your judgment: you authored them, you know what each one
adds, and you will see each batch's yield as you go.

## Your cells

{cells}

## How to work

- `generate --headless {gen} --scene <s> [--strategy <t>] [--phase <p>]
  --num_envs {num_envs} --seed <n> --env_draw <i> --solve_draw <j>` runs one
  graded batch; every episode's verdict lands in
  `{gen}/data/<batch>/ep_NNNN/meta.json`. Distinct `--env_draw` slices give
  distinct per-env world draws; distinct `--solve_draw` gives a new solve
  hyperparameter set — vary both across batches, they are diversity for free.
- A cell that keeps yielding nothing is yours to abandon; a cell with rare but
  real successes is yours to push on. Spend your time where the diversity is.
- When you are done, write **`{gen}/base_set.json`**: a JSON list of exactly
  {k} episode paths relative to the campaign root, e.g.
  `["data/batch_20260827_101500/ep_0007", ...]`. Every entry must be a graded
  SUCCESS. Pick for diversity — across cells, across draws — not for ease.

## Acceptance (mechanical, checked by the orchestrator)

Exactly {k} entries; no duplicates; every entry exists and its meta.json says
`"success": true`. If your delivery is rejected you will be re-briefed with
the reason; the previous rejection (empty on the first session) was:

    {rejection}

You are running autonomously: no one answers questions; your final message ends
the session. Leave `FARM_NOTES.md` in the campaign root: how you allocated
batches across cells, the yields you observed, and why you picked the {k} you
picked.
