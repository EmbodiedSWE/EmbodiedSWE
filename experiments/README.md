# experiments/

Scratch workspace for **agent runs, generated solutions, and manual tests**. Everything in this
folder is **gitignored** except this README — so the folder exists in the repo (ready to use), but no
solution, log, checkpoint, or learned skill is ever committed. This keeps the benchmark pristine: any
agent (including you, by hand) starts from the same clean `robobench/` tasks.

## What goes here

Put each run in its own subdirectory, e.g.:

```
experiments/
  2026-06-27_nut_thread_franka_osc/      # one run
    solution.py            # the policy / script the agent wrote
    logs/                  # stdout, sim logs
    skills/                # any skills the run learned / reused (continual learning)
    checkpoints/           # trained weights (e.g. residual-RL)
    eval.json              # verifier score / success metrics
```

Naming is up to you (`<date>_<task>_<agent>` works well). Nothing here is tracked, so feel free to be
messy.

## What does NOT go here (stays committed, elsewhere)

- **Tasks / environments** → `robobench/` (scenes, robots, controllers, verifiers).
- **Reusable agent machinery** (run-loop, residual-RL, skill library) → `agent/` *(planned)*.
- **Experiment specs** (which env + agent + ablations) → `configs/` *(planned)*.
- **Env-correctness tests** → `tests/` *(planned)*.

The rule: **committed = specs + machinery; gitignored = anything an agent produces.** Holding that
line keeps the path open to a cap-x-style `configs/` + skill-library setup later — purely additive,
no reorg.
