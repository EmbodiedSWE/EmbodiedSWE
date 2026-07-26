# Robot task

You are working inside a container prepared for robot tasks in Isaac Lab.
Your job: write code that completes the task described in this folder, and
leave your final solution behind. Some experiments run several tasks in
sequence — each stage gets its own instructions like these, and your
workspace carries over between stages.

## Your environment

- `/bench` (read-only): the `robobench` benchmark package — already on
  `PYTHONPATH`. Read its source freely to understand scenes, robots, and
  controllers.
- `/opt/venv`: Python with Isaac Sim + Isaac Lab installed (`python` on PATH).
- `/workspace` (the ONLY writable directory): your working area. You start here.
- An NVIDIA GPU is available.

## This folder (/task)

- `instructions.md` — this file.
- `task.md` — the task: the scene, the robot, and the goal. Read it first.
- `rules/` — if present: restrictions in force for this experiment. Each file
  states something that is disabled or constrained. Rules are binding.
- `hints/` — if present: optional guidance provided for this run (API notes,
  workflows, examples). Reading every hint before starting usually saves time.

Only `task.md` is always there; a missing folder simply means no rules or no
hints apply to this run.

## Deliverable

Leave `/workspace/solution/solve.py` exposing:

```python
def solve(env) -> None:
    ...
```

It will be graded separately: a fresh environment for the same task is built
and reset elsewhere, then handed to your `solve`, which must complete the
task by stepping forward. During grading, `env.reset()`, `env.set_states(...)`,
and any other shortcut that writes sim state directly are blocked — a graded
run is: reset (not yours) → your `solve` steps → verification.

- Keep module level import-safe: do not launch the app or build an env at
  import time (the grader imports your file inside its own running app).
- For your own iteration, a `if __name__ == "__main__":` block that builds an
  env and calls `solve(env)` is a good pattern — during development anything
  goes; only the graded run has the restrictions above.

Iterate as much as you need in `/workspace` — only `solution/solve.py` is
the deliverable.

## Practical notes

- Running simulations headless saves time: `AppLauncher(headless=True)` —
  create it BEFORE importing anything that touches `isaaclab.sim`. The first
  sim launch takes ~1 minute; later launches are faster.
- Long sim runs: launch in the background and poll their logs.
- Kit sometimes hangs on app close — after your script prints its final
  status, `os._exit(0)` is an acceptable way to end it.
- You can check the scene visually: capture camera frames to PNG, or record
  entire videos to debug a whole attempt.
