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
- `skills/` — if present: optional guidance (API notes, workflows, examples).
  Read the relevant skill when the router selects that capability; do not preload
  every skill before establishing the baseline.
- `tool_router.md` — if present: the concise workflow and eligibility gates for
  exactly the tools granted to this run. It is included below in these initial
  instructions.
- `tools.md` — if present: the complete API reference for those granted tools.
  Their modules are installed at `/task/tools/`, which is on your `PYTHONPATH`.
  Open the relevant section when the router sends you to a tool; the full
  reference remains available without occupying the initial prompt.

A missing optional entry means those rules, skills, or tools do not apply to
this run.

## Deliverable

The deliverable is the whole `/workspace/solution/` folder. Its entry point
is `solution/solve.py`, exposing:

```python
def solve(env) -> None:
    ...
```

Everything else in the folder ships with it — helper modules, calibration
data, whatever `solve` needs. At grading time the folder is mounted
read-only with `solve.py`'s directory on the import path, so plain sibling
imports (`import helpers`) and data files read relative to `__file__` work;
nothing outside `solution/` comes along.

It will be graded separately: a fresh environment for the same task is built
and reset elsewhere, then handed to your `solve`, which must complete the
task by stepping forward. During grading, `env.reset()`, `env.set_states(...)`,
and any other shortcut that writes sim state directly are blocked — a graded
run is: reset (not yours) → your `solve` steps → verification.

Whenever your solution reaches a state worth grading — a first working
grasp, any measurable improvement — run `submit "one line on what this
version achieves"`: it freezes a numbered snapshot of `solution/` under
`/submissions/` (its own folder, outside your workspace) and returns
immediately; keep working. Each submission is graded and becomes one point
on your score-versus-cost curve, so submit early and often — an unsubmitted
improvement earns nothing if the session ends before the next one. Your
final `solution/` always counts as the last submission.

- Keep module level import-safe: do not launch the app or build an env at
  import time (the grader imports your file inside its own running app).
- For your own iteration, a `if __name__ == "__main__":` block that builds an
  env and calls `solve(env)` is a good pattern — during development anything
  goes; only the graded run has the restrictions above.

Iterate as much as you need in `/workspace` — only `solution/` is the
deliverable.

## Practical notes

- The environment builds with a fixed seed, so every fresh build starts from
  the same initial condition. This bench build has no `reset(seed=...)`
  parameter — plain `env.reset()` is what exists. Test against the build's own
  initial condition first, then make the solution robust where you can:
  grading also tests other initial conditions, and GPU physics is not
  bit-deterministic (small errors compound), so closed-loop corrections beat
  open-loop replay.
- Each development script is a fresh process and starts from a fresh
  environment. If this run grants a state-restoration tool, only an explicit
  restore inside that script changes its origin. Never assume simulator state
  carried over from a previous script.
- Running simulations headless saves time: `AppLauncher(headless=True)` —
  create it BEFORE importing anything that touches `isaaclab.sim`. The first
  sim launch takes ~1 minute; later launches are faster.
- Long sim runs: launch in the background and poll their logs.
- Kit sometimes hangs on app close — after your script prints its final
  status, `os._exit(0)` is an acceptable way to end it.
- You can check the scene visually: capture camera frames to PNG, or record
  entire videos to debug a whole attempt. A filepath is not a visual
  inspection: transport/open the actual image pixels before drawing a
  conclusion from them.
- Development runs, checkpoints, and search results are evidence, not the
  official verdict. The final integrated solution must still pass the
  harness's queued end-to-end verification from a fresh reset.
