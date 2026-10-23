# RoboBench

Write a robot control program that completes the goal in `/task/task.md`.
The benchmark and simulator are installed; an NVIDIA GPU is available.

- Read `/task/task.md` and every file in `/task/rules/` before solving.
- `/bench/robobench` contains the task, robot, and controller source. It is on
  `PYTHONPATH`. Treat this benchmark code and the initial task configuration as fixed.
- `/opt/venv/bin/python` (`python` on PATH) runs the correct simulator backend.
  Launch `isaaclab.app.AppLauncher` before importing simulator modules. Keep
  simulator experiments in separate processes and wait for them to finish.
- Work in `/workspace`. Optional guidance and tools are under `/task/skills/`
  and `/task/tools/`; the latter is also on `PYTHONPATH`.

## Deliverable and checkpoints

Leave `/workspace/solution/solve.py` exposing `def solve(env) -> None`. The
whole `solution/` directory is the deliverable: include all helper modules,
weights, and data files it needs. Use paths relative to `__file__` for data.
Module-level code must be import-safe: the evaluator starts the simulator,
builds and resets the prescribed environment, then imports and calls `solve`.
Do not depend on files elsewhere in your workspace or packages installed
during development: grading uses a fresh copy of the supplied runtime image.

Run `submit "brief note"` whenever you have a version worth grading. It freezes
the solution into `/submissions/` and returns immediately. Do not edit frozen
submissions. Do not write to solution/ concurrently with submit. Files must be
ordinary files and directories; symlinks and special files are not supported.
The checkpoint limit, seeds, and time limits are recorded in `/task/protocol.json`.
The final solution/ is also evaluated even if you never call submit.

Each distinct submitted solution is evaluated in fresh simulator processes,
once for every evaluation seed. The highest mean rubric score wins; ties are
broken by success rate, then the earliest submission (final comes last).
Success is reported for that same selected solution. Identical directory
contents are evaluated once. Partial credit uses the existing RoboBench rubric.

During grading, solve must advance the environment through robot actions.
Resets, restoring states, directly moving objects, modifying physics or the
task, changing the prescribed controller, accessing hidden graders, and writing
grading artifacts are forbidden. Development access is for experimentation;
the delivered solve must obey these restrictions.

This is an autonomous evaluation. No one will answer questions. Your final
message ends the agent phase; leave your best solution on disk before replying.
Checkpoints already submitted remain eligible if time runs out.
