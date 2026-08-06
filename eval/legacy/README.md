# legacy/ — retired, kept for reference

Nothing here is on a live path. Each entry records why it was retired, so a future reader does
not have to re-derive it (or worse, revive it by accident).

## The docker runners — `run_agent.py`, `run_interactive.py`, `run_sequence.py`

Retired 2026-07-30, superseded by `scripts/run_agent_sandbox.py`.

All three drive `docker run`. There is no docker on the devbox this harness runs from, and every
run ever recorded used a non-docker substrate — 20 × `run_agent_sandbox`, 60 × the pod-port
harness, 7 × `run_agent_pod` — so none of the three has produced a run here. `run_agent_sandbox.py`
is the same contract on the env-manager sandbox substrate (its `SUBSTRATE:` comments map each
docker step to its replacement).

What they still document, if you need it:

- `run_agent.py` — the original single-run reference: the `-v` mount set, the container env, and
  the poll loop that `run_agent_sandbox.py` reproduces.
- `run_sequence.py` — multi-stage transfer runs (workspace carried stage to stage). The sandbox
  runner is deliberately single-stage; if sequences come back, port this rather than reinvent it.
- `run_interactive.py` — human-in-the-loop sessions, paired with the `interactive_session` rule.

They were retired BEFORE the condition refactor, so they still call the old
`prompts.render_task_dir(hints=..., rules=...)` surface and will not run as-is: `hints` is now
`skills`, and a condition is loaded by `envbuild.condition`.

## `prompts_configs/` — the old condition files

The pre-refactor `eval/prompts/configs/*.yaml`, kept so an older run's `config:` path still
resolves to something readable. Conditions now live in `eval/configs/` and declare
`rules / skills / tools / features`; these declare only `hints` + `rules`.

Name changes when reading an old file: `hints:` → `skills:`, `save_snapshot` →
`save_checkpoint`, `nosnap.yaml` → `no_checkpoint.yaml`, and the build flags
`--no-set-states` / `--no-freeze-controller` → the condition's `features:`.

## `cosigen_null.py`, `cosigen_rl.py`, `skills-cosigen-rl-subpolicy/`

Earlier retirements: the null-embodiment prompt surface (no session ever ran on it) and the RL
sub-policy engine with its skills.
