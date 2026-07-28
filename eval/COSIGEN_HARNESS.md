# The CoSiGen agent harness (`eval/cosigen_*.py`)

A second way to run an agent against these tasks, alongside the container harness in
this directory. The container harness hands a bare coding agent a machine with Isaac
Sim and asks for a `solve(env)` it can grade later. This one keeps **one booted env
alive on a render pod** and gives the agent tools against it, so a session is a
sequence of programs run against a persistent world rather than one script written
blind.

Both target the same `robobench` presets, so a task is defined once.

## What the agent gets

- `execute(path=...)` — run a program from its workspace against the current state.
  The program's namespace holds the control toolkit (`move_to`, grippers, `step`), the
  read-only queries, and the raw `env` / `api` handles for Isaac Lab-flavoured code.
- Every run that moves the world returns **an image of the state it reached**, next to
  the printed numbers, with no call needed. `view(frames=k)` samples k frames across
  the run when the end state alone does not explain what happened.
- `assess(...)` — required after any run that moved the world, before the next one: what
  the image shows, what the log shows, the failure modes visible in them, and whether to
  keep the state. Keeping it saves the program as a node.
- `checkpoint(label=, path=)` / `goto(node=, confirm=)` — a tree of saved world states.
  `goto` first shows that node's image, log and program and asks for confirmation, so
  backtracking is never blind.
- `optimize(program=, space=, objective=, setup=)` — runs one program as N parallel
  copies with different values for named top-level constants, scores each end state with
  the agent's own objective file, and searches with CMA-ES. Asynchronous: the call
  returns at once and each round's best lands in later tool results.

## Layout

| file | role |
|---|---|
| `cosigen_render_server.py` | the pod: boots one env, serves `/run_policy`, `/ping`, `/reload` |
| `cosigen_session.py` | per-turn execution, prompt assembly, api selection per preset |
| `cosigen_apis.py` | the control api the agent's programs call, per scene family |
| `cosigen_agentic_loop.py` | the driver: tools, the inspection gate, artifacts (runs off-pod) |
| `cosigen_opt.py` | the parameter search (CMA-ES over per-env program copies) |
| `cosigen_prompts.py` | the agent-facing documentation, composed per enabled facility |
| `cosigen_watchdog.py` | an LLM judge that reports a run's hand-picked constants |
| `cosigen_view.py`, `cosigen_config.py`, `cosigen_loop.py` | oracle views, config, aggregator |
| `skills/` | agent skills: staged planning, the checkpoint tree, parameter search |

Facilities are gated per run (`CAPX_DISABLE_FEATURES`), so an arm without the checkpoint
tree or without the search tool is offered neither the tool, nor its skill, nor any
mention of it — the ablation is enforced at the method, not just in the prompt.

## Notes for reviewers

- The driver runs outside this repo (it drives pods and holds the agent's workspace);
  these files are the pod-side harness plus the driver module it imports.
- Success uses the suite's grader when one exists for the scene, falling back to the
  scene's own predicate — so a session here and a graded delivery from the container
  harness agree on what "solved" means.
- `eval/examples/`, which the agent prompt points at, is derived at deploy time from
  `references/Fable_5/examples/` rather than kept as a second copy here. The ikea solver
  is excluded everywhere: agents are evaluated on that task.
