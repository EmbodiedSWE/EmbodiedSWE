# Rule: batch solve

You can focus on `num_envs=1` while developing. The delivered `solve`,
however, must be compatible with running in batch: grading may build the
environment with ANY `num_envs` — several copies of the scene stepped
together, each env's object spawns independently randomized. Each env's
trajectory is scored and judged separately — every success counts, and
results are aggregated as successes/total.

So `solve` must run unchanged at any count: read it at runtime
(`env.num_envs`), never assume it is 1.

## Practical notes

The rule ends above; these notes are guidance, not requirements — what
tends to break when a solver written at `num_envs=1` runs in batch, and how
working batched solvers handle it.

- **One state dict per env.** All per-episode run state (phase, timers,
  measured calibrations, integrators) in
  `sts = [fresh_state() for _ in range(env.num_envs)]` — nothing
  module-level, nothing shared between envs.
- **Index every read by env.** `root_pos_w[e]`, `joint_pos[e, ...]` —
  hard-coded `[0]` is the classic way batch support silently rots. Envs
  diverge (different spawns, contact chaos): env 0's measurements do not
  apply to the others, so calibrate per env.
- **One step per control tick.** Assemble a single `(num_envs, action_dim)`
  action tensor from all envs and call `env.step` once.
- **Finished envs still receive an action every tick.** `env.step` is
  global, so an env that finishes early keeps getting your action rows
  until every env is done. What to command there is a task choice — e.g.
  retreat to a clear pose and stay, or simply keep the arm where it is —
  the only real constraint is not disturbing the completed goal. And
  "staying put" depends on the control mode: with relative task-space
  actions (pose deltas) it is the ZERO action — re-sending the last
  nonzero delta drifts the arm; with absolute joint-position targets it is
  a fixed target such as that env's current joints — an all-zeros action
  would yank the arm to the zero pose.
- **Avoid per-value GPU reads in the loop.** Sim state lives on the GPU,
  but per-env decision logic is Python — and every individual value pulled
  from a GPU tensor (`.item()`, `float()`, an `if` on one element) forces a
  GPU->CPU sync. At dozens per env per step that dominates wall-clock (2x+
  slower, measured). A sync costs about the same whether it moves one value
  or the whole state, so either snapshot once per control step — a few
  batched `.cpu()` transfers, then all scalar reads are free — or keep the
  logic fully vectorized on the GPU and never transfer at all.
- **Failure is per env.** An env in an unrecoverable state is marked failed
  and held — exiting the loop early throws away the envs that were still
  working.

As an EXAMPLE, these notes compact into a generic skeleton that fits any
task and any control mode — joint-space or task-space, single-arm or
bimanual (a composite robot just widens `action_dim`). The `solve` loop is
task-independent; the plug-in functions are yours to write:

```python
import torch


def setup(env) -> None:
    """One-time work before the loop: controller retunes (gains, scales,
    control period), joint/body index lookups, constants of the scene.
    Everything here is already batched — it applies to all envs."""
    ...


def pull(env) -> dict:
    """ONE batched GPU->CPU snapshot per control step: every tensor your
    logic reads, all (num_envs, ...) shaped."""
    art = env.robot.articulation
    return {"jp": art.data.joint_pos.cpu(),
            "ee_p": art.data.body_pos_w[:, EE_IDX].cpu(),
            # + whatever object states your task needs
            }


def fresh_state() -> dict:
    """ALL mutable per-episode state for ONE env: phase, timers, per-env
    calibrations, the done/failed flags."""
    return {"phase": "start", "t": 0, "done": False, "failed": False}


def act(S: dict, e: int, st: dict) -> torch.Tensor:
    """Your FSM for one env. Reads S[...][e] only, mutates st, returns one
    (action_dim,) row. On an unrecoverable state set
    st["failed"] = st["done"] = True — never raise or break."""
    ...


def idle_row(S: dict, e: int) -> torch.Tensor:
    """What a finished env keeps receiving until all envs are done — your
    choice per task, as long as it doesn't disturb the completed goal.
    Staying put: zeros for relative task-space actions (+ your gripper
    command), a fixed target (e.g. that env's current joints, S["jp"][e])
    for absolute joint control. A retreat-to-clear-pose is better written
    as a final phase in act() before setting st["done"]."""
    ...


def solve(env) -> None:
    setup(env)
    E, dim = env.num_envs, env.robot.action_dim
    sts = [fresh_state() for _ in range(E)]
    A = torch.zeros(E, dim)
    for _ in range(MAX_STEPS):
        S = pull(env)
        for e in range(E):
            A[e] = idle_row(S, e) if sts[e]["done"] else act(S, e, sts[e])
        env.step(A.to(env.device))
        if all(st["done"] for st in sts):
            return
```

The example above is a closed-loop FSM — phases advance on measured state.
Other common shapes batch the same way:

- **Time-driven (more open-loop) solutions** — keyframes or scripted
  waypoints keyed on time rather than state. Same skeleton: keep the clock
  PER ENV (`st["t"]`, advanced inside `act`) and key the schedule on it, so
  a stall or retry in one env never shifts another env's timing (a single
  global clock would). And because spawns differ per env, compute the
  waypoints per env from THAT env's measured poses (in `setup` or on the
  first step) — a schedule authored against env 0's layout misses the
  others.
- **Mixed** — many solvers are time-driven within a phase (descend for 2 s)
  and state-driven between phases (advance on measured touchdown); the
  per-env dict simply holds both the timer and the trigger measurements.
