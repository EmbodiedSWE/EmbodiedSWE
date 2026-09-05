# rl — stock RL baseline for robobench tasks

How does straight RL perform on the same tasks the coding agents are evaluated on?
No agent, no prompts, no hand-authored observations or rewards: build a registered preset
with many envs, observe the flattened `env.get_states()`, reward from the suite's private
grader, train PPO, and grade checkpoints with the **same protocol** the agents are graded by.

Independent of `eval/` (which runs agents in containers). It reuses `robobench` presets and
graders, nothing else.

## Design rules (one rule each, uniform across tasks)

| Piece | Rule | Where |
|---|---|---|
| Observation | flatten `get_states()` env-local, drop actuator setpoints/controller state, add EE pose + last action | `robobench_rl/obs.py` |
| Action | the task's frozen controller preset, policy in [-1,1]; only gripper dims re-scaled | `configs/tasks/*.yaml` `action.affine` |
| Reward (`rewards/progress`) | delta of the grader's weighted rubric progress + success bonus. **Privileged**: agents never see the grader | `robobench_rl/reward.py` |
| Reward (`rewards/shaped`) | delta of a hand-designed dense potential per task, written from public scene accessors (reach, lift, transport, orient, approach, insert/thread) | `robobench_rl/task_rewards/<scene>.py` |
| Episode | the task's own horizon in seconds; success terminates, timeout flagged | `configs/tasks/*.yaml` |
| Algorithm | rsl_rl PPO; defaults in the base, per-task overrides in the task file (deep-merged) | `configs/base.yaml`, `configs/tasks/*.yaml` |
| Reporting | grader verdicts of exported checkpoints only; training reward is never a result | step 3 |

Config layering: `base.yaml` <- `tasks/<task>.yaml` <- `rewards/<reward>.yaml` <- `key=value` overrides.
Deep-merged, so a task file overrides any base key by writing the same path, e.g. a `ppo:` block
with just `algorithm.learning_rate`. The merged config is dumped next to every run.

## Steps

1. **Smoke / throughput** (this step): wrapper correctness and env-steps per second.
   ```
   python rl/scripts/smoke.py --task bulb_franka_osc --num_envs 16 --steps 200 --headless
   ```
2. **Train**: `rl/scripts/train.py` — rsl_rl PPO for a fixed number of iterations, checkpoints on its save interval.
3. **Export + grade**: each checkpoint becomes a `solution/` folder (generic `solve.py` + weights)
   and is graded by `eval/scripts/run_grade.py --solution ...` or `verify_solution.py`.

## Measured throughput (step 1, RTX 5090, 2026-09-05)

The wall clock per batched control step is nearly independent of the env count (87% of it is
PhysX `fetch_results`; the wrapper is ~1 ms), so env-steps/s scale linearly with envs. The
per-step cost is set by the scene's collision: the SDF-thread scenes (bulb, nut_thread) cost
~0.2-0.4 s per step, the plain-mesh scenes ~0.05 s:

| task | envs | env-steps/s | GPU mem |
|---|---|---|---|
| bulb (16 substeps @1/240) | 1 / 16 / 64 | 5 / 75 / 296 | 4.8 GB |
| bulb | 256 | 1,106 | 5.9 GB |
| bulb | 1024 | 4,221 | 17.6 GB |
| nut_thread (32 substeps @1/480) | 256 | 606 | 5.9 GB |
| pen_holder (8 substeps @1/120) | 256 | 4,622 | 4.4 GB |
| tool_packing (8 substeps @1/120) | 256 | 3,403 | 4.8 GB |
| slice (joint mode, 48 Hz, 1/240) | 256 | 5,844 | 6.0 GB |

At 1024 envs, 1000 PPO iterations x 64 steps/env = 65M env-steps ≈ 4.3 h. PhysX GPU buffers
(collision stack, contact/patch counts) are scaled with the env count in `vec_env.py`; 1024 bulb
envs overflowed the scene's default collision stack. Rerun any row with
`python rl/scripts/smoke.py --task <task> --num_envs N --steps 60 --headless` (`--profile` for cProfile).

## Task subset

| task | preset | grasp | grader | shaped reward |
|---|---|---|---|---|
| bulb | assembly.bulb.franka.osc | friction | existing | `task_rewards/bulb.py` |
| nut_thread | assembly.nut_thread.franka.osc | friction | **missing** (teammates) | `task_rewards/nut_thread.py` |
| pen_holder | packing.pen_holder.franka.osc | friction | **missing** (teammates) | `task_rewards/pen_holder.py` |
| tool_packing | packing.tool_packing.franka.osc | friction | **missing** (teammates) | `task_rewards/tool_packing.py` |
| slice | cutting.slice.franka.joint | friction (knife) | existing | `task_rewards/slice.py` (handle point to calibrate) |

Graders for nut_thread, pen_holder and tool_packing are owned by the benchmark team and not yet
written; until they land, the wrapper refuses those tasks (it needs a grader for success
termination, the `progress` reward, and for grading checkpoints). Bulb and slice run today. Isaac Lab's Factory-NutThread is the last phase of our
nut_thread (nut starts in the gripper above the bolt); an in-hand-start variant of our scene
would reproduce it inside this harness as a positive control.

`pc_ram` was tried (256 envs, 1.1k env-steps/s) and dropped: its grasp relies on the
weld-on-closure contract, whose per-env joint pool is finite for the life of the process (8
welds per part), so a training env silently loses the ability to grasp after a few episodes.
The same holds for every grasp-weld scene.

## Grader partial reset

Graders are single-trajectory objects. Training auto-resets individual envs, so
`GraderReward.reset(env_ids)` re-runs the grader's `setup()` and restores the rows of every
un-reset env from a snapshot of the grader's per-env tensors, then zeroes the reset envs'
"once" milestones. This is generic over any `BaseGrader` whose per-env state lives in tensor
attributes with a leading `num_envs` dimension (all shipped graders).
