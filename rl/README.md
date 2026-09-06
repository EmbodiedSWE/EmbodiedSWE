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
| Action | the task's frozen controller preset, policy in [-1,1]; `action.affine` re-scales gripper dims or maps arm dims to joint deltas | `configs/tasks/*.yaml`, `vec_env.py` |
| Reward (`rewards/progress`) | the grader's weighted rubric progress, paid every step, + success bonus. **Privileged**: agents never see the grader | `robobench_rl/reward.py` |
| Reward (`rewards/shaped`) | a hand-designed dense potential per task, paid every step, written from public scene accessors (reach, lift, transport, orient, approach, insert/thread) | `robobench_rl/task_rewards/<scene>.py` |
| Episode | the task's own horizon in seconds, always run to the limit (Isaac Lab style); success is logged, not terminal | `configs/tasks/*.yaml` |
| Algorithm | rsl_rl PPO; defaults in the base, per-task overrides in the task file (deep-merged) | `configs/base.yaml`, `configs/tasks/*.yaml` |
| Reporting | grader verdicts of exported checkpoints only; training reward is never a result | `robobench_rl/export.py` |

Config layering: `base.yaml` <- `tasks/<task>.yaml` <- `rewards/<reward>.yaml` <- `key=value` overrides.
Deep-merged, so a task file overrides any base key by writing the same path, e.g. a `ppo:` block
with just `algorithm.learning_rate`. The merged config is dumped next to every run.

## Steps

1. **Smoke / throughput** (this step): wrapper correctness and env-steps per second.
   ```
   python rl/scripts/smoke.py --task bulb_franka_osc --num_envs 16 --steps 200 --headless
   ```
2. **Train**: rsl_rl PPO for `ppo.max_iterations`, checkpoints every `ppo.save_interval`. Optional
   warm-start curriculum (`curriculum.hover_start_frac`): a fraction of envs starts each episode with the
   hand servoed above the part (the shaped reward's `hover_target`); grading still starts from home.
   ```
   python rl/scripts/train.py --task slice_franka_joint --reward shaped --headless \
       --set task.num_envs=256 task.episode_seconds=20 ppo.max_iterations=300     # debug-sized
   python rl/scripts/tb_summary.py rl/runs/slice_franka_joint/shaped/<stamp> --every 50
   ```
   Run dir `rl/runs/<task>/<reward>/<stamp>/`: `config.yaml` (merged config actually used),
   `env.json` (obs layout, action bounds, control rate), tensorboard events, `model_<it>.pt`,
   `summary.json`. `--set` overrides are for debugging; a real run changes the task yaml.
3. **Export + grade**: a checkpoint becomes a self-contained `solution/` folder — generic
   `solve.py`, TorchScript actor (`policy.pt`, normalizer included, no rsl_rl needed), a copy of
   the observation rule (`rl_obs.py`), `env.json` — and is graded exactly like an agent's:
   ```
   python rl/scripts/export.py rl/runs/slice_franka_joint/shaped/<stamp> --checkpoint model_299.pt
   python eval/scripts/verify_solution.py --preset cutting.slice.franka.joint --solution <run>/solutions/model_299
   python eval/scripts/run_grade.py <exp> --solution <run>/solutions/model_299 --out <dir>   # full rubric, container
   ```

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
