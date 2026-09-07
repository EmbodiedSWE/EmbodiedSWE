# rl — stock RL baseline for robobench tasks

How does straight RL perform on the same tasks the coding agents are evaluated on?
No agent, no prompts, no hand-authored observations or rewards: build a registered preset
with many envs, observe the flattened `env.get_states()`, reward from the suite's private
grader, train PPO, and grade checkpoints with the **same protocol** the agents are graded by.

Independent of `eval/` (which runs agents in containers). It reuses `robobench` presets and
graders, nothing else.

## Layout and the three conditions

```
robobench_rl/
  obs.py            generic observation rule (flattened get_states, env-local, + EE pose + last action)
  reward.py         per-step reward from a potential: grader progress (privileged) or a dense one
  vec_env.py        RoboBenchEnv — the base RL env (rsl_rl VecEnv) with Isaac-Lab-style task hooks:
                    defaults, _setup, _process_actions (ActionMap), _get_extra_obs, _get_terminated, _hover_target
  tasks/            one module per task: <Scene>DenseReward (the DENSE condition: full-task potential proposed
                    once, up front, never iterated), <Scene>TunedReward + <Scene>TunedEnv (the TUNED condition:
                    expert-iterated first-stage reward + env changes); common.py = shared geometry terms
  export.py         checkpoint -> solution/ (TorchScript actor + torch-only package copy + env.json); the solve
                    ATTACHES the same task env to the graded env
scripts/            smoke.py · train.py · export.py · play.py (grade / --warm diagnostic / video target) · tb_summary.py
configs/            base.yaml <- tasks/<task>.yaml <- rewards/<reward>.yaml
```

| condition | command | what it tests |
|---|---|---|
| **progress** | `--task bulb_franka_osc --reward progress` | stock RL on the grader's own rubric (privileged reward), generic env |
| **dense** | `--task bulb_franka_osc --reward dense` | stock RL on a full-task dense reward proposed once, up front (`tasks/bulb.py: BulbDenseReward`), generic env, no iteration |
| **tuned** | `--task bulb_franka_osc_tuned --reward dense` | expert-iterated RL toward the first rubric stage: `tasks/bulb.py: BulbTunedReward` (measured grasp geometry, per-finger grasp, settled lift baseline) inside `BulbTunedEnv` (horizon, finger PD, warm-start curriculum, termination) |

The generic env is `RoboBenchEnv`: config-driven action scaling, the observation rule, fixed-length episodes,
no termination. A task env subclass overrides only what its condition needs.

## Design rules (one rule each, uniform across tasks)

| Piece | Rule | Where |
|---|---|---|
| Observation | flatten `get_states()` env-local, drop actuator setpoints/controller state, add EE pose + last action | `robobench_rl/obs.py` |
| Action | the task's frozen controller preset, policy in [-1,1]; ActionMap re-scales gripper dims or maps arm dims to joint deltas | `robobench_rl/vec_env.py` |
| Reward (`rewards/progress`) | the grader's weighted rubric progress, paid every step, + success bonus. **Privileged**: agents never see the grader | `robobench_rl/reward.py` |
| Reward (`rewards/dense`) | a hand-designed dense potential per task, paid every step, written from public scene accessors (reach, lift, transport, orient, approach, insert/thread) | `robobench_rl/tasks/<scene>.py` |
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
2. **Train**: rsl_rl PPO for `ppo.max_iterations`, checkpoints every `ppo.save_interval` iterations and/or every `ppo.snapshot_minutes` of wall clock; `ppo.max_wall_minutes` stops the process (after a final save) regardless of iterations. Optional
   warm-start curriculum (`curriculum.hover_start_frac`): a fraction of envs starts each episode with the
   hand servoed above the part (the dense reward's `hover_target`); grading still starts from home.
   ```
   python rl/scripts/train.py --task slice_franka_joint --reward dense --headless \
       --set task.num_envs=256 task.episode_seconds=20 ppo.max_iterations=300     # debug-sized
   python rl/scripts/tb_summary.py rl/runs/slice_franka_joint/dense/<stamp> --every 50
   ```
   Run dir `rl/runs/<task>/<reward>/<stamp>/`: `config.yaml` (merged config actually used),
   `env.json` (obs layout, action bounds, control rate), tensorboard events, `model_<it>.pt`,
   `summary.json`. `--set` overrides are for debugging; a real run changes the task yaml.
3. **Export + grade**: a checkpoint becomes a self-contained `solution/` folder — generic
   `solve.py`, TorchScript actor (`policy.pt`, normalizer included, no rsl_rl needed), a copy of
   the observation rule (`rl_obs.py`), `env.json` — and is graded exactly like an agent's:
   ```
   python rl/scripts/export.py rl/runs/slice_franka_joint/dense/<stamp> --checkpoint model_299.pt
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

| task | preset | grader (main) | first rubric stage = the pick target | tuned env |
|---|---|---|---|---|
| bulb | assembly.bulb.franka.osc | yes | `lifted` (bulb 4 cm up, quasi-static) | `bulb_tuned` — 8/8 from home |
| nut_thread | assembly.nut_thread.franka.osc | yes | `lifted` (nut origin at the bolt-top height) | `nut_tuned` — 8/8 `lifted` from home (0.333), initialised from the bulb policy; resuming it with the carry weights `--set "reward.weights={reach: 0.1, grasp: 0.1, lift: 0.2, transport: 0.3, upright: 0.1, approach: 0.2}"` reached 8/8 `lifted` + `aligned` (0.667) after 50 more iterations |
| slice | cutting.slice.franka.joint | yes | `knife_taken` (knife above the rail height) | `slice_tuned` — 8/8 from home on two seeds (0.111): joint_delta 0.15 (the arm PD caps joint speed at scale·kp/kd), LIFT_FULL 0.08 (the knife sinks 2.1 cm into the notches after reset while the grader keeps the unsettled height), lockstep partial reset, bridging pre-roll; 300 it + 300 it resume, best checkpoint by grading every 50 |
| pen_holder | packing.pen_holder.franka.osc | yes | `pens_in` (a pen inserted tip-up) — pick + carry + insert | `pen_tuned` — FAILED (0/8 on every checkpoint, ~14 runs): pick learned only from warm / in-hand starts (needed PEN_WIDTH 0.0075: the pencil collider is far thinner than its visual, finger kp 2000, filtered finger targets); from home the hand stops beside the pen and never closes; insertions only from an in-hand-over-holder training start |
| tool_packing | packing.tool_packing.franka.osc | yes | `stowed` (tool in its drawer) — the toolbox starts SHUT, so a drawer must be opened first | `tool_packing_tuned` — FAILED (0/8, versions v4–v13 up to 2800 it): the assigned drawer is OPENED from home in 8/8 (7–12 cm) and the stapler is pinched exactly, but never lifted (PPO action std grew to 7–9: rsl_rl has no std cap; entropy 0 recovers slowly); the knife slot is out of the Franka's pinch reach |

Each tuned env is one subclass of `RoboBenchEnv` (`<Scene>TunedEnv` in `robobench_rl/tasks/<scene>.py`). `pc_ram` and the other
grasp-weld scenes are excluded: the weld joint pool is finite per process (8 welds per part), so a
training env silently loses the ability to grasp after a few episodes.

## Reference tasks the tuned envs borrow from

| reference | what we reuse |
|---|---|
| Isaac Lab `Isaac-Lift-Cube-Franka`: reach `1-tanh(d/0.1)`, lift bonus (weight 15) above 4 cm, object-dropping termination, 5 s episodes | the pick rungs (reach → grasp → lift with a threshold bonus), the off-table termination |
| Isaac Lab `Factory-NutThread`: keypoint pose error with coarse/fine kernels, engaged + success bonuses, part starts in the gripper | keypoint reach to a grasp POSE; the warm-start curriculum stands in for the in-hand start |
| ManiSkill `PickCube-v1`: reach `1-tanh(5d)`, +1 while grasped (contact-based), place gated on grasp, success 5 | grasp-gated later rungs; our per-finger closure window replaces the contact check (no contact sensors here) |
| ManiSkill `PegInsertionSide-v1`: pre-insertion alignment (×3) then insertion (×5), success 10 | pen_holder's carry / tip-up / insert rungs |
| Isaac Lab `Franka-Cabinet` (direct + manager-based `Open-Drawer`): handle distance `(1/(1+d²))²`, gripper-axis alignment, a grasp ladder — `align_grasp_around_handle` (one finger each side), `approach_gripper_handle` (per-finger distance, ×5), `grasp_handle` (finger closure paid only near the handle, ×0.5) — then drawer joint × 7.5 with milestone bonuses | the per-finger grasp ladder mirrors our per-finger window; the drawer stage a future `tool_tuned` env needs before its pick (the toolbox starts shut; the rubric pays nothing for an open drawer alone) |
| Isaac Lab `Dexsuite` (Kuka-Allegro lift / reorient): every tracking term gated on a contact-sensor grasp (thumb + one finger over a force threshold) | the gating idea; no contact sensors in our scenes, so proximity + closure stands in |
| Isaac Lab pick-and-place / stacking envs (GR-1, G1, Galbot, Agibot, Franka stack) | ship NO RL configs — imitation-learning setups; the RL zoo stops at cube lift + drawer + Factory insertions |

## Grader partial reset

Graders are single-trajectory objects. Training auto-resets individual envs, so
`GraderReward.reset(env_ids)` re-runs the grader's `setup()` and restores the rows of every
un-reset env from a snapshot of the grader's per-env tensors, then zeroes the reset envs'
"once" milestones. This is generic over any `BaseGrader` whose per-env state lives in tensor
attributes with a leading `num_envs` dimension (all shipped graders).
