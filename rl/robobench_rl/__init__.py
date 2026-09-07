"""robobench_rl — a stock-RL baseline for robobench tasks.

Independent of the eval harness and of any agent: build a registered preset with many
envs, observe the flattened `env.get_states()`, reward from the suite's private grader
(rubric-progress delta, or a hand-dense potential), train PPO (rsl_rl), and export checkpoints
as `solve(env)` folders so the SAME grading protocol scores RL and coding agents alike.

Layout:
  config.py   layered yaml: base <- tasks/<task> <- arms/<arm> <- CLI overrides
  obs.py      the one observation rule (flatten get_states, env-local, + EE pose, + last action)
  reward.py   grader-backed reward with a per-env partial reset shim (graders are single-trajectory)
  vec_env.py  rsl_rl VecEnv over BaseEnv: action mapping, episode clock, auto-reset
"""
