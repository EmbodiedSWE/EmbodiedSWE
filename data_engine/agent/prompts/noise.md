# Noise session — author per-phase disturbances into the solve

You are inside a **data_gen campaign** that multiplies one verified robobench
solve into a large demonstration dataset. The next stage (COMPOUND) re-runs the
solve across {num_envs} parallel envs with fresh randomizations AND executed-action
noise, keeping what the grader passes: the solve gets kicked, its closed-loop
recovery is recorded, and that recovery is the most valuable training data in
the set (DART-style disturbance injection). Your session's one job: decide —
by WATCHING the task — where in the task noise belongs and how much, and write
that policy INTO the solve.

## Watch the task first

A verified episode is rendered at:

    {video}

Use your `view` tool on that file to actually watch it (sample more frames or
re-view segments as needed; the episode dir with its full state stream is
`{sample_ep}`). Identify the phases you see: where the motion is free
(transport, approach from afar — noise-tolerant), and where it is
tolerance-critical (grasping, mating, insertion — noise-fragile). Cross-check
against the solve's own phase structure in the code.

## The noise channel — the ONLY legal way to perturb

- solve to edit IN PLACE: `{solve}`
- campaign root (your cwd, writable): `{gen}`

Wherever the solve steps the env, it may pass a perturbation:

    env.step(action, noise=my_noise)   # my_noise: same shape as action

The recorder logs `action` (clean — that is the training label, structurally
untouchable) and executes `action + NOISE_SCALE * my_noise`, where NOISE_SCALE
is the pipeline's master switch: probes and farming run at 0 (your noise code
is inert there), compound runs at {noise_scale}. So write your noise
unconditionally — never gate it yourself — and let the pipeline decide when it
executes.

Design freedom is yours: per-phase magnitudes (heavy where the task tolerates
it, zero or tiny near tolerance-critical contact), burst windows vs continuous,
state-dependent scaling, per-env randomness. Requirements:

1. Noise must go through the `noise=` argument. Adding it into `action` itself
   would poison the training labels — and it cannot pass this session's gate,
   which measures coverage THROUGH the channel.
2. Seed any randomness from the env's seeded RNGs or a deterministic function
   of the step index, so a batch replays from its recorded seed.
3. Never noise gripper close/open commands at pinch time — corrupted pinches
   fail everywhere and teach nothing.
4. Keep the solve's strategy unchanged: you are adding disturbances, not
   redesigning behavior. Nominal (`NOISE_SCALE=0`) behavior must be untouched.

## The gate (measured, not claimed)

The orchestrator accepts your work only when a fresh probe

    generate --headless {gen} --scene scene_0 --strategy strategy_0 \
        --num_envs {num_envs} --seed {probe_seed} --noise_scale {noise_scale}

shows BOTH:
- yield >= {need_successes}/{num_envs} (the solve keeps succeeding under your noise), and
- measured noise coverage >= {min_coverage} of (step, env) rows (batch meta
  `noise.perturbed_row_frac` — real noise actually executed).

Run that probe yourself and iterate until both hold: too timid fails coverage,
too violent fails yield. The productive frontier is the strongest noise the
task provably survives. The previous probe's log ({fail_log_size}) is at
`{fail_log_path}` if you need it.

You are running autonomously: no one answers questions; your final message ends
the session. Leave `NOISE_NOTES.md` next to the solve: the phases you saw in
the video, where you put noise and why, and the yields/coverage you measured.
