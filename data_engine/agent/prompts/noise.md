# Noise session — author per-phase disturbances into the solve

You are inside a **data_gen campaign** that multiplies one verified robobench
solve into a large demonstration dataset. The harvest that follows re-runs the
solve across {num_envs} parallel envs with fresh randomizations AND executed-action
noise, keeping what the grader passes: the solve gets kicked, its closed-loop
recovery is recorded, and that recovery is the most valuable training data in
the set (DART-style disturbance injection). Your session's one job: decide —
by WATCHING the task — where in the task noise belongs and how much, and write
that policy INTO the solve.

## Watch the task first

A verified episode is rendered at:

    {video}

Extract frames from it (`ffmpeg -i <video> -vf fps=1 frame_%03d.png` and read the
images; the episode dir with its full state stream is `{sample_ep}`). Identify
the phases you see: where the motion is free
(transport, approach from afar — noise-tolerant), and where it is
tolerance-critical (grasping, mating, insertion — noise-fragile). Cross-check
against the solve's own phase structure in the code.

## The noise channel — the ONLY legal way to perturb

- solve to edit IN PLACE: `{solve}`
- the start scene: `{scene}` — it must declare `PHYSICAL_PARAMS` (bands of the
  world physics the task has: friction, masses, small pose offsets, nominal =
  today's values, ranges the solve still succeeds under). Without it every env
  of a harvest batch is the same world. If it is missing, add it (this is the
  one permitted edit to the start scene).
- campaign root (your cwd, writable): `{gen}`
- the dynamics base (the phase set) was drawn from these cells; the harvest runs
  each of them WITH your noise, so put the channel into every one of these
  solves (phase cells execute their own `solve_by_phase.py`). Certification is
  measured on the base cell only:
{cell_solves}
- dynamics clock left: {hours_left} h

Wherever the solve steps the env, it may pass a perturbation:

    env.step(action, noise=my_noise)   # my_noise: same shape as action

The recorder logs `action` (clean — that is the training label, structurally
untouchable) and executes `action + NOISE_SCALE * my_noise`, where NOISE_SCALE
is the pipeline's master switch: probes and the ladder sets run at 0 (your noise code
is inert there), the harvest runs at {noise_scale}. So write your noise
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

## The gate: certification (measured, not claimed)

The orchestrator accepts your work through a PAIRED probe — one clean batch and
one noisy batch under identical seed and draws:

    generate --headless {gen} --scene scene_0 --strategy strategy_0 \
        --num_envs {num_envs} --seed {probe_seed}                      # clean
    generate --headless {gen} --scene scene_0 --strategy strategy_0 \
        --num_envs {num_envs} --seed {probe_seed} --noise_scale {noise_scale}

Certification requires ALL of these, mechanically:

1. the base solve passes `noise=` to `env.step` and `scene_0/scene/scene.py`
   declares `PHYSICAL_PARAMS`;
2. the CLEAN batch still has at least one success — your edit left nominal
   behavior working;
3. the NOISY batch's measured coverage (batch meta `noise.perturbed_row_frac`)
   is greater than zero — the noise demonstrably executes through the channel;
4. at least one PERTURBED noisy episode succeeds AND REPLAYS (its recorded clean
   actions reproduce the success when fed back) — recoverable disturbed data
   that is usable actually exists. Not a yield percentage: the minimum proof the
   harvest will not burn its clock producing nothing.

The orchestrator runs this pair itself after your session; you do not need to
re-run an identical pair, but do test your noise at a small width first. The
verdict is recorded PER CODE FINGERPRINT and never re-measured for the same
code: if the previous certification below rejected the current solve, only a
change to the solve (or scene) can pass — and a solve that works at a small
width can still score 0 at {num_envs} envs, so test at full width before you
finish.

The previous certification measured: {certification}

The harvest keeps ONLY rollouts that pass the grader under your noise, within a
fixed time budget — your yield under noise IS your data output, so find the
strongest noise the task genuinely survives, and check the noisy probe's yield
yourself before finishing. The previous noisy probe's log ({fail_log_size}) is
at `{fail_log_path}` if you need it.

You are running autonomously: no one answers questions; your final message ends
the session. Leave `NOISE_NOTES.md` next to the solve: the phases you saw in
the video, where you put noise and why, and the yields/coverage you measured.
