# Trajectory diversification — protocol

Input: a task plus a program that verifiably solves it. Output: a sampler (program + parameter schema)
and the verified trajectories it yields. Numbers are measured on `assembly.nut_thread.franka.osc`,
batch `nut_b2` (100 episodes, 26 successes).

## What gets randomized

1. **World** — spawn poses, physical properties, object set/count. Dimensions come from `cfg.tunables()`
   minus the fields that ARE the verifier (`seat_z`, `align_xy`, `align_axis_deg`). Cheap because a
   closed-loop program re-derives targets from measured state instead of replaying a path.
2. **Numbers in the program** — constants lifted to distributions, bands taken from the program's own
   record of itself (comments, argparse help, repair log).
3. **Structure of the program** — subtask order, target choice within a goal region, strategy, and the
   control structure itself. Only this class changes what is *possible*. Sampling stays inside one
   homotopy class; a different strategy (regrasp vs hook) is a separate agent session on this harness.

A bad band costs yield, not data quality — the verifier filters every episode, and the 74 failures in
`nut_b2` contaminate nothing. Class-3 edits change the policy that produced earlier data:
regression-check against θ that already passed, and version the dataset by program.

## Phase 0 — gate

- Closed-loop? The loop must read state between steps, not replay constants; open-loop replay has no
  tolerance to sample, so converting it is the first edit.
- Success predicate over the final state only. If absent, write one. No intermediate rubric.
- Measure the program's own success rate (6–8 unmodified runs) first. A documented solve can be one
  lucky draw: this scene re-jitters the spawn ±10 mm every reset (`reset_pos_jitter = 0.01`) and the
  reference never disables it, so its recorded 291 s solve does not reproduce.

## Phase 1 — annotate (one LLM pass, O(1) per task)

Emit `params.yaml` (`env` / `policy` / `strategy` / `frozen`) and the program refactored to read `theta`.

- Every entry carries a reason; a constant whose reason you cannot state is `frozen`.
- Bands come from what the program says about itself: `pinch_wind_n` ≥ 15 N because "3 N cams over the
  hex corners once engaged (needs ~0.15 N·m)"; the xy-integrator cap is frozen because "a big clamp at
  480 Hz slams the hand sideways and ejects the nut".
- Read the verifier for free dimensions: goal regions the program collapsed to a point (`pen_holder`:
  `xy_tol` 3.5 cm, `depth_min` 3.5 cm) and symmetries (hex nut → six equivalent jaw landings).
- Couplings: derived quantities are recomputed from the sampled value, never sampled separately.
- Lift the plan where the predicate is a final-state conjunction — `pen_holder.success()` makes all 4!
  orders legal, so four hardcoded blocks become a loop over a sampled order. Precedence comes out of
  the marginals, not out of a declared model.
- One multiplier over phase durations buys velocity diversity at no risk.

## Phase 2 — sample and record

- θ = f(episode index), pure: worker *k* runs episode *k*, reproducible, no coordinator. Low-discrepancy
  over continuous dims, cycled discrete choices, episode 0 = nominal as control.
- Success is Bernoulli given θ, not a property of θ — spawn jitter and non-deterministic physics both
  move it.
- Capture state before the step, or `(s, a)` pairs shift by one control period. Video must come from the
  same rollout the verifier judged; contact-rich GPU physics is not run-to-run identical, so a re-render
  from replay is a different episode.

## Phase 3 — repair (2–3 rounds, mandatory)

From per-parameter marginals plus a few failure telemetries, three moves:

1. Reclassify a dimension as frozen — shrinks the space.
2. Narrow a band or add a coupling — shrinks the space.
3. Rewrite the program so the failing region becomes feasible (gate the rewind lift on measured
   engagement depth, make a fixed waypoint live-retargeted) — the only move that grows it.

The annotation pass will misclassify, which is why this is not optional: `nut_center` was sampled as a
strategy choice although the help string said "REQUIRED at the flat layout" — 6% yield in that arm
against 45%, 44 of 62 episodes spent. And never let a stopping condition double as a success test:
`dz <= stop_dz` also accepts a nut knocked to the floor (21 episodes reported done at stroke 0).