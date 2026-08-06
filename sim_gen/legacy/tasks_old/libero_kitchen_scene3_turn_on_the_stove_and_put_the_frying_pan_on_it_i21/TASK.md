# short_order — cook every patty on the already-hot burner, in time, one at a time

**Registered as:** `SCENES["short_order"]`, env `simgen.short_order` (robot="null",
scene-level; robot bindings are a later stage).

**Tier: medium — 3 stages per item x 2-3 items, partial execution order REQUIRED**
(within each item the order is strict: load the burner -> timed cook -> timely removal
and delivery; across items the order is free but SERIALIZED by the single-slot burner,
so the solver must schedule 2-3 items through one shared station).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it.py`)
- Seed plan: toggle the flat stove's knob past 0.5 (turn it ON), then pick-and-place the
  frying pan ON the burner site. Success is a STATIC terminal relation, checked once:
  `stove_on AND pan_on_stove`. The stove's heat has no consequences; time plays no role;
  the moka pot is a passive distractor; the episode ends with the object resting on the
  powered stove forever.

## What changed, and why it is strategically different

Kept: the kitchen-counter setting, a flat stove with a single burner as the central
fixture, and "something must go onto the burner" as a necessary subgoal.

Changed — the PLAN, not the numbers:

1. **The seed's goal state is this task's canonical failure.** The burner is already ON
   (glowing disc + lit lamp; there is nothing to toggle — stage 1 of the seed is deleted
   outright). Heat now has consequences: an object resting on the burner accrues heat
   every physics substep, and past the burn threshold it is IRREVERSIBLY burnt, capping
   its credit at 0.05 forever. Executing the seed's plan — put the thing on the hot
   stove and stop — is negative control A and permanently destroys the episode.
2. **"On the stove" flips from terminal relation to transient stage.** The on-burner
   relation the seed judges once must here be entered AND exited inside a bounded
   residence window (cooked at 1.5 s, burnt at 4.0 s of accumulated settled residence).
   The deliverable relation is elsewhere (cooked patties settled on the serving plate).
   No memorized final snapshot of the seed satisfies any part of it.
3. **Residence-time scheduling under a capacity constraint** — a skill family absent
   from the seed. The burner disc holds exactly one patty (geometric: two 60 mm discs
   whose centres are both inside the 28 mm cook radius would have to interpenetrate; a
   stacked patty sits above the accrual height band — both proven in the smoke). With
   2-3 patties the solver must pipeline items through the shared station, watching each
   item's clock, instead of executing one pick-and-place.
4. **State is perceivable, not hidden:** patties repaint on latch flips (raw pink ->
   cooked brown -> burnt black), and the count (2 vs 3) plus the stove/plate layout
   (including which side of the counter each is on) resample every episode.

Versus sibling tasks: no low-clearance slide (i4/i9), no pouring/emptying (i2/i5), no
stacking-to-height (i11), no size-keyed hanging (i12), no spring plate (i6), no dial
goals (i7); the claimed axes here are *bounded-dwell timing*, *irreversible hazard*, and
*throughput scheduling through a single-slot station*.

## Physics honesty

The oracle places patties kinematically, but every judged quantity is physical: heat
accrues ONLY while the patty's centre is inside the cook radius, at resting height on
the disc, and near-still — so residence is real stepped physics (a teleport across the
region accrues at most one substep; parking beside the burner or on the stove body,
12 mm below the disc, accrues nothing). `success()` is a current-state check: every
present patty cooked, not burnt, resting settled on the plate.

## Rubric

Per present patty: 0.15 seared (latched, >= 0.25 s on the burner), 0.70 cooked
(latched), 1.0 while also delivered (cooked, settled on the plate); burnt caps that
patty at 0.05 forever. Score = mean over present patties, capped at 0.95 unless
success; exactly 1.0 iff `success()`; ~0 for doing nothing (all credit starts at the
burner). Score is deliberately non-monotone in wall time (burning loses credit); the
monotonicity contract holds along the correct plan's milestones and is tested there.

## Smoke battery (16 checks)

1. settle/no-NaN (finite states, at rest on the prep board, score ~0 at reset)
2. randomization-is-real: stove/plate/patty poses move across seeded resets (READBACK)
3. randomization: present patty count varies (2 vs 3) across seeded resets
4. null-policy-fails: 300 idle steps -> score <= 0.02, no success
5-7. oracle reaches success() with score exactly 1.0 on 3 seeds
8. subset episode: oracle succeeds judged on the sampled subset; absent patties parked
9. rubric monotonicity: milestone scores strictly increase along the correct plan to 1.0
   (+ paired check that the final milestone is success at exactly 1.0)
10. single-slot A: only the centred patty accrues heat; the closest stable pair accrues none
11. single-slot B: a patty stacked on the cooking one accrues nothing
12. negative control A (the seed's own strategy): leaving the patty on the hot stove
    burns it — credit collapses to the burnt cap, and success is impossible afterwards
    even with everything served (two checks)
13. negative control B: skipping the stove (raw patties straight onto the plate) ~0
14. near-miss/tolerance: undercooked at 60% of the window -> sear credit only, no success
15. calibration probe: measured accrual rate + cook->burn latch timings consistent with
    the configured window
16. burnt latch irreversible after removal from the stove

`smoke.py` records the run and saves `frames.npz` to the CWD, prints
`SIM_GEN_SMOKE: ALL PASS n/n` on success, and hard-exits (watchdog + `os._exit`).
