# pressure_plate (push_button_i6)

**Env name:** `sim_gen.pressure_plate` (scene `pressure_plate`, robot `null`)
**Tier:** easy — 2 stages (pick the load block; place it on the plate so it stays down).
**Execution order:** natural (must lift before placing); no other ordering constraint.

## Seed provenance

Seed: `rlbench/push_button`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/push_button.py`) — a Franka reaches a
small articulated button and gives it a momentary fingertip press. One skill: reach +
poke. No object interaction, no lasting physical requirement.

## What changed, and why it is strategically different

The "button" is inverted into a **weight-activated pressure plate**: a bright-red plate
riding 35 mm of vertical prismatic travel on a pedestal, with a spring return
(`f = k·(home − z) − c·v`, k = 80 N/m — the jointed-button mechanism proven in
robobench's microwave scene). A pedestal lamp glows green while it is pressed.

The seed's entire plan — reach and press — is **useless here by construction**:

1. A momentary press springs straight back (nothing latches on touch).
2. Even an **infinitely patient sustained press fails**: success requires the plate held
   at ≥ 80 % travel **by a settled resting block** (geometric on-plate clause), sustained
   90 consecutive substeps. A bare end-effector press has no block to show.

The solver instead needs a different plan: **select the one sufficiently heavy object**
(dark-red 0.50 kg load cube; the two pale 60 g foam cubes are decoys that sag the plate
only ~7 mm, both stacked ~15 mm — still under the 28 mm threshold), **pick it up, place
it on the 10 cm plate, and walk away** with the weight keeping the plate bottomed. That
is pick-and-place onto a compliant, moving target with a force threshold — not a poke.
Physics is honest: the oracle teleports, but success/score judge only settled poses,
live plate depth, and real spring compression under real block weight.

## Randomization (per episode)

- Pedestal xy jitter ±3 cm + free yaw (plate teleported rigidly with it — the authored
  joint sees an unchanged relative pose).
- Block-to-slot permutation over the scatter arc + per-block xy jitter ±4 cm + free yaw.

## Rubric (score in [0, 1], partial progress latched)

- 0.2 — latched once any block is lifted clear of the floor.
- 0.5 — latched once a block reaches the plate top.
- 0.5–0.9 — live, a block on the plate, scaling with depth toward the press threshold.
- 1.0 — iff `success()`: plate ≥ 80 % travel, loaded by a settled resting block, plate
  settled, sustained 90 consecutive substeps (the anti-poke gate).

## Smoke checks (14 — ALL PASS on the forge)

1. settle: no NaN, plate at home, score 0
2. randomization is real (readback differs across seeded resets)
3. null policy: score ~0, no success
4. – 6. oracle seeds 0/1/2: success + score 1.0 (teleport pick + genuine drop)
7. rubric monotone staged scores (0 < lift 0.2 < foam partial < success 1.0)
8. **negative (seed strategy)**: sustained direct press (150 substeps > press_hold,
   kinematically pinned) never succeeds, scores 0 — expressible, and it fails
9. plate springs back to home after the press is released
10. negative (near-miss): one foam block — partial sag, partial score, never success
11. negative: both foams stacked — still under threshold, never success
12. negative: load block dropped beside the plate (pedestal rim) — no press, score
    pinned at the lift latch
13. – 14. calibration sweep: drop-offset → success rate, 3 seeds each
    (0/20/40/60/80 mm); asserts ≤ 20 mm always presses (6/6) and ≥ 60 mm (past the
    plate edge) never does (0/6). Measured capture radius on the forge: **20 mm**
    (tighter than the 50 mm geometric half-width — drops at 40 mm bounce/tip off the
    sprung plate), i.e. the plate rewards a deliberate centred placement.
