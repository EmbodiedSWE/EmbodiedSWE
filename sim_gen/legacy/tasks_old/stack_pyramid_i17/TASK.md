# vault_unstack (`stack_pyramid_i17`) — un-stack the tower, store the blocks, free the prize

**Env name:** `simgen.vault_unstack` (scene `vault_unstack`, robot `null` — scene-level task)
**Tier:** medium — **4 stages** (store top block, store mid block, store cap block → well
opens, extract + deliver the prize).
**Execution order:** partially forced by physics — the cap can only come off after the
blocks above it, and the prize can only leave the well after the cap (asserted by a kick
probe); the three tray placements themselves are order-free. Declared: *partial ordering
required*.

## Seed provenance

- Seed: `maniskill/stack_pyramid`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/stack_pyramid.py`)
- Seed plan: three loose cubes (red / green / blue) scattered on a table; pick and place
  to BUILD a pyramid — red next to green, blue stacked on top of both. Success is a
  purely constructive relative-bbox check (blue on top of red AND green). The core skill
  is additive free-space stacking; the world starts flat and ends tall.

## What changed, and why it is strategically different

The construction is **inverted into demolition-for-access**, and stacking — the seed's
entire skill — is rewarded **nowhere** (both directions are explicit negative controls):

1. **The world starts tall and must end flat.** The red/green/blue tower already exists
   at reset, capping the square well of a vault in which a golden prize cube is buried.
   The seed's goal state is (approximately) this task's *initial* state.
2. **The judged objects invert.** The seed judges the stacked cubes' mutual geometry;
   here the tower blocks are *obstacles* whose only requirement is to be OUT of the way
   and tidily stored, and the deliverable is a fourth object the seed doesn't have — the
   occluded prize. The plan is: uncover → store → extract → deliver, not arrange-on-top.
3. **Occlusion creates a physical ordering constraint the seed lacks.** The 80 mm cap
   block spans the 64 mm aperture, so the prize is physically imprisoned while the tower
   stands: a 1.5 m/s vertical kick cannot raise it past the rim under the intact tower,
   and the same kick clears the rim once the blocks are removed (both measured in the
   smoke — the imprisonment is physics, not rubric fiction).
4. **The seed's own strategy is expressible and fails twice over.** Re-building the
   tower perfectly anywhere on open ground scores ≤ 0.15 with no success (negative A);
   and even inside the goal container, a block stacked on another block is rejected by
   the rest-on-the-tray-floor gate (negative C) — blocks must be laid out individually.
5. **Brute-force shortcuts fail.** Toppling the tower "frees" the prize but scatters the
   blocks: blocks dumped on the ground beside the tray count for nothing even with the
   prize delivered (negative B, score ~0.5, no success) — controlled placement into the
   walled tray is load-bearing, like the seed's "without it falling off" clause but for
   the inverse motion.

A solver therefore needs a different **plan** (top-down disassembly under an access
constraint + tidy containment + retrieval-and-delivery), not different parameters.

## Scene summary (fully procedural, no asset files)

- **Vault** (kinematic compound body): 64 mm square well, 45 mm deep, in a slate plinth;
  rim at 57 mm.
- **Tower** (dynamic): red cap 80 mm / 0.15 kg, green mid 65 mm / 0.08 kg, blue top
  50 mm / 0.05 kg, stacked on the rim with per-level alignment jitter + yaw.
- **Prize** (dynamic): gold 36 mm / 0.04 kg cube inside the well.
- **Storage tray** (kinematic compound): inner 270 × 140 mm, 40 mm walls — all three
  blocks fit side by side on the floor; blocks must be lifted over the walls.
- **Delivery pad** (kinematic): 110 mm violet slab; the prize must rest fully on it.
- **Randomization per episode:** vault xy, tray xy + yaw, pad xy, per-block stack jitter
  + yaw, prize position in the well (verified by readback).

## Rubric (graded, latched transients; 1.0 iff success)

- +0.15 per block **stored**: settled inside the tray walls, bottom resting on the tray
  floor (rejects stacking in the tray and perching on walls);
- +0.10 **well uncovered** (latched in `post_step` when no tower block overlaps the
  aperture — a transient achievement that survives later clutter);
- +0.15 prize currently **out of the well**;
- +0.25 prize settled **on the pad** (fully on, correct height);
- **1.0 iff success** = all 3 stored + prize on pad (max partial 0.95; the largest
  reachable non-success score is 0.80). Doing nothing scores exactly 0.

## Smoke battery (16 checks)

1. settle/no-NaN + score exactly 0 at rest, tower covers the well;
2. reset sanity: prize in the well, latch clear;
3. randomization is real (readback: vault, tray pose+yaw, pad);
4. null policy fails (240 idle steps → score ~0);
5–7. teleport-oracle reaches success() and score 1.0 on 3 seeds;
8–9. rubric monotonicity ladder hits the declared milestones
   [0.15, 0.30, 0.55, 0.70, 1.0] strictly increasing; all partials < 1.0;
10. negative A — the seed's own strategy (perfect re-stack elsewhere) ≤ 0.15, no success;
11. negative B — blocks dumped beside the tray + prize delivered: no block counts, ~0.5,
    no success;
12. negative C — a block stacked on another block inside the tray does not count;
13–14. near-miss (prize beside the pad → 0.70, no success) + recovery to success;
15–16. occlusion calibration probe — the capped well physically imprisons the prize
    (measured max kick height under cover vs. uncovered).

Oracle: kinematic pose writes only for staging (teleport-oracle); every judgment is on
settled physical outcomes after real simulation steps.
