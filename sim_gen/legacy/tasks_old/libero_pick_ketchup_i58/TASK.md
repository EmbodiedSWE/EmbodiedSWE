# height_lineup — stand the bottles on the shelf, sorted by height

## Seed provenance

- Seed id: `libero/libero_pick_ketchup`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero/libero_pick_ketchup.py`
  ("Pick the ketchup and place it in the basket": identify the one named target among
  six grocery objects, grasp it, transport it, drop it in a basket; success = a
  relative-bbox containment check against the basket; the five distractors are inert).

## What changed, and why it is strategically different

The seed's plan is **single-target identification + prehensile containment**: find the
named object, grab it, get it inside a container. Nothing about the other objects, their
properties, or their relative placement matters.

This task keeps the LIBERO grocery-tabletop world (a clutter of bottle-like objects and
a basket) but replaces that plan with a **relational arrangement goal over a physical
attribute** — a plan family the seed never touches:

- **No single target.** Every present bottle (3–5 of 5, subset-sampled per episode) is a
  goal object. There is nothing to "identify among distractors" — instead the solver
  must **compare the bottles' heights** (95/125/155/185/215 mm, visually unambiguous)
  and derive an arrangement from the comparison.
- **No containment, no slots, no keying.** The destination is an open display-shelf top
  with **no marked positions**: the goal is a *pairwise order relation* — each taller
  bottle strictly further along the shelf (in the shelf's own body frame) than the next
  shorter one, shortest anchored nearest a green datum post. Where any bottle goes is
  only defined *relative to the others*, and the correct assignment changes with the
  sampled subset and the shelf's random pose (free yaw). A memorized fixed trajectory
  cannot solve it.
- **The seed's container is demoted to tested bait.** The basket is in the scene and
  scores nothing; the seed's own strategy (put bottles in the basket) is negative
  control A at ~0.
- **Reorientation + precision standing placement.** Bottles spawn standing *or* fallen
  (sampled per bottle), so several must be righted; all must end **standing upright,
  cap up, settled** on the shelf — tall thin cylinders with a real ~11.6° topple cliff
  (the calibration probe measures it).

A solver replaying the seed's plan (grab something, drop it in the basket) scores ~0.
The winning plan — compare heights, choose an ordering layout in the shelf frame, right
and place k bottles so the relation holds — does not appear in the seed at all.

### Sibling differentiation (claimed-axis check)

- `track_screwdriver_i12` (size-keyed hanging): correspondence enforced by *physical
  keying to fixed slots*; here there are **no slots and no keying** — order is a pure
  relation between free placements.
- `registry_i28` (musical cans): permutation to *color-matched designated homes* with
  occupancy conflicts and a forced buffer; here there are no homes and no occupancy
  mechanics.
- `libero_pick_milk_i3` (righting a fallen carton): single-object reorientation as the
  terminal goal; here righting is incidental staging, the goal is the multi-object
  **sorted lineup**.
- No sibling claims comparative/sortedness ("arrange by attribute") as an axis.

## Difficulty tier / stages

- **Tier: medium. Declared stage count: 3–5** (one right-and-place stage per present
  bottle, k = 3..5 sampled), bound together by the global ordering relation.
- **Execution order: NOT required** — bottles may be placed in any sequence; only the
  final settled arrangement is judged (the relation itself is about *space*, not time).

## Judging (physical outcomes only)

A bottle **counts** when (shelf body frame): bottom-face center on the shelf top
(x within ±0.28 m, |y| ≤ 0.06 m, bottom within 12 mm of the surface), axis upright cap
up (≤ 10°; the tallest bottle's static tip angle is 11.6°, so a lean past the gate
topples on its own — honest by construction), and settled (|v| < 0.05 m/s,
|ω| < 0.5 rad/s). An **adjacent height-pair counts** when both bottles count and the
taller sits ≥ 20 mm further along shelf +x (two touching bottles are 44 mm apart —
honest margin). Stacking (min bottle 95 mm ≫ 12 mm z-tol), lying placements (upright
gate), and basket placements (bottom at 15 mm above ground, off-shelf) never count.

`score()` = 0.10·(latched ever-placed fraction) + 0.40·(counted-now fraction) +
0.35·(ordered adjacent-pair fraction); **1.0 iff success()** = all present counted and
all pairs ordered; ~0 for doing nothing and for the seed's basket dump. Dry-computed
milestones (k=3/4/5): one placed 0.17/0.13/0.10 → two placed 0.51/0.37/0.29 →
all-placed-one-pair-swapped 0.68/0.73/0.76 → 1.0 (strictly increasing).

Randomization (READBACK-verified): present subset k∈{3..5} + which bottles; shelf xy
jitter + free yaw (judging frame moves); per-bottle standing/fallen rest mode, scatter
arc slot, xy jitter, free yaw; basket side/jitter/yaw. All assets procedural (compound
spawners: cylinder+visual-cap bottles, kinematic shelf with datum post, kinematic
basket).

## Smoke check list (17 checks — acceptance: `SIM_GEN_SMOKE: ALL PASS 17/17`)

1. settle/no-NaN — reset settles finite, score 0, no success.
2. randomization-is-real — shelf pose spread, ≥2 distinct subset patterns, both rest
   modes seen, reset never scores (5 seeded resets, readback).
3. null-policy-fails — 240 idle steps → score ≤ 0.02, no success.
4–6. oracle × 3 seeds — teleport-oracle sorted lineup → success(), score 1.0.
7. rubric monotonicity — 5-stage ladder strictly increasing.
8. ladder final = success at exactly 1.0.
9. near-miss/tolerance — all k bottles counted, one adjacent pair swapped → NOT
   success, score ≤ 0.88.
10. negative A (the seed's own strategy) — all bottles stood inside the basket,
    settled → score ≤ 0.02, no latch, no success.
11. negative B — tallest bottle laid flat at its slot → never counts, no success.
12. negative C — bottle stood on top of another → never counts (z gate).
13. pair-gap tolerance — dx = 5 mm < 20 mm margin: pair rejected; dx = 100 mm: counted.
14. latch persistence — post-solve knock-off: success dies, mid-band score, ever-placed
    latch persists.
15. calibration probe — topple cliff: tilt 0/6° stand, 16/26° fall (analytic 11.6°).
16. calibration — stand-set monotone in tilt.
17. state roundtrip — get_state/set_state restores a success lineup.
