# reach_and_drag_i34 — compass_crate ("pivot it, don't ship it")

**Scene:** `simgen.compass_crate` (robobench scene-level task, `robot="null"`)
**Seed:** `rlbench/reach_and_drag`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/reach_and_drag.py`)
**Tier:** easy — **1 stage** (a single reorientation skill), difficulty carried by the
aiming precision (15 deg cone) and the anti-transport constraint.
**Execution order:** none required (single stage; any rotation route that ends aimed,
upright, settled and un-strayed counts).

## What the seed does

Franka reaches a stick, grasps it, and uses it to DRAG a cube across the table onto a
distant target zone. The whole task is *transporting an object to an absolute goal
region*, with the stick as a reach-extension tool.

## What this task does instead

A 12 cm crate with a bright red badge on ONE face rests on the ground; a tall orange
beacon post stands ~0.5 m away at a random bearing (initial aiming error sampled in
[100, 175] deg, random sign); the seed's stick lies nearby as an optional free tool.
The goal is to rotate the crate **in place** about the vertical axis until the badge
face points at the beacon (within 15 deg), leaving it upright, settled, at its spawn
spot. A permanent `strayed` latch trips if the crate center ever leaves an 18 cm radius
around its spawn and caps the score at 0.05 forever — returning the crate home does not
un-trip it.

## Why strategically different (not just different numbers)

- **Goal type inverted:** the seed's goal is a *position* (get the cube into a zone);
  here position must be *conserved* and the goal is an *orientation* (aim a marked face
  at a landmark). The seed has no orientation term at all; here translation earns
  nothing and, past 18 cm, permanently spoils the episode.
- **The seed's plan is the tested failing control:** "drag the object to the visible
  landmark" — expressed as a kinematic incremental drag to the beacon — trips the stray
  latch: score capped at 0.05, success impossible, even after aligning at the beacon or
  dragging the crate back home and aligning there (latch permanence is smoke-tested).
- **Different skill for a solver:** controlled pivoting/regrasping/nudging about a
  vertical axis with a bearing read from the scene, instead of tool-mediated planar
  transport. The stick remains usable (as a pusher to pivot the crate), so the tool
  affordance survives while the plan built on it does not.
- **Sibling differentiation:** no sibling claims an orientation-only / anti-transport
  goal. base_i25 (ordered toppling) uses an anti-*carry* latch but its goal is knocking
  pillars over; nothing here topples, transits an aperture (i1/i14), balances (i23), or
  places onto anything.

## Physical honesty

The oracle rotates the crate kinematically in ~12 deg increments (center pinned, zero
velocities) and then releases it: success()/score() judge the **settled physical pose**
(upright cone, resting height on the ground, velocity gates, in-place radius). Partial
credit uses a *latched best aiming error* that only latches while the crate is upright,
in place and slow (lin < 0.5 m/s, ang < 1.5 rad/s) — a wild spin flying past the right
heading does not latch, and a carried crate latches nothing.

## Rubric

- `score = 0.8 * clamp((90deg - best_err) / (90deg - 15deg), 0, 1)`; exactly **1.0 iff
  success**; capped at **0.05** forever once strayed. Doing nothing scores exactly 0
  (initial error >= 100 deg, credit anchor at 90 deg).
- `success()` = upright (10 deg) & in place (10 cm) & grounded & settled & aimed
  (15 deg) & never strayed.

## Randomization (verified by readback in the smoke)

Crate spawn xy (+/-5 cm) and yaw (+/-180 deg); beacon bearing (via the sampled initial
error band [100, 175] deg, random sign) and distance (0.42–0.58 m); stick pose jitter.

## Smoke check list (19)

1. settle/no-NaN: reset finite, upright, grounded, at rest
2. score ~0 at reset, no success
3. randomization is real (readback: yaw, beacon position, error band)
4. null policy fails (240 idle steps -> score ~0)
5–7. oracle reaches success() + score 1.0 on 3 seeds
8. monotonicity: score strictly increases along an 80→50→25→8 deg aim ladder
9. monotonicity: partials < 1.0; final inside tolerance is success
10. negative A (seed strategy): drag-to-beacon trips the stray latch
11. negative A: aligned at the beacon stays capped (<= 0.05, no success)
12. negative A: latch permanence — dragged back home + aligned still fails
13. negative B: badge aimed away scores 0
14. negative C: tipped-but-aimed crate rejected by the upright gate
15. negative C recovery: righted in place -> success
16. near-miss: 23 deg off -> partial credit only, no success
17. near-miss recovery: finishing the rotation -> success
18. calibration: success cliff at the 15 deg tolerance (5,10 in; 20,25,40,70 out)
19. calibration: score monotone non-increasing with final aiming error
