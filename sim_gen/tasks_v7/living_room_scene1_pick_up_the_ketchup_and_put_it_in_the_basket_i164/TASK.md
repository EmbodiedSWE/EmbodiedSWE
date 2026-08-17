# hook_hang_basket — hang the basket on the HIGH wall peg, then load the ketchup into it

`sim_gen` task `living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i164`
— env `simgen.hook_hang_basket` (robot="null"; bodies driven by pose writes / wrenches).

## Seed provenance

Seed: `libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket`
(RoboVerse `roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py`).
The seed is a one-stage pick-and-place: ketchup + three distractor groceries + a passive
open basket sitting on a table; success = ketchup CoM inside the basket's
`contain_region` bounding box. Kept from the seed: the red ketchup bottle, a brown
distractor sauce bottle, the green open basket, and "put the ketchup in the basket" as
the final containment predicate.

## What changed & why it is strategically different

**The receptacle itself must first be suspended.** The basket here has a handle with a
horizontal grab bar, and the goal state is the basket HANGING by that bar from a wall
peg, dangling clear of the floor, with the ketchup inside and everything at rest. The
seed's entire strategy — carry the bottle to a passive, grounded container — is
executed verbatim by smoke check 6 and tops out at the 0.10 containment latch: it is a
*graded distractor*, not a solution. New load-bearing skills:

1. **Hook-hang**: place a 12 mm bar onto a 16 mm round peg so the catch, support
   transfer and pendulum swing-out settle into a stable hung state (contact physics —
   never written).
2. **Metric peg choice (perception-gated)**: hang depth (bar → basket bottom) is
   186 mm; the LOW peg (~130 mm) leaves the basket grounded — geometrically excluded
   from the hang band + floor-clearance + upright gates. WHICH side is high is
   Bernoulli-randomized, so the layout must be read, not memorized.
3. **Deposit into a compliant, swinging container**: the hanging basket is a pendulum;
   a load dropped in swings it and can knock it off the peg — the final state must
   survive its own placement dynamics.
4. **Exclusion**: the brown bbq bottle in the basket (alone or additionally) forfeits
   success.

Distinct from the other examined tasks_v7 packages: **i292** (roll a ball through a
garage: clear-the-blocker + rolling dynamics; no suspension, no receptacle handling),
**i80** (beam-balance counterweight: statics), **i8** (sliding-lid hamper: open →
insert → close articulation ordering), and robobench `pen_holder` (fill a grounded
cup). Nothing else examined hangs the *container* before filling it.

## Scene (procedural only)

- Kinematic wall STAND (board 0.36 × 0.48 m on a base) at x ≈ 0.52 m, yaw ≈ π ± 10°,
  lateral ± 5 cm.
- Two kinematic steel PEGS (r 8 mm, length 156 mm, red retaining stub at the tip),
  re-posed on the board every reset: one HIGH (0.36 ± 0.012 m), one LOW
  (0.13 ± 0.012 m), side Bernoulli-swapped.
- Dynamic green BASKET 0.16 × 0.12 m, rim at 0.11 m, handle bar (12 mm square) at
  0.20 m, mass 0.22 kg, origin/CoM at the floor bottom (hangs level); heavy damping
  models a lossy wicker pendulum (a bar rocking on a round peg is otherwise a
  near-lossless PhysX limit cycle).
- Two dynamic squeeze bottles (55 mm dia): RED ketchup / white cap (target), dark
  BROWN bbq / black cap (distractor), Bernoulli slot-swapped ± jitter.

Randomization verified by READBACK in smoke checks 2–4 (side flips, slot swaps with
bottles always opposite, continuous jitter in peg z / stand yaw / basket xy).

## Rubric (latched; anchored in the demonstrated solution)

Streaks advance only in `post_step` (10 consecutive steps to latch):

| credit | condition |
|---|---|
| 0.25 | `hung_ever`: basket engaged-hung on a peg (bar in the peg-frame window, basket origin in the hang band −0.215..−0.145, z ≥ 0.06, upright ≤ 28°) |
| 0.10 | `in_ever`: ketchup CoM inside the basket cavity (basket-frame, below rim − 15 mm) |
| 0.30 | `loaded_ever`: engaged-hung AND contained simultaneously |
| cap 0.65 | latched base |
| 1.00 | `success()` live: engaged-hung ∧ ketchup contained ∧ bbq NOT contained ∧ basket + ketchup still (lin < 0.05 m/s, ang < 0.7 rad/s) |

## Teleport solution (solve.py) — transport only; interactions are physics

- P0: reset, settle, layout readback (which peg is high, stand yaw, slots). Score 0.000.
- P1: ONE pose write stages the basket in free air, bar 25 mm above the HIGH peg,
  nothing touching. Free fall, bar-on-peg catch, support transfer and pendulum
  settle are contact dynamics. Score 0.250.
- P2: ONE pose write stages the ketchup upright 8 mm above the hanging basket's
  opening (in the basket's CURRENT frame, on the board side so the loaded tilt swings
  the rim away from the board). The drop through the opening, impact, induced swing
  and ring-down are contact dynamics. Score 1.000.
- P3: ≥ 3.3 simulated seconds hands-off; success must persist. Score 1.000.

Measured on the forge: seeds 0 and 1 (high = A) and seed 2 (high = B) all print
non-decreasing `SIM_GEN_SCORE` 0.0000 → 0.2500 → 1.0000 → 1.0000 and
`SIM_GEN_SOLVE: SUCCESS` (~20 s wall each).

## Execution order

Order is NOT required and is declared in `describe()`: hang-empty-then-load (the
demonstrated path) and load-on-the-floor-then-hang-the-loaded-basket are both legal;
the rubric latches either way and `success()` judges only the final physical state.

## Embodiment (single Franka, OSC, base at the origin)

- **Basket**: grasp the 12 mm square handle bar with the parallel jaw (bar top at
  0.206 m when the basket stands on the floor at 0.23–0.29 m radius); carry ~0.24 kg
  well under payload; hook the bar over the high peg from above at ~0.37–0.40 m height,
  0.40–0.45 m radius, then release — the same drop-catch the solve stages. Off-center
  grasps are fine: the peg engages wherever the bar crosses it, and the retaining stub
  blocks sliding off the tip.
- **Ketchup**: 55 mm body < 80 mm jaw opening; top-down or side pinch at 0.20–0.28 m
  radius; lower through the 144 × 88 mm opening of the hanging basket (rim at ~0.27 m
  height under the high peg) and release a few mm above the interior — exactly the
  gentle drop demonstrated.
- All manipulands and goal poses lie in a 0.20–0.55 m radius annulus at 0–0.42 m
  height in front of the base: inside the Franka workspace with standard OSC gains;
  forces are ~2 N scale.

## Checks

- `solve.py`: 3 forge runs (seeds 0, 1, 2 — both high-side branches), each
  `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores and a 3.3 s hands-off hold.
- `smoke.py`: `SIM_GEN_SMOKE: ALL PASS 15/15` on the forge, frames.npz (432 × 600 × 960)
  recorded:
  1. reset settled, finite, score ≤ 0.02
  2. HIGH side flips across seeds (readback)
  3. bottle slots swap; bottles always on opposite slots (readback)
  4. continuous jitter present (peg z, stand yaw, basket xy) (readback)
  5. null policy 2.5 s → score ≤ 0.02, no success
  6. SEED strategy (ketchup into the grounded basket) → 0.10 latch only, not success
  7. LOW-peg decoy (bar hooked on the low peg, basket grounded/leaning) → not
     engaged-hung, ~0 credit
  8. hang alone → 0.25, not success
  9. wrong bottle (bbq) in the hung basket → no containment credit, ≤ 0.26
  10. both bottles inside → exclusion holds, capped ≤ 0.651, never success
  11. success reproduction via the solve recipe → success True, score 1.0
  12. latch regression: ketchup yanked out → success collapses, 0.65 latch persists
  13. latch regression: basket knocked off the peg → same
  14. rejection audit: success never True at any rejection-battery judged point
  15. final no-NaN
