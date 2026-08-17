# capsize_recovery — right the capsized crate over the trapped juice carton, then load it

`simgen.capsize_recovery` · task dir `living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket_i178`

## Seed provenance

Seed: `roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket.py`
— pick the orange juice carton from among seven grocery objects and drop it into a
PASSIVE open basket; `_terminated` is a bbox containment check against the basket's
`contain_region` site.

Kept from the seed: the cast (an orange-juice carton, a same-shaped white milk carton
as the distractor-by-color, an open-top container) and the final act (the orange carton
must end contained in the container).

## What changed

- The container (a green open crate, 16 × 16 × 11.8 cm, procedural) starts **capsized
  — mouth down — on top of the juice carton**. The target is physically caged and
  hidden; the crate is unusable as a receptacle.
- The **load-bearing interaction is righting the crate by rolling it over its ground
  edges**: quarter-roll 1 (capsized → side wall, over a ground rim edge, ~49° balance
  diagonal) simultaneously **frees the carton**; quarter-roll 2 (side → base, over the
  wall/floor-slab edge, ~31° balance diagonal) makes the crate a usable receptacle.
  The crate has an authored CoM at local z = 48 mm, which is what gives the two rolls
  distinct balance diagonals and makes upright terminal (a topple landing cannot vault
  the next edge — the angular-momentum transfer across the face slap is negative).
- Only then does the seed's move exist: lift the freed carton over the 11.8 cm rim and
  lower it in.
- Success is judged on the settled physical state: crate **upright** AND **resting on
  the floor** AND **juice contained** AND **milk NOT contained** AND everything still.
- Anti-cheat geometry the rubric closes: a carton caged under the capsized crate
  already satisfies pure crate-frame containment geometry — therefore containment is
  only credited (and success only possible) with the crate upright and grounded, which
  the smoke battery demonstrates (reset earns 0.00 despite `contained()` = True).

## Strategic difference argument

- **vs the seed**: the seed's plan (grasp target → carry over open basket → release)
  cannot even begin — the target is unreachable under the crate and there is no
  opening to drop anything into. The dominant subtask is re-orienting the receptacle
  itself through a sequence of gravity-fighting edge pivots; grasp-carry-drop is only
  the epilogue.
- **vs the examined corpus**: no examined task rights/rolls a capsized container
  (nearest neighbours: `stack_cups_i92` ends with an enclosure placed mouth-DOWN over
  a target — the opposite reorientation, achieved by a carry; `lift_peg_upright_i116`
  props a lid with a peg — a wedging insertion, not a topple sequence; the basket
  tasks i164/i38 use passive or hangable receptacles that never change orientation).
  A target hidden/caged beneath the receptacle is also novel among examined tasks.

## Teleport solution outline (solve.py)

1. **P0** reset + settle; layout/mass readback; asserts the caged carton's body-frame
   containment earns nothing (score 0.00).
2. **P1 roll 1 (contact)**: world torque about ẑ×d̂ (d̂ = the crate-local face
   direction most aligned with away-from-the-milk), applied every step, rate-capped at
   2.2 rad/s, escalated on stall; **cut at up_z ≥ −0.58** (just past the −0.656
   balance); gravity completes the quarter-roll onto the side wall. Unveils the
   carton. A pure roll torque is frame-drag-immune by construction (rotation since
   reset is about the torque axis itself). Score 0.20.
3. **P2 roll 2 (contact)**: same axis, **cut at up_z ≥ +0.62** (past the +0.515
   balance); the crate topples onto its base and rings down; upright + grounded +
   righted latch. Score 0.50.
4. **P3 deposit (transport teleport)**: ONE pose write stages the freed carton LYING
   30 mm above the upright crate's mouth in the crate's current frame (asserted NOT
   contained at the staging pose), zero velocity; the fall, impact and ring-down are
   contact physics. Score 1.00.
5. **P4** hands-off persistence ≥ 3.3 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS`.

Teleports move only the juice carton through free air; every change of the crate's
pose is torque-driven contact dynamics. Verified on the forge: seeds 0 and 1 both
`SIM_GEN_SOLVE: SUCCESS` (rc=0), scores 0.00 → 0.20 → 0.50 → 1.00, non-decreasing.

## Embodiment argument (single Franka, base at the origin)

- Everything lives in x ∈ [0.2, 0.75], |y| ≤ 0.45 — inside a Franka's ~0.85 m reach
  envelope from a base at the origin.
- **Roll 1 and 2 (crate)**: push HIGH on the crate's upper wall/edge with the closed
  fingertips — a horizontal push at ~0.10–0.12 m height topples the 0.35 kg crate
  over its ground edge (needs ~2.5 N; the arm delivers tens of N). The side-lying
  crate's 8 mm wall also fits the 80 mm jaw for a pinch-assist. The roll direction is
  lateral (away from the milk and from the base), so the crate never translates into
  the robot.
- **Deposit (carton)**: the freed carton is a 50 mm square brick — well within the
  80 mm parallel jaw; lift over the 118 mm rim and lower/release above the mouth,
  exactly the staged drop the solve certifies.

## Execution order declaration

1. `scene.py` — minimal goal predicate + scene first.
2. `solve.py` — iterated on the forge until the goal state was physically reached
   (seeds 0, 1: SUCCESS first attempt; τ = 0.6 N·m, no stall escalation needed).
3. Rubric anchored in the demonstrated solution (latched unveil 0.20 / righted 0.30 /
   loaded 0.15, cap 0.65; 1.0 iff live success).
4. `smoke.py` — rejection battery, run on the forge until `ALL PASS`.

## Check list (smoke.py, 16 checks)

1. Reset sanity: settled, capsized, covered, caged containment earns ~0, no success,
   finite.
2. Randomization readback: crate xy jitter + free yaw spread (8 seeds).
3. Randomization readback: caged carton offset + yaw vary; covered at every seed.
4. Randomization readback: milk side Bernoulli flips + xy jitter.
5. Null policy 2.5 s: score ≤ 0.02, no success.
6. Seed strategy: carton set on the capsized crate's upturned base → NOT contained,
   not success.
7. Side-lying cavity trap: carton inside the sideways cavity → body-frame containment
   True but not upright → no loaded latch, not success.
8. Upright empty crate: unveil+right latches only (~0.50), not success.
9. Wrong carton: milk inside the upright crate → success excluded, no credit.
10. Both cartons inside: capped 0.651, never success.
11. Near-miss: juice standing against the OUTSIDE wall → not contained, not success.
12. Success reproduction: upright crate + the solve's drop recipe → success, 1.0.
13. Latch regression: juice yanked out → success collapses, 0.65 credit persists.
14. Latch regression: crate knocked back capsized → same.
15. Rejection audit: success() never True at any rejection-judged point.
16. Final: no NaN.
