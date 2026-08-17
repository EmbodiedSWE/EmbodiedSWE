# sauce_balance (i253)

## Seed provenance

- Seed id: `libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.py`
- Seed strategy: PICK a freestanding sauce can off a table (disambiguated from
  freestanding distractors by identity), CARRY it through free space, LOWER it into a
  STATIC wooden tray. Success = can centre inside the tray's contain-region bounding
  box. One grasp, one placement, open-loop; the scene never talks back.

## What changed and WHY it is strategically different

Kept from the seed only the opening move — the red sauce can does get picked up and
placed into a brown wooden tray — and demoted it to the near-worthless first step of
a different job: the tray is now one PAN OF AN ANALOG BEAM BALANCE, and the task is
to WEIGH the can.

1. **Placement is the floor, not the goal.** Executing the seed's entire strategy —
   can seated perfectly in the brown tray pan — earns the 0.20 latch and nothing
   else, and the scene *visibly demonstrates* the failure: the loaded beam swings to
   its +12° hard stop by itself. In the seed the same act ends the episode at 1.0.
2. **The success predicate is a physics equilibrium, not a containment box.** Success
   = the hinged beam HOLDS |tilt| ≤ 3.5° for an unbroken 75-substep streak with the
   can in the tray pan and ≥ 1 block in the counter pan. The authored dynamics
   guarantee only a true torque equilibrium can do that: an out-of-equilibrium beam
   forced level falls out of the band in 10 substeps (measured in smoke check 13,
   with the authored K, inertia and damping), and a one-block-grade miss settles at
   ≈ 4.8°, outside the band, with its damped overswing unable to re-enter it.
3. **A hidden, randomized quantity forces a CLOSED LOOP.** The can's mass is sampled
   per episode from {150, 200, 250, 300, 350} g via physx-view mass+inertia writes —
   visually unobservable. The only instrument that can measure it is the balance
   itself, so a solver must iterate: read the tilt SIGN, add the largest untried
   block, remove it on overshoot, repeat (greedy over the {200, 150, 100, 50} g
   block set terminates exactly for every target — subset-sum asserted in the cfg).
   The seed needs zero sensing after the grasp; here perception-act-perceive cycles
   are the task.
4. **Multi-object, order-sensitive, reversible manipulation.** Up to 4 additional
   size/color-coded objects must be placed — stacked, since the blue counter pan is
   deliberately one block wide — and sometimes UN-placed (overshoot recovery is part
   of the certified solution on seed 0). Nothing in the seed is ever removed.

A solver needs a different PLAN (load → iterate weigh-and-correct → hold) and a
different CODE STRUCTURE (spawn-authored revolute joint with hard stops, trimmed
pendulum beam with keel, per-episode physx mass writes, streak-gated equilibrium
rubric) — not different parameters of pick-and-place. Differentiation from the
same-seed siblings: i166 (sauce_chute) is captive-object mechanism actuation with a
container carried twice; i225 (tilt_dispenser family) presses/holds a mechanism.
Here the mechanism is never actuated by the hand at all — it is a passive
INSTRUMENT that is only ever loaded, and its READOUT drives the plan.

## Teleport-solution outline (solve.py)

All teleports are transport-only (one free object carried across free space, set
down 3 mm above its rest, velocities zeroed); every load-bearing outcome is
hands-off joint + contact physics. The can's sampled mass is NEVER read.

- **P0** reset, settle; assert the empty beam level (trim is real), can on the
  floor, score exactly 0.
- **P1 LOAD (transport teleport)**: the can set down into the brown tray pan while
  the beam is level; the beam then swings to its +12° stop BY ITSELF (asserted) —
  the seed strategy's end state, worth only the 0.20 latch.
- **P2 WEIGH (closed loop on the tilt readback ONLY)**: repeat — settle (min-dwell
  + 30-step quiet-hinge streak), read tilt; tray-side down → set the largest
  untried block onto the blue-pan stack top (beam-frame aligned drop); counter-side
  down → carry the last block back to its old floor slot, never retry it; |tilt|
  inside the band → stop. Certified runs: seed 0 = add 200, add 150, overshoot to
  −5.4°, REMOVE 150, add 100, balanced at −0.33° (can was 300 g).
- **P3 HOLD**: hands off until the 75-substep level streak fills; success()
  asserted, score 1.0.
- **P4** ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS` only if
  success() still holds. `SIM_GEN_SCORE` printed at every phase boundary,
  non-decreasing (0.00 → 0.20 → 1.00 → 1.00 → 1.00). Passes on seeds 0 and 1
  (forge logs).

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: arm base at (0, −0.75, 0). The balance stand spawns at the
origin (±3 cm, yaw-jittered, pan side flipped 0/180°); the object scatter lane is at
y ≈ −0.34, x ∈ [−0.31, 0.31]. Pans sit at radius 0.24 from the hinge, z ≈ 0.24
(floor of the wells) — everything actionable lies in a 0.35–0.80 m frontal band at
comfortable heights, nothing above 0.30 m.

- **Red can (Ø 60 × 115 mm, 150–350 g)**: side pinch at mid-height — 60 mm across
  the 80 mm jaw span; 350 g worst case is a trivial payload. Set-down into the
  brown pan tolerates ±15 mm xy (the pan is 70 mm square around a 60 mm can and the
  rubric window is cfg-asserted to accept every in-pan rest); the pan sits at the
  outer end of the beam with open sky above it — a straight top-down lower.
- **Weight blocks (47/43/37/30 mm cubes, 200/150/100/50 g)**: top pinch across two
  faces — every edge is under the 75 mm effective opening with ≥ 28 mm of finger
  room; density-coded sizes make them visually distinguishable. Stacking into the
  one-block-wide blue pan is a centred drop with ±20 mm tolerance; block faces need
  only rough yaw alignment to the pan (9 mm total clearance on the largest block —
  the certified drops use exactly this alignment). Removal is the same pinch in
  reverse; the top of the stack is always exposed.
- **Reading the scale needs no touch.** The tilt sign is visible from the beam
  posture (±12° at the stops is a 10 cm end-to-end height difference); the arm only
  ever touches the can and the blocks. The beam swings BETWEEN interactions — the
  hand is never near the mechanism while it moves, and the pans present open tops
  at every tilt (≤ 12° never occludes a drop).
- **Gentle set-downs suffice**: 3 mm drops; the pans' 55 mm walls retain their
  contents through every certified swing (24° end-to-end worst case).

## Execution order

Physically forced: the can must be loaded before any tilt readback means anything —
an empty-beam tilt is 0 and the rubric gates ALL counterweight/level credit on the
can being seated (a pre-loaded blue pan pegs the beam the other way and earns
nothing, smoke check 7). The weigh loop's order is forced by information, not fiat:
which subset is correct is unknowable without iterating on the readback. Blocks are
removable at any time; nothing else is sequenced. describe() states the decision
rule (tray down = add, blue down = remove-and-try-smaller) explicitly.

## Rubric

success(): unbroken 75-substep streak (0.63 s) of |tilt| ≤ 3.5° with, at every step
of it, the can seated upright inside the brown pan window (beam-frame xy ± 15 mm,
z band rejects rim perches, upright ≤ 20° in the pan frame) and ≥ 1 block inside
the blue pan window — revoked live the moment any part breaks.
score(): 0.20 loaded (10-substep seated streak) + 0.15 counterweighting begun
(gated on loaded) + 0.25 near-level (|tilt| ≤ 8° sustained 20, gated on load AND
counterweight present) + 0.25 level (the success streak itself) — all latched
rising-only, exactly 0 for doing nothing, cap 0.85; 1.0 iff success() holds live.

## Check list (smoke.py — rejection battery, 15 checks)

1. settle/no-NaN: empty beam LEVEL, can + blocks on the floor, still.
2. reset score ≈ 0, no success.
3. randomization: stand xy/yaw and the floor-slot shuffle vary (readback, 8 seeds).
4. randomization: the can's hidden MASS varies (physx get_masses readback).
5. null policy: 240 idle steps → score ≈ 0, no success, beam stays level (a level
   beam alone earns nothing).
6. SEED-STRATEGY end state: can seated in the brown pan, nothing else → the beam
   pegs at its stop by itself; score ≤ 0.201, no success.
7. counterweight without a load: blocks in the blue pan, can on the floor → beam
   pegs counter-side, score ≈ 0 (all credit is load-gated).
8. can not presented: can lying sideways across the tray pan → upright check
   rejects, no load credit.
9. one granule short: exact subset MINUS 50 g settled → ≈ +4.8°, outside the level
   band, level never latches, score ≤ 0.601, no success.
10. latched credit survives the can being yanked back out; still no success.
11. one granule over: exact subset PLUS 50 g → counter-side down, same cap, no
    success.
12. anti-prop: the full 4-block floor tower under the sunken tray pan reaches no
    beam underside — the beam stays pegged, no extra credit.
13. held-level fake: beam + can + WRONG counterweight (the 50 g block alone, a
    ≥ 100 g imbalance for every can mass) teleported level in one coherent burst,
    hands off → the streak dies in 10 substeps (measured) and the beam falls back
    out of the band; success never fires (the streak is an equilibrium proof).
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.
